import {
  If,
  Return,
  builtin,
  compileModule,
  compileModuleJs,
  externFn,
  f32,
  f32T,
  fn,
  module,
  packModule,
  storageBuffer,
  uniformStruct,
  u32T,
  vec3uT,
  voidT,
  type CpuModule,
  type ModuleDecl,
  type Pack,
  type ReadonlyNode,
  type StorageBuffer,
} from "typeshade";
import {
  ELEMENTWISE_FUSION_REGISTRY,
  type ElementwiseFusionRegistry,
  type ElementwiseFusionOperation,
} from "./kernel-registry.js";
import type {
  ElementwiseFusionCandidate,
  ElementwiseFusionValueRef,
} from "./elementwise-fusion.js";

export const ELEMENTWISE_FUSION_ENTRY = "fusedElementwiseKernel";
const WORKGROUP_SIZE = 64;

export interface ElementwiseFusionProgram {
  readonly signature: string;
  /** The one graph-derived TypeShade declaration shared by CPU and GPU. */
  readonly module: ModuleDecl;
  readonly entry: typeof ELEMENTWISE_FUSION_ENTRY;
  readonly inputNames: readonly string[];
}

export interface ElementwiseFusionCpuProgram extends ElementwiseFusionProgram {
  /** Emitted CPU module, with interpreter fallback only for CSP eval rejection. */
  readonly cpu: CpuModule;
  readonly cpuBackend: "emitted-js" | "csp-interpreter-fallback";
}

export interface ElementwiseFusionGpuProgram extends ElementwiseFusionProgram {
  readonly pack: Pack;
}

interface CachedProgram {
  readonly shared: ElementwiseFusionProgram;
  cpu?: ElementwiseFusionCpuProgram;
  pack?: Pack;
}

const programs = new Map<string, CachedProgram>();

function isCspEvalFailure(error: unknown): boolean {
  return (
    (typeof EvalError !== "undefined" && error instanceof EvalError) ||
    (error instanceof Error &&
      /unsafe-eval|code generation from strings disallowed/i.test(
        error.message,
      ))
  );
}

function helperDeclaration(
  operation: ElementwiseFusionOperation,
  registry: ElementwiseFusionRegistry,
) {
  const declaration = registry.helpers.get(operation.helperName);
  if (!declaration)
    throw new Error(
      `TypeShade operation helper was not generated: ${operation.helperName}.`,
    );
  return declaration;
}

function helperCall(
  operation: ElementwiseFusionOperation,
  registry: ElementwiseFusionRegistry,
) {
  const declaration = helperDeclaration(operation, registry);
  const output = declaration.params[0];
  if (
    !output ||
    output.mode !== "inout" ||
    output.type.kind !== "scalar" ||
    output.type.scalar !== "f32"
  )
    throw new Error(
      `TypeShade operation helper ${operation.helperName} has no validated @inout f32 output.`,
    );
  const expectedInputs = operation.helperInputNames;
  if (declaration.params.length !== expectedInputs.length + 1)
    throw new Error(
      `TypeShade operation helper ${operation.helperName} has an unexpected parameter count.`,
    );

  const params: Record<string, typeof f32T> = { [output.name]: f32T };
  for (let index = 0; index < expectedInputs.length; index++) {
    const parameter = declaration.params[index + 1];
    if (
      !parameter ||
      parameter.name !== expectedInputs[index] ||
      parameter.mode !== undefined ||
      parameter.type.kind !== "scalar" ||
      parameter.type.scalar !== "f32"
    )
      throw new Error(
        `TypeShade operation helper ${operation.helperName} input signature no longer matches its registered ports.`,
      );
    params[parameter.name] = f32T;
  }
  if (declaration.ret.kind !== "void")
    throw new Error(
      `TypeShade operation helper ${operation.helperName} must return void.`,
    );
  return externFn(declaration.name, params, voidT);
}

function resolveValue(
  reference: ElementwiseFusionValueRef,
  inputs: StorageBuffer<ReadonlyNode<"f32">>[],
  values: ReadonlyNode<"f32">[],
  index: ReadonlyNode<"u32">,
): ReadonlyNode<"f32"> {
  if (reference.kind === "leaf") {
    const input = inputs[reference.slot];
    if (!input)
      throw new Error(
        `Fusion graph references missing input slot ${reference.slot}.`,
      );
    return input.at(index);
  }
  const value = values[reference.id];
  if (!value)
    throw new Error(
      `Fusion graph references a value before it is defined: ${reference.id}.`,
    );
  return value;
}

function buildModule(
  candidate: ElementwiseFusionCandidate,
  registry: ElementwiseFusionRegistry,
): ModuleDecl {
  const shape = uniformStruct(
    "FusionShape",
    { group: 0, binding: 0, as: "shape" },
    { length: u32T },
  );
  const inputNames = candidate.leaves.map((_, slot) => `input${slot}`);
  const inputs = inputNames.map((name, slot) =>
    storageBuffer(name, f32T, {
      group: 0,
      binding: slot + 1,
      access: "read",
    }),
  );
  const output = storageBuffer("output", f32T, {
    group: 0,
    binding: inputs.length + 1,
    access: "read_write",
  });
  const operations = new Map(
    candidate.steps.map((step) => [step.operation.helperName, step.operation]),
  );
  const helpers = [...operations.values()].map((operation) =>
    helperDeclaration(operation, registry),
  );
  const calls = new Map(
    [...operations.values()].map((operation) => [
      operation.helperName,
      helperCall(operation, registry),
    ]),
  );

  const kernel = fn(
    ELEMENTWISE_FUSION_ENTRY,
    { gid: builtin("global_invocation_id", vec3uT) },
    voidT,
    ({ gid }, builder) => {
      If(gid.x.ge(shape.field.length), () => Return());
      const values: ReadonlyNode<"f32">[] = [];
      for (const step of candidate.steps) {
        const definition = calls.get(step.operation.helperName);
        if (!definition)
          throw new Error(
            `No call handle was created for ${step.operation.helperName}.`,
          );
        const result = builder.var(`value${step.id}`, f32T, f32(0));
        const arguments_ = step.inputs.map((reference) =>
          resolveValue(reference, inputs, values, gid.x),
        );
        builder.call(definition(result, ...arguments_));
        values[step.id] = result;
      }
      output
        .at(gid.x)
        .assign(resolveValue(candidate.outputValue, inputs, values, gid.x));
    },
    { stage: "compute", workgroupSize: WORKGROUP_SIZE },
  );

  return module({
    uses: [shape, ...inputs, output],
    funcs: [...helpers, kernel],
  });
}

/** Build/cache the authoritative ModuleDecl lowered from registered graph operations. */
export function elementwiseFusionProgram(
  candidate: ElementwiseFusionCandidate,
  registry: ElementwiseFusionRegistry = ELEMENTWISE_FUSION_REGISTRY,
): ElementwiseFusionProgram {
  const signature = `${registry.fingerprint}:${candidate.signature}:workgroup=${WORKGROUP_SIZE}`;
  const cached = programs.get(signature);
  if (cached) return cached.shared;

  const sharedModule = buildModule(candidate, registry);
  const shared: ElementwiseFusionProgram = {
    signature,
    module: sharedModule,
    entry: ELEMENTWISE_FUSION_ENTRY,
    inputNames: candidate.leaves.map((_, slot) => `input${slot}`),
  };
  programs.set(signature, { shared });
  return shared;
}

/** Compile CPU code only when the graph is executed on the CPU backend. */
export function elementwiseFusionCpuProgram(
  candidate: ElementwiseFusionCandidate,
  registry: ElementwiseFusionRegistry = ELEMENTWISE_FUSION_REGISTRY,
): ElementwiseFusionCpuProgram {
  const shared = elementwiseFusionProgram(candidate, registry);
  const cached = programs.get(shared.signature)!;
  if (cached.cpu) return cached.cpu;

  let cpu: CpuModule;
  let cpuBackend: ElementwiseFusionCpuProgram["cpuBackend"] = "emitted-js";
  try {
    cpu = compileModuleJs(shared.module, { precision: "f32" });
  } catch (error) {
    if (!isCspEvalFailure(error)) throw error;
    cpu = compileModule(shared.module, { precision: "f32" });
    cpuBackend = "csp-interpreter-fallback";
  }
  const program: ElementwiseFusionCpuProgram = {
    ...shared,
    cpu,
    cpuBackend,
  };
  cached.cpu = program;
  return program;
}

/** Pack the same ModuleDecl only when the graph is executed on the GPU backend. */
export function elementwiseFusionGpuProgram(
  candidate: ElementwiseFusionCandidate,
  registry: ElementwiseFusionRegistry = ELEMENTWISE_FUSION_REGISTRY,
): ElementwiseFusionGpuProgram {
  const shared = elementwiseFusionProgram(candidate, registry);
  const cached = programs.get(shared.signature)!;
  if (!cached.pack) cached.pack = packModule(shared.module);
  return { ...shared, pack: cached.pack };
}

/** Execute the generated or interpreter CpuModule using the public number[] bindings. */
export function dispatchElementwiseFusionCpu(
  cpu: CpuModule,
  inputNames: readonly string[],
  leaves: readonly Float32Array[],
  length: number,
): Float32Array {
  if (inputNames.length !== leaves.length)
    throw new Error("Fused CPU bindings do not match graph leaf slots.");

  for (let slot = 0; slot < leaves.length; slot++)
    cpu.setBinding(inputNames[slot]!, Array.from(leaves[slot]!));

  const output = new Array<number>(length).fill(0);
  cpu.setBinding("shape", { length });
  cpu.setBinding("output", output);
  if (length > 0)
    cpu.dispatch(ELEMENTWISE_FUSION_ENTRY, [
      Math.ceil(length / WORKGROUP_SIZE),
      1,
      1,
    ]);
  return Float32Array.from(output);
}
