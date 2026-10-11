import {
  If,
  Return,
  builtin,
  compileModule,
  compileModuleJs,
  f32,
  f32T,
  fn,
  max,
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
import type {
  ElementwiseFusionCandidate,
  ElementwiseFusionExpression,
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

function expressionSignature(expression: ElementwiseFusionExpression): string {
  if (expression.kind === "input") return `i${expression.slot}`;
  if (expression.kind === "relu")
    return `relu(${expressionSignature(expression.input)})`;
  return `${expression.kind}(${expressionSignature(expression.left)},${expressionSignature(expression.right)})`;
}

function isCspEvalFailure(error: unknown): boolean {
  return (
    (typeof EvalError !== "undefined" && error instanceof EvalError) ||
    (error instanceof Error &&
      /unsafe-eval|code generation from strings disallowed/i.test(
        error.message,
      ))
  );
}

function lowerExpression(
  expression: ElementwiseFusionExpression,
  inputs: StorageBuffer<ReadonlyNode<"f32">>[],
  index: ReadonlyNode<"u32">,
): ReadonlyNode<"f32"> {
  if (expression.kind === "input") return inputs[expression.slot]!.at(index);
  if (expression.kind === "relu")
    return max(lowerExpression(expression.input, inputs, index), f32(0));
  const left = lowerExpression(expression.left, inputs, index);
  const right = lowerExpression(expression.right, inputs, index);
  return expression.kind === "add" ? left.add(right) : left.mul(right);
}

function buildModule(candidate: ElementwiseFusionCandidate): ModuleDecl {
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
  const kernel = fn(
    ELEMENTWISE_FUSION_ENTRY,
    { gid: builtin("global_invocation_id", vec3uT) },
    voidT,
    ({ gid }) => {
      If(gid.x.ge(shape.field.length), () => Return());
      output
        .at(gid.x)
        .assign(lowerExpression(candidate.expression, inputs, gid.x));
    },
    { stage: "compute", workgroupSize: WORKGROUP_SIZE },
  );

  return module({ uses: [shape, ...inputs, output], funcs: [kernel] });
}

/** Build/cache the authoritative ModuleDecl lowered from the walked Tensor expression. */
export function elementwiseFusionProgram(
  candidate: ElementwiseFusionCandidate,
): ElementwiseFusionProgram {
  const signature = `f32:${WORKGROUP_SIZE}:${expressionSignature(candidate.expression)}`;
  const cached = programs.get(signature);
  if (cached) return cached.shared;

  const sharedModule = buildModule(candidate);
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
): ElementwiseFusionCpuProgram {
  const shared = elementwiseFusionProgram(candidate);
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
): ElementwiseFusionGpuProgram {
  const shared = elementwiseFusionProgram(candidate);
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
