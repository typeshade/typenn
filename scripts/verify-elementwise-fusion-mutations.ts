import { mkdir, mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";

type Check = (
  id: string,
  label: string,
  assertion: () => void | Promise<void>,
) => Promise<void>;

interface MutationAssertions {
  readonly moduleHelpers: (
    module: unknown,
    declarations: readonly unknown[],
  ) => void;
  readonly storage: (program: unknown, expectedInputCount: number) => void;
  readonly noCodegenFallback: (report: unknown, entry: string) => void;
  readonly output: (
    actual: ArrayLike<number>,
    expected: ArrayLike<number>,
    label: string,
  ) => void;
  readonly storageLimit: (events: readonly string[]) => void;
  readonly freshData: (
    actual: ArrayLike<number>,
    expected: ArrayLike<number>,
    label: string,
  ) => void;
  readonly diamondOutput: (
    actual: ArrayLike<number>,
    expected: ArrayLike<number>,
    label: string,
  ) => void;
  readonly diamondLocals: (module: unknown) => void;
  readonly emittedCpuExecution: (program: unknown, dispatches: number) => void;
}

export interface FusionMutationContext {
  readonly check: Check;
  readonly baseCandidate: any;
  readonly baseProgram: any;
  readonly expectedHelpers: readonly any[];
  readonly helperFingerprint: string;
  readonly changedSourceExpected: ArrayLike<number>;
  readonly f32Candidate: any;
  readonly f32Inputs: readonly Float32Array[];
  readonly f32Expected: ArrayLike<number>;
  readonly freshCandidate: any;
  readonly freshExpected: ArrayLike<number>;
  readonly diamondCandidate: any;
  readonly diamondExpected: ArrayLike<number>;
  readonly cspMutationOutput: any;
  readonly cspMutationExpected: ArrayLike<number>;
  readonly staleInputs: readonly Float32Array[];
  readonly createRegistry: (input: {
    readonly declarations: readonly any[];
    readonly fingerprint: string;
  }) => any;
  readonly generateElementwiseOpIr: (options: {
    readonly sourceOverrides: Readonly<Record<string, string>>;
    readonly outputDirectory: string;
    readonly scratchDirectory: string;
  }) => Promise<{
    readonly declarations: readonly any[];
    readonly fingerprint: string;
  }>;
  readonly elementwiseFusionProgram: (candidate: any, registry?: any) => any;
  readonly elementwiseFusionCpuProgram: (candidate: any, registry?: any) => any;
  readonly dispatchElementwiseFusionCpu: (
    cpu: any,
    inputNames: readonly string[],
    leaves: readonly Float32Array[],
    length: number,
  ) => Float32Array;
  readonly evaluateCpuWithStats: (tensor: any) => Promise<any>;
  readonly compileModuleJs: (
    module: unknown,
    options: { readonly precision: "f32" | "f64" },
  ) => any;
  readonly generateModuleJs: (
    module: unknown,
    options: { readonly precision: "f32" },
  ) => any;
  readonly GpuGraphExecutor: new (...args: any[]) => any;
  readonly DeviceBuffers: new () => any;
  readonly assertions: MutationAssertions;
}

function expectPredicateToReject(
  predicate: () => void,
  mutation: string,
  expectedDiagnostic: string,
): void {
  try {
    predicate();
  } catch (error) {
    if (
      !(error instanceof Error) ||
      !error.message.includes(expectedDiagnostic)
    )
      throw new Error(
        `${mutation} failed for an unexpected reason; expected diagnostic ` +
          `${JSON.stringify(expectedDiagnostic)}, received ${String(error)}.`,
        { cause: error },
      );
    return;
  }
  throw new Error(
    `${mutation} survived its normal acceptance assertion (${expectedDiagnostic}).`,
  );
}

function cloneFunctionDeclaration(func: any): any {
  const declaration = typeof func === "function" ? func.decl : func;
  if (!declaration || !Array.isArray(declaration.body))
    throw new Error(
      "Mutation requires an actual TypeShade function declaration.",
    );
  return { ...declaration, body: structuredClone(declaration.body) };
}

function cloneModule(module: any): any {
  return {
    ...module,
    funcs: module.funcs.map(cloneFunctionDeclaration),
    bindings: module.bindings.map((binding: any) => ({ ...binding })),
  };
}

function projectProgramWithClonedModule(program: any): any {
  return { ...program, module: cloneModule(program.module) };
}

function projectRootScratchDirectory(prefix: string): Promise<string> {
  const root = path.resolve(".typenn-build", "fusion-mutations");
  return mkdir(root, { recursive: true }).then(() =>
    mkdtemp(path.join(root, `${prefix}-`)),
  );
}

function addRawStatementToHelper(module: any): any {
  const mutated = cloneModule(module);
  const helper = mutated.funcs.find(
    (func: any) => func.name === "typennAddF32",
  );
  if (!helper)
    throw new Error("FI04 requires the real add helper in the module.");
  helper.body = [...helper.body, { s: "raw", wgsl: "" }];
  return mutated;
}

function redirectDiamondRightMulIntoSharedLocal(
  module: any,
  entryName: string,
): any {
  const mutated = cloneModule(module);
  const entry = mutated.funcs.find((func: any) => func.name === entryName);
  if (!entry) throw new Error("FI02 requires the real T05 compute entry.");

  let redirects = 0;
  const visit = (value: any): void => {
    if (Array.isArray(value)) {
      for (const item of value) visit(item);
      return;
    }
    if (!value || typeof value !== "object") return;

    if (
      value.s === "call" &&
      value.expr?.op === "call" &&
      value.expr.fn === "typennMulF32" &&
      value.expr.args?.[0]?.op === "varref" &&
      value.expr.args[0].name === "value2"
    ) {
      const args = [...value.expr.args];
      args[0] = { ...args[0], name: "value0" };
      value.expr = { ...value.expr, args };
      redirects++;
    }

    for (const child of Object.values(value)) visit(child);
  };
  visit(entry.body);
  if (redirects !== 1)
    throw new Error(
      `FI02 expected one real right-branch mul output target, found ${redirects}.`,
    );
  return mutated;
}

/**
 * Apply one real TypeNN/TypeShade-path mutation per acceptance assertion. A fault
 * only counts as killed when the same normal predicate rejects its mutated value;
 * failures while constructing a mutation are allowed to escape and fail the test.
 */
export async function verifyFusionMutations(
  context: FusionMutationContext,
): Promise<void> {
  const {
    check,
    baseCandidate,
    baseProgram,
    freshCandidate,
    diamondCandidate,
    diamondExpected,
    cspMutationOutput,
    cspMutationExpected,
    expectedHelpers,
    helperFingerprint,
    assertions,
  } = context;

  await check(
    "FI01-helper-provenance-mutation",
    "source provenance rejects changed helper body",
    async () => {
      const addPath = path.resolve("src/kernels/add.shade.ts");
      const source = await readFile(addPath, "utf8");
      const original = "result = left + right;";
      if (source.split(original).length !== 2)
        throw new Error("FI01 expected exactly one production add assignment.");
      const changedSource = source.replace(original, "result = left - right;");
      const scratchDirectory = await projectRootScratchDirectory("fi01");
      const outputDirectory = await mkdtemp(
        path.join(tmpdir(), "typenn-fusion-fi01-output-"),
      );
      try {
        const generated = await context.generateElementwiseOpIr({
          sourceOverrides: { add: changedSource },
          outputDirectory,
          scratchDirectory,
        });
        const changedRegistry = context.createRegistry(generated);
        const changedProgram = context.elementwiseFusionProgram(
          baseCandidate,
          changedRegistry,
        );
        expectPredicateToReject(
          () =>
            assertions.moduleHelpers(changedProgram.module, expectedHelpers),
          "FI01 changed source helper",
          "Composed module must preserve source FuncDecl typennAddF32",
        );
      } finally {
        await rm(scratchDirectory, { recursive: true, force: true });
        await rm(outputDirectory, { recursive: true, force: true });
      }
    },
  );

  await check(
    "FI02-shared-local-clobber-mutation",
    "T05 assertions reject a branch writing into its shared input local",
    () => {
      const diamondProgram = context.elementwiseFusionProgram(diamondCandidate);
      const mutatedModule = redirectDiamondRightMulIntoSharedLocal(
        diamondProgram.module,
        diamondProgram.entry,
      );
      const cpu = context.compileModuleJs(mutatedModule, {
        precision: "f32",
      });
      const inputs = diamondCandidate.leaves.map((leaf: any) => {
        if (leaf.node.kind !== "input")
          throw new Error("FI02 T05 candidate has a non-input leaf.");
        return leaf.node.value;
      });
      const actual = context.dispatchElementwiseFusionCpu(
        cpu,
        diamondProgram.inputNames,
        inputs,
        inputs[0]?.length ?? 0,
      );
      expectPredicateToReject(
        () => assertions.diamondOutput(actual, diamondExpected, "diamond"),
        "FI02 shared-local numeric mutation",
        "diamond[",
      );
      expectPredicateToReject(
        () => assertions.diamondLocals(mutatedModule),
        "FI02 shared-local output-target mutation",
        "each operation writes to its own fresh local",
      );
    },
  );

  await check(
    "FI03-intermediate-storage-mutation",
    "storage assertion rejects an added intermediate binding",
    () => {
      const mutatedProgram = projectProgramWithClonedModule(baseProgram);
      const outputBinding = mutatedProgram.module.bindings.find(
        (binding: any) =>
          binding.space === "storage" && binding.name === "output",
      );
      if (!outputBinding)
        throw new Error("FI03 requires the real output storage binding.");
      mutatedProgram.module.bindings.push({
        ...outputBinding,
        name: "intermediate",
      });
      expectPredicateToReject(
        () => assertions.storage(mutatedProgram, baseCandidate.leaves.length),
        "FI03 intermediate binding",
        "storage is inputs plus final output",
      );
    },
  );

  await check(
    "FI04-real-codegen-fallback-mutation",
    "pinned TypeShade and CPU execution assertions reject fallback paths",
    async () => {
      const mutatedModule = addRawStatementToHelper(baseProgram.module);
      const report = context.generateModuleJs(mutatedModule, {
        precision: "f32",
      });
      expectPredicateToReject(
        () => assertions.noCodegenFallback(report, baseProgram.entry),
        "FI04 raw helper fallback",
        "TypeShade emitted every module function",
      );

      const originalFunction = globalThis.Function;
      let functionConstructionAttempts = 0;
      globalThis.Function = new Proxy(originalFunction, {
        apply() {
          functionConstructionAttempts++;
          throw new EvalError("FI04 simulated CSP eval prohibition.");
        },
        construct() {
          functionConstructionAttempts++;
          throw new EvalError("FI04 simulated CSP eval prohibition.");
        },
      });
      let execution: any;
      try {
        execution = await context.evaluateCpuWithStats(cspMutationOutput);
      } finally {
        globalThis.Function = originalFunction;
      }

      if (functionConstructionAttempts === 0)
        throw new Error(
          "FI04 CSP mutation did not intercept Function creation.",
        );
      if (execution.fusionProgram?.cpuBackend !== "csp-interpreter-fallback")
        throw new Error(
          `FI04 CSP mutation did not select the interpreter fallback: ${String(execution.fusionProgram?.cpuBackend)}.`,
        );
      assertions.output(
        execution.value,
        cspMutationExpected,
        "FI04 CSP fallback oracle",
      );
      expectPredicateToReject(
        () =>
          assertions.emittedCpuExecution(
            execution.fusionProgram,
            execution.dispatches,
          ),
        "FI04 CSP fallback backend mutation",
        "CPU backend is emitted-js",
      );
    },
  );

  await check(
    "FI05-no-f32-rounding-mutation",
    "f32 oracle assertion rejects actual f64-precision module output",
    () => {
      const f32Program = context.elementwiseFusionProgram(context.f32Candidate);
      const mutant = context.compileModuleJs(f32Program.module, {
        precision: "f64",
      });
      const actual = context.dispatchElementwiseFusionCpu(
        mutant,
        f32Program.inputNames,
        context.f32Inputs,
        context.f32Inputs[0]?.length ?? 0,
      );
      expectPredicateToReject(
        () => assertions.output(actual, context.f32Expected, "FI05 f64 mutant"),
        "FI05 no-f32-rounding mutant",
        "FI05 f64 mutant[",
      );
    },
  );

  await check(
    "FI06-stale-helper-fingerprint-mutation",
    "T02 output assertion rejects stale-key cached helper",
    async () => {
      const addPath = path.resolve("src/kernels/add.shade.ts");
      const source = await readFile(addPath, "utf8");
      const original = "result = left + right;";
      if (source.split(original).length !== 2)
        throw new Error("FI06 expected exactly one production add assignment.");
      const scratchDirectory = await projectRootScratchDirectory("fi06");
      const outputDirectory = await mkdtemp(
        path.join(tmpdir(), "typenn-fusion-fi06-output-"),
      );
      try {
        const generated = await context.generateElementwiseOpIr({
          sourceOverrides: {
            add: source.replace(original, "result = left - right;"),
          },
          outputDirectory,
          scratchDirectory,
        });
        const staleRegistry = context.createRegistry({
          declarations: generated.declarations,
          fingerprint: helperFingerprint,
        });
        const staleProgram = context.elementwiseFusionCpuProgram(
          baseCandidate,
          staleRegistry,
        );
        const actual = context.dispatchElementwiseFusionCpu(
          staleProgram.cpu,
          staleProgram.inputNames,
          baseCandidate.leaves.map((leaf: any) => leaf.node.value),
          baseCandidate.leaves[0]?.node.value.length ?? 0,
        );
        expectPredicateToReject(
          () =>
            assertions.output(
              actual,
              context.changedSourceExpected,
              "FI06 stale fingerprint",
            ),
          "FI06 stale helper fingerprint",
          "FI06 stale fingerprint[",
        );
      } finally {
        await rm(scratchDirectory, { recursive: true, force: true });
        await rm(outputDirectory, { recursive: true, force: true });
      }
    },
  );

  await check(
    "FI07-device-cap-wiring-mutation",
    "T12 assertion rejects the executor using an overlarge injected cap",
    async () => {
      const events: string[] = [];
      const buffers = new context.DeviceBuffers();
      const fakeRuntime = {
        tier: "webgpu",
        device: {
          limits: {
            maxStorageBuffersPerShaderStage: freshCandidate.leaves.length,
          },
        },
        frame() {
          return {
            dispatch() {
              events.push("dispatch");
            },
            async submit() {},
          };
        },
        load() {
          return {
            async compute() {
              return {};
            },
          };
        },
      };
      const executor = new context.GpuGraphExecutor(
        fakeRuntime,
        new Map([
          ["add", {}],
          ["mul", {}],
          ["relu", {}],
        ]),
        buffers,
        undefined,
        {
          maxStorageBuffers: freshCandidate.leaves.length + 1,
          onAnalysis(decision: any) {
            events.push(`decision:${decision.status}:${decision.code ?? ""}`);
          },
          onResource(resource: string) {
            events.push(resource);
          },
        },
      );
      try {
        await executor.execute(freshCandidate.output);
        if (
          !events.some((event) => event.startsWith("decision:fused:")) ||
          !events.includes("output-allocation") ||
          !events.includes("fusion-pack")
        )
          throw new Error(
            "FI07 cap mutation did not reach the fused resource path: " +
              events.join(", "),
          );
        expectPredicateToReject(
          () => assertions.storageLimit(events),
          "FI07 executor cap override",
          "storage limit decision is refused",
        );
      } finally {
        buffers.dispose();
      }
    },
  );

  await check(
    "FI08-stale-input-values-mutation",
    "T10 freshness assertion rejects prior leaf buffers",
    () => {
      const program = context.elementwiseFusionCpuProgram(
        context.freshCandidate,
      );
      const actual = context.dispatchElementwiseFusionCpu(
        program.cpu,
        program.inputNames,
        context.staleInputs,
        context.staleInputs[0]?.length ?? 0,
      );
      expectPredicateToReject(
        () =>
          assertions.freshData(
            actual,
            context.freshExpected,
            "FI08 stale leaf bindings",
          ),
        "FI08 stale input bindings",
        "FI08 stale leaf bindings[",
      );
    },
  );
}
