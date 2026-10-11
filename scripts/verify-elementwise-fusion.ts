import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import {
  compileModule,
  compileModuleJs,
  compileTsSource,
  module as shadeModule,
} from "typeshade";
import { generateModuleJs } from "../node_modules/typeshade/src/core/cpu-codegen.ts";
import { generateElementwiseOpIr } from "./build-elementwise-op-ir.mjs";
import {
  buildFusionFixture,
  FUSION_CASES,
} from "./verify-elementwise-fusion-cases.js";
import {
  assertFloat32ArraysEqual,
  evaluateFusionFixtureF32,
} from "./verify-elementwise-fusion-oracle.js";
import { FusionVerificationRecorder } from "./verify-elementwise-fusion-report.js";
import { verifyFusionMutations } from "./verify-elementwise-fusion-mutations.js";

const ENTRY = "fusedElementwiseKernel";

function assertModuleHelpers(module: any, declarations: readonly any[]): void {
  const functions = new Map(module.funcs.map((func: any) => [func.name, func]));
  for (const declaration of declarations) {
    assert.deepEqual(
      functions.get(declaration.name),
      declaration,
      `Composed module must preserve source FuncDecl ${declaration.name}.`,
    );
  }
}

function assertOneEntryAndStorage(
  program: any,
  expectedInputCount: number,
): void {
  const entries = program.module.funcs.filter(
    (func: any) => (func.stage ?? func.decl?.stage) === "compute",
  );
  assert.equal(entries.length, 1, "module has exactly one compute entry");
  assert.equal(
    entries[0]?.name,
    program.entry,
    "the entry is the shared compute entry",
  );
  const storage = program.module.bindings.filter(
    (binding: any) => binding.space === "storage",
  );
  assert.equal(
    storage.length,
    expectedInputCount + 1,
    "storage is inputs plus final output",
  );
  assert.deepEqual(
    storage.map((binding: any) => binding.name),
    [...program.inputNames, "output"],
    "storage contains only graph leaves and final output",
  );
}

function assertEntryWritesFinalOutput(program: any): void {
  const entry = program.module.funcs.find(
    (func: any) => func.name === program.entry,
  );
  assert.ok(entry, "module has the declared compute entry");
  assert.ok(
    JSON.stringify(entry.body).includes('"name":"output"'),
    "entry writes final output",
  );
}

function assertNoCodegenFallback(report: any, entry: string): void {
  assert.deepEqual(
    report.fallbacks,
    [],
    "TypeShade emitted every module function",
  );
  assert.ok(
    report.fns.some((source: string) =>
      source.includes(`${JSON.stringify(entry)}: function`),
    ),
    `generated JavaScript includes ${entry}`,
  );
  assert.ok(
    report.factorySource.includes(entry),
    "factory source contains the entry",
  );
}

function assertEmittedCpuExecution(program: any, dispatches: number): void {
  assert.equal(program?.cpuBackend, "emitted-js", "CPU backend is emitted-js");
  assert.ok(program?.cpu, "production CPU program is present");
  assert.equal(dispatches, 1, "production CPU graph uses one fused dispatch");
}

function assertStorageLimitBeforeResources(events: readonly string[]): void {
  assert.ok(
    events.includes("decision:refused:storage-buffer-limit"),
    "storage limit decision is refused",
  );
  assert.equal(
    events.includes("output-allocation"),
    false,
    "refusal happens before output allocation",
  );
  assert.equal(
    events.includes("fusion-pack"),
    false,
    "refusal happens before fusion pack",
  );
}

function collectLocalVariableNames(
  value: unknown,
  names: string[] = [],
): string[] {
  if (Array.isArray(value)) {
    for (const item of value) collectLocalVariableNames(item, names);
  } else if (value && typeof value === "object") {
    const object = value as Record<string, unknown>;
    if (object.s === "var" && typeof object.name === "string")
      names.push(object.name);
    for (const item of Object.values(object))
      collectLocalVariableNames(item, names);
  }
  return names;
}

function assertDiamondOutput(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  label: string,
): void {
  assertFloat32ArraysEqual(actual, expected, label);
  assert.equal(actual[0], 25);
}

function assertDiamondLocals(module: any): void {
  const entry = module.funcs.find((func: any) => func.name === ENTRY);
  assert.ok(entry, "diamond has the declared compute entry");
  const locals = collectLocalVariableNames(entry.body);
  assert.deepEqual(
    locals,
    ["value0", "value1", "value2", "value3", "value4"],
    "shared node is computed once and each op has a distinct local",
  );
  const calls = entry.body
    .filter((statement: any) => statement.s === "call")
    .map((statement: any) => statement.expr)
    .filter((expression: any) =>
      ["typennAddF32", "typennMulF32", "typennReluF32"].includes(expression.fn),
    );
  assert.deepEqual(
    calls.map((call: any) => call.args[0]?.name),
    ["value0", "value1", "value2", "value3", "value4"],
    "each operation writes to its own fresh local",
  );
}

function makeFixtureWithInputs(
  testCase: any,
  inputs: Record<string, number[]>,
) {
  return { ...testCase, inputs };
}

function fixtureInputs(testCase: any): Record<string, Float32Array> {
  return Object.fromEntries(
    Object.entries(testCase.inputs).map(([name, values]) => [
      name,
      Float32Array.from(values as number[]),
    ]),
  );
}

function standaloneAddResult(
  source: string,
  left: number,
  right: number,
): number {
  const compiled = compileTsSource(source, {
    fileName: "src/kernels/add.shade.ts",
    emit: false,
  });
  const errors = compiled.diagnostics.filter(
    (diagnostic) => diagnostic.category === "error",
  );
  assert.deepEqual(errors, [], "standalone add shader source compiles");
  const standalone = shadeModule({
    consts: compiled.consts,
    structs: compiled.structs.map((item: any) => item.decl ?? item),
    bindings: compiled.bindings,
    overrides: compiled.overrides,
    vars: compiled.vars,
    funcs: compiled.funcs,
  });
  const cpu = compileModuleJs(standalone, { precision: "f32" });
  const result = [0];
  cpu.setBinding("shape", {
    length: 1,
    rank: 1,
    dimensions: [1, 1, 1, 1, 1, 1, 1, 1],
    aStrides: [1, 0, 0, 0, 0, 0, 0, 0],
    bStrides: [1, 0, 0, 0, 0, 0, 0, 0],
  });
  cpu.setBinding("a", [left]);
  cpu.setBinding("b", [right]);
  cpu.setBinding("output", result);
  cpu.dispatch("addKernel", [1, 1, 1]);
  return Math.fround(result[0]!);
}

async function generateIsolatedRegistry(
  sourceOverrides: Record<string, string>,
) {
  const directory = await mkdtemp(path.join(tmpdir(), "typenn-fusion-source-"));
  try {
    const result = await generateElementwiseOpIr({
      sourceOverrides,
      outputDirectory: path.join(directory, "generated"),
    });
    return { directory, result };
  } catch (error) {
    await rm(directory, { recursive: true, force: true });
    throw error;
  }
}

async function withCpuRuntime<T>(
  body: (runtime: any) => Promise<T>,
): Promise<T> {
  const { device } = await import("../src/index.js");
  const runtime = await device("cpu");
  try {
    return await body(runtime);
  } finally {
    await runtime.dispose();
  }
}

export async function runElementwiseFusionVerification(input: {
  readonly commit: string;
  readonly worktreeFingerprint: string;
}): Promise<{
  readonly commit: string;
  readonly worktreeFingerprint: string;
  readonly helperFingerprint: string;
  readonly gpu: "not-run";
  readonly tests: readonly any[];
  readonly passed: boolean;
}> {
  const recorder = new FusionVerificationRecorder();
  const { device } = await import("../src/index.js");
  const { analyzeElementwiseFusion, inspectElementwiseFusion } =
    await import("../src/runtime/elementwise-fusion.js");
  const { evaluateCpuWithStats, evaluateCpuUnfusedWithStats } =
    await import("../src/runtime/cpu-executor.js");
  const {
    dispatchElementwiseFusionCpu,
    elementwiseFusionCpuProgram,
    elementwiseFusionGpuProgram,
    elementwiseFusionProgram,
  } = await import("../src/runtime/elementwise-fusion-ir.js");
  const { createElementwiseFusionRegistry, ELEMENTWISE_FUSION_REGISTRY } =
    await import("../src/runtime/kernel-registry.js");
  const { DeviceBuffers } = await import("../src/runtime/device-buffers.js");
  const { GpuGraphExecutor } =
    await import("../src/runtime/gpu-graph-executor.js");

  const helperFingerprint = ELEMENTWISE_FUSION_REGISTRY.fingerprint;
  let cspMutationOutputForFault: any = null;
  let cspMutationExpectedForFault: ArrayLike<number> = [];
  const runtime = await device("cpu");
  try {
    const base = FUSION_CASES.find(
      (testCase) => testCase.id === "add-mul-relu",
    )!;
    const baseGraph = buildFusionFixture(runtime, base);
    const baseCandidate = analyzeElementwiseFusion(baseGraph.output);
    assert.ok("decision" in baseCandidate, "base graph is a candidate");

    await recorder.run("T01-source-provenance", async (check) => {
      const program = elementwiseFusionProgram(baseCandidate);
      await check("original FuncDecl bodies and modes are preserved", () => {
        assertModuleHelpers(program.module, [
          ...ELEMENTWISE_FUSION_REGISTRY.helpers.values(),
        ]);
      });
      await check(
        "registered helpers have one inout f32 output and by-value f32 inputs",
        () => {
          for (const helper of ELEMENTWISE_FUSION_REGISTRY.helpers.values()) {
            assert.equal(helper.ret.kind, "void");
            assert.equal(helper.params[0]?.mode, "inout");
            assert.equal(helper.params[0]?.type.kind, "scalar");
            assert.equal(helper.params[0]?.type.scalar, "f32");
            for (const parameter of helper.params.slice(1)) {
              assert.equal(parameter.mode, undefined);
              assert.equal(parameter.type.kind, "scalar");
              assert.equal(parameter.type.scalar, "f32");
            }
          }
        },
      );
      await check(
        "entry contains registered function calls rather than formula selection",
        () => {
          const entry = program.module.funcs.find(
            (func: any) => func.name === ENTRY,
          )!;
          const text = JSON.stringify(entry.body);
          for (const helper of ELEMENTWISE_FUSION_REGISTRY.helpers.values())
            assert.ok(text.includes(helper.name), `entry calls ${helper.name}`);
        },
      );
    });

    await recorder.run("T02-source-change-production-path", async (check) => {
      const addPath = path.resolve("src/kernels/add.shade.ts");
      const addSource = await readFile(addPath, "utf8");
      const originalAssignment = "result = left + right;";
      assert.equal(addSource.split(originalAssignment).length - 1, 1);
      const changedSource = addSource.replace(
        originalAssignment,
        "result = left - right;",
      );
      const isolated = await generateIsolatedRegistry({ add: changedSource });
      const scalarCase = makeFixtureWithInputs(base, {
        a: [5],
        b: [3],
        c: [2],
      });
      const scalarGraph = buildFusionFixture(runtime, scalarCase);
      const scalarCandidate = analyzeElementwiseFusion(scalarGraph.output);
      assert.ok("decision" in scalarCandidate);
      try {
        const changedRegistry = createElementwiseFusionRegistry({
          declarations: isolated.result.declarations,
          fingerprint: isolated.result.fingerprint,
        });
        const changedGraph = buildFusionFixture(runtime, base);
        const changedCandidate = analyzeElementwiseFusion(changedGraph.output);
        assert.ok("decision" in changedCandidate);
        const changedProgram = elementwiseFusionCpuProgram(
          changedCandidate,
          changedRegistry,
        );
        const changedValue = dispatchElementwiseFusionCpu(
          changedProgram.cpu,
          changedProgram.inputNames,
          changedCandidate.leaves.map((leaf: any) => leaf.node.value),
          4,
        );
        const scalarChangedCandidate = analyzeElementwiseFusion(
          buildFusionFixture(runtime, scalarCase).output,
        );
        assert.ok("decision" in scalarChangedCandidate);
        const scalarChangedProgram = elementwiseFusionCpuProgram(
          scalarChangedCandidate,
          changedRegistry,
        );
        const scalarChangedValue = dispatchElementwiseFusionCpu(
          scalarChangedProgram.cpu,
          scalarChangedProgram.inputNames,
          scalarChangedCandidate.leaves.map((leaf: any) => leaf.node.value),
          1,
        );
        const scalarBaselineProgram =
          elementwiseFusionCpuProgram(scalarCandidate);
        const scalarBaselineValue = dispatchElementwiseFusionCpu(
          scalarBaselineProgram.cpu,
          scalarBaselineProgram.inputNames,
          scalarCandidate.leaves.map((leaf: any) => leaf.node.value),
          1,
        );
        const expectedChanged = evaluateFusionFixtureF32(
          base.fixture,
          fixtureInputs(base),
        );
        const corrected = Float32Array.from(expectedChanged, (_, index) => {
          const a = Math.fround(base.inputs.a![index]!);
          const b = Math.fround(base.inputs.b![index]!);
          const c = Math.fround(base.inputs.c![index]!);
          return Math.fround(Math.max(0, Math.fround(Math.fround(a - b) * c)));
        });
        await check(
          "production generator and composer use changed helper semantics",
          () => {
            assertFloat32ArraysEqual(
              changedValue,
              corrected,
              "mutated production path",
            );
            assert.equal(
              isolated.result.fingerprint === helperFingerprint,
              false,
            );
            assert.notEqual(
              changedProgram.signature,
              elementwiseFusionProgram(baseCandidate).signature,
            );
          },
        );
        await check(
          "standalone kernel compiled from changed source also changes",
          () => {
            assert.equal(standaloneAddResult(addSource, 5, 3), 8);
            assert.equal(standaloneAddResult(changedSource, 5, 3), 2);
          },
        );
        await check(
          "source override changes fused scalar result from 16 to 4",
          () => {
            assertFloat32ArraysEqual(scalarBaselineValue, [16], "baseline");
            assertFloat32ArraysEqual(scalarChangedValue, [4], "changed source");
          },
        );
      } finally {
        await rm(isolated.directory, { recursive: true, force: true });
      }
    });

    await recorder.run(
      "T03-helper-contract-negative-fixtures",
      async (check) => {
        const addPath = path.resolve("src/kernels/add.shade.ts");
        const addSource = await readFile(addPath, "utf8");
        const mutations = [
          [
            "output-unwritten",
            addSource.replace("result = left + right;", ""),
            "must assign its output exactly once",
          ],
          [
            "output-read-before-write",
            addSource.replace(
              "result = left + right;",
              "result = result + right;",
            ),
            "reads its output before assigning it",
          ],
          [
            "external-parameter-write",
            addSource.replace("result = left + right;", "left = right;"),
            "must assign its output exactly once in a straight-line body",
          ],
          [
            "multiple-output-writes",
            addSource.replace(
              "result = left + right;",
              "result = left + right; result = left;",
            ),
            "must assign its output exactly once",
          ],
          [
            "global-read",
            addSource
              .replace(
                '"use typeshade";',
                '"use typeshade";\ndeclare const offset: uniform<f32>;',
              )
              .replace("result = left + right;", "result = left + offset;"),
            "reads undeclared parameter offset",
          ],
          [
            "wrong-output-mode",
            addSource.replace("@inout result: f32", "result: f32"),
            "must begin with one @inout f32 output parameter",
          ],
          [
            "wrong-output-type",
            addSource.replace("@inout result: f32", "@inout result: i32"),
            "TypeShade diagnostics",
          ],
          [
            "unknown-call",
            addSource.replace(
              "result = left + right;",
              "result = notRegistered(left, right);",
            ),
            'Unknown function "notRegistered"',
          ],
        ] as const;
        const directory = await mkdtemp(
          path.join(tmpdir(), "typenn-fusion-negative-"),
        );
        try {
          for (const [id, source, expectedDiagnostic] of mutations) {
            await check(
              `${id} is rejected by the production generator`,
              async () => {
                await assert.rejects(
                  generateElementwiseOpIr({
                    sourceOverrides: { add: source },
                    outputDirectory: path.join(directory, id),
                  }),
                  (error: any) => {
                    const diagnostic = `${error?.message ?? ""}\n${error?.stderr?.toString?.() ?? ""}`;
                    assert.ok(
                      diagnostic.includes(expectedDiagnostic),
                      `Expected ${JSON.stringify(expectedDiagnostic)} in generator error, got:\n${diagnostic}`,
                    );
                    return true;
                  },
                );
              },
            );
          }
        } finally {
          await rm(directory, { recursive: true, force: true });
        }
      },
    );

    await recorder.run("T04-generic-dag-composition", async (check) => {
      const cases = FUSION_CASES.filter((item) =>
        [
          "add-relu",
          "add-mul-relu",
          "mul-add-relu",
          "reordered-four-op",
          "repeated-add-mul",
        ].includes(item.id),
      );
      for (const testCase of cases) {
        const graph = buildFusionFixture(runtime, testCase);
        const candidate = analyzeElementwiseFusion(graph.output);
        assert.ok("decision" in candidate, `${testCase.id} is eligible`);
        const expected = evaluateFusionFixtureF32(
          testCase.fixture,
          fixtureInputs(testCase),
        );
        const fused = await evaluateCpuWithStats(graph.output);
        await check(
          `${testCase.id} matches independent nodewise f32 DAG`,
          () => {
            assertFloat32ArraysEqual(fused.value, expected, testCase.id);
          },
        );
        await check(`${testCase.id} uses one actual fused CPU dispatch`, () => {
          assert.equal(fused.dispatches, 1);
          const entry = fused.fusionProgram?.module.funcs.find(
            (f: any) => f.name === ENTRY,
          );
          assert.equal(entry?.stage ?? entry?.decl?.stage, "compute");
        });
        await check(`${testCase.id} preserves authored operation order`, () => {
          assert.deepEqual(
            candidate.operations,
            testCase.fixture.nodes.map((node) => node.op),
          );
        });
        const unfusedGraph = buildFusionFixture(runtime, testCase);
        const unfused = await evaluateCpuUnfusedWithStats(unfusedGraph.output);
        await check(
          `${testCase.id} agrees with fresh TypeNN nodewise execution`,
          () => {
            assertFloat32ArraysEqual(
              unfused.value,
              expected,
              `${testCase.id} unfused`,
            );
            assert.equal(unfused.dispatches, testCase.fixture.nodes.length);
          },
        );
      }
    });

    await recorder.run("T05-shared-diamond", async (check) => {
      const testCase = FUSION_CASES.find(
        (item) => item.id === "diamond-shared-node",
      )!;
      const graph = buildFusionFixture(runtime, testCase);
      const candidate = analyzeElementwiseFusion(graph.output);
      assert.ok("decision" in candidate);
      const actual = await evaluateCpuWithStats(graph.output);
      const expected = evaluateFusionFixtureF32(
        testCase.fixture,
        fixtureInputs(testCase),
      );
      await check(
        "diamond result preserves both branches and shared node",
        () => {
          assertDiamondOutput(actual.value, expected, "diamond");
        },
      );
      await check(
        "shared node is computed once and each op has a distinct local",
        () => {
          assert.equal(candidate.steps.length, 5);
          assert.equal(
            candidate.steps.filter((step: any) => step.operation.name === "add")
              .length,
            2,
          );
          assertDiamondLocals(actual.fusionProgram!.module);
        },
      );
      await check(
        "all unique leaves and shared references are represented",
        () => {
          assert.equal(candidate.leaves.length, 2);
          assert.equal(actual.dispatches, 1);
        },
      );
    });

    await recorder.run("T06-repeated-leaf-dedup", async (check) => {
      const testCase = FUSION_CASES.find(
        (item) => item.id === "repeated-leaf",
      )!;
      const graph = buildFusionFixture(runtime, testCase);
      const candidate = analyzeElementwiseFusion(graph.output);
      assert.ok("decision" in candidate);
      const actual = await evaluateCpuWithStats(graph.output);
      await check("repeated input value uses one leaf slot", () => {
        assert.equal(candidate.leaves.length, 1);
        assert.equal(actual.fusionProgram?.inputNames.length, 1);
      });
      await check(
        "repeated input semantics match independent f32 evaluation",
        () => {
          assertFloat32ArraysEqual(
            actual.value,
            evaluateFusionFixtureF32(testCase.fixture, fixtureInputs(testCase)),
            "repeated leaf",
          );
        },
      );
      await check(
        "one input storage and one output storage are composed",
        () => {
          assertOneEntryAndStorage(actual.fusionProgram, 1);
        },
      );
    });

    await recorder.run("T07-one-entry-storage-contract", async (check) => {
      const program = elementwiseFusionProgram(baseCandidate);
      await check("base module has one entry and only graph storage", () => {
        assertOneEntryAndStorage(program, baseCandidate.leaves.length);
      });
      await check(
        "intermediate results are local variables, not storage bindings",
        () => {
          const entry = program.module.funcs.find(
            (func: any) => func.name === ENTRY,
          )!;
          const body = JSON.stringify(entry.body);
          assert.ok(body.includes("value0"));
          assert.ok(body.includes("value1"));
          assert.ok(body.includes("value2"));
          assert.equal(
            program.module.bindings.some((binding: any) =>
              /^value\d+$/.test(binding.name),
            ),
            false,
          );
        },
      );
      await check("CPU and GPU share the same TypeShade ModuleDecl", () => {
        const cpu = elementwiseFusionCpuProgram(baseCandidate);
        const gpu = elementwiseFusionGpuProgram(baseCandidate);
        assert.equal(cpu.module, program.module);
        assert.equal(gpu.module, program.module);
        assert.equal(gpu.signature, cpu.signature);
        for (const helper of ["typennAddF32", "typennMulF32", "typennReluF32"])
          assert.match(
            gpu.pack.wgsl,
            new RegExp(`fn ${helper}\\(result: ptr<function, f32>`),
          );
        assert.doesNotMatch(
          gpu.pack.wgsl,
          /fn typenn(?:Add|Mul|Relu)F32\([^)]*ptr<storage/,
        );
      });
      await check(
        "two-, three-, four-, and five-op topologies keep inputs-plus-output storage",
        () => {
          for (const id of [
            "add-relu",
            "mul-add-relu",
            "reordered-four-op",
            "repeated-add-mul",
            "diamond-shared-node",
          ]) {
            const testCase = FUSION_CASES.find((item) => item.id === id)!;
            const graph = buildFusionFixture(runtime, testCase);
            const candidate = analyzeElementwiseFusion(graph.output);
            assert.ok("decision" in candidate, `${id} is eligible`);
            const candidateProgram = elementwiseFusionProgram(candidate);
            assertOneEntryAndStorage(candidateProgram, candidate.leaves.length);
          }
        },
      );
    });

    await recorder.run("T08-real-cpu-emission-path", async (check) => {
      const custom = {
        id: "cpu-emission-unique",
        fixture: {
          nodes: [
            { id: "v0", op: "add", inputs: [{ input: "a" }, { input: "b" }] },
            { id: "v1", op: "relu", inputs: [{ value: "v0" }] },
            { id: "v2", op: "mul", inputs: [{ value: "v1" }, { input: "c" }] },
            { id: "v3", op: "add", inputs: [{ value: "v2" }, { input: "a" }] },
            { id: "out", op: "relu", inputs: [{ value: "v3" }] },
          ],
          output: { value: "out" },
        },
        inputs: { a: [2, -3], b: [4, 5], c: [3, -2] },
      } as const;
      const graph = buildFusionFixture(runtime, custom);
      const candidate = analyzeElementwiseFusion(graph.output);
      assert.ok("decision" in candidate);
      const nativeFunction = globalThis.Function;
      const captured: string[] = [];
      const wrapper = new Proxy(nativeFunction, {
        construct(target, args, newTarget) {
          if (typeof args[1] === "string") captured.push(args[1]);
          return Reflect.construct(target, args, newTarget);
        },
      });
      let execution: any;
      try {
        globalThis.Function = wrapper;
        execution = await evaluateCpuWithStats(graph.output);
      } finally {
        globalThis.Function = nativeFunction;
      }
      const generated = generateModuleJs(execution.fusionProgram.module, {
        precision: "f32",
      });
      const interpretedModule = compileModule(execution.fusionProgram.module, {
        precision: "f32",
      });
      const inputValues = candidate.leaves.map((leaf: any) => leaf.node.value);
      for (
        let slot = 0;
        slot < execution.fusionProgram.inputNames.length;
        slot++
      ) {
        interpretedModule.setBinding(
          execution.fusionProgram.inputNames[slot]!,
          Array.from(inputValues[slot]!),
        );
      }
      const outputBuffer = new Array<number>(custom.inputs.a.length).fill(0);
      interpretedModule.setBinding("shape", { length: custom.inputs.a.length });
      interpretedModule.setBinding("output", outputBuffer);
      interpretedModule.dispatch(execution.fusionProgram.entry, 1);
      const interpretedValue = Float32Array.from(outputBuffer);
      await check(
        "production CPU execution constructs and runs emitted JavaScript",
        () => {
          assertEmittedCpuExecution(
            execution.fusionProgram,
            execution.dispatches,
          );
          assert.equal(captured.length, 1);
          assert.ok(captured[0]!.includes(ENTRY));
        },
      );
      await check(
        "pinned TypeShade generator reports no per-function fallback",
        () => {
          assertNoCodegenFallback(generated, ENTRY);
          assert.equal(
            generated.module.funcs.some((f: any) => f.name === ENTRY),
            true,
          );
        },
      );
      await check(
        "actual emitted entry agrees with independent f32 DAG",
        () => {
          assertFloat32ArraysEqual(
            execution.value,
            evaluateFusionFixtureF32(custom.fixture, fixtureInputs(custom)),
            "emitted entry",
          );
        },
      );
      await check(
        "same ModuleDecl interpreter agrees with independent f32 DAG",
        () => {
          const expected = evaluateFusionFixtureF32(
            custom.fixture,
            fixtureInputs(custom),
          );
          assertFloat32ArraysEqual(
            interpretedValue,
            expected,
            "IR interpreter",
          );
          assertFloat32ArraysEqual(
            interpretedValue,
            execution.value,
            "IR interpreter versus emitted CPU",
          );
        },
      );

      const cspCustom = {
        id: "cpu-csp-fallback-unique",
        fixture: {
          nodes: [
            {
              id: "product0",
              op: "mul",
              inputs: [{ input: "a" }, { input: "b" }],
            },
            { id: "positive", op: "relu", inputs: [{ value: "product0" }] },
            {
              id: "sum",
              op: "add",
              inputs: [{ value: "positive" }, { input: "c" }],
            },
            {
              id: "product1",
              op: "mul",
              inputs: [{ value: "sum" }, { input: "b" }],
            },
            { id: "out", op: "relu", inputs: [{ value: "product1" }] },
          ],
          output: { value: "out" },
        },
        inputs: { a: [2, -3, 1.5], b: [4, 2, -2], c: [1, -5, 3] },
      } as const;
      const cspGraph = buildFusionFixture(runtime, cspCustom);
      const cspCandidate = analyzeElementwiseFusion(cspGraph.output);
      assert.ok("decision" in cspCandidate);
      const nativeFunctionForCsp = globalThis.Function;
      let deniedFunctionConstructions = 0;
      let fallbackProgram: any;
      try {
        globalThis.Function = new Proxy(nativeFunctionForCsp, {
          construct() {
            deniedFunctionConstructions++;
            throw new EvalError(
              "Code generation from strings disallowed for this context",
            );
          },
        });
        fallbackProgram = elementwiseFusionCpuProgram(cspCandidate);
      } finally {
        globalThis.Function = nativeFunctionForCsp;
      }
      const fallbackValue = dispatchElementwiseFusionCpu(
        fallbackProgram.cpu,
        fallbackProgram.inputNames,
        cspCandidate.leaves.map((leaf: any) => leaf.node.value),
        cspCustom.inputs.a.length,
      );
      await check(
        "CSP EvalError exercises the uncached interpreter fallback",
        () => {
          assert.ok(deniedFunctionConstructions > 0);
          assert.equal(fallbackProgram.cpuBackend, "csp-interpreter-fallback");
        },
      );
      await check(
        "CSP interpreter fallback agrees with independent f32 DAG",
        () => {
          assertFloat32ArraysEqual(
            fallbackValue,
            evaluateFusionFixtureF32(
              cspCustom.fixture,
              fixtureInputs(cspCustom),
            ),
            "CSP interpreter fallback",
          );
        },
      );

      const cspMutationFixture = {
        id: "cpu-csp-fault-unique",
        fixture: {
          nodes: [
            {
              id: "product0",
              op: "mul",
              inputs: [{ input: "a" }, { input: "b" }],
            },
            {
              id: "sum0",
              op: "add",
              inputs: [{ value: "product0" }, { input: "c" }],
            },
            { id: "positive", op: "relu", inputs: [{ value: "sum0" }] },
            {
              id: "product1",
              op: "mul",
              inputs: [{ value: "positive" }, { input: "a" }],
            },
            {
              id: "sum1",
              op: "add",
              inputs: [{ value: "product1" }, { input: "b" }],
            },
            { id: "out", op: "relu", inputs: [{ value: "sum1" }] },
          ],
          output: { value: "out" },
        },
        inputs: { a: [2, -3, 1.5], b: [4, 2, -2], c: [1, -5, 3] },
      } as const;
      const cspMutationGraph = buildFusionFixture(runtime, cspMutationFixture);
      const cspMutationCandidate = analyzeElementwiseFusion(
        cspMutationGraph.output,
      );
      assert.ok("decision" in cspMutationCandidate);
      cspMutationOutputForFault = cspMutationGraph.output;
      cspMutationExpectedForFault = evaluateFusionFixtureF32(
        cspMutationFixture.fixture,
        fixtureInputs(cspMutationFixture),
      );
    });

    await recorder.run("T09-f32-rounding-and-bounds", async (check) => {
      const baseCase = FUSION_CASES.find((item) => item.id === "add-mul-relu")!;
      const lengths = [1, 63, 64, 65, 67];
      for (const length of lengths) {
        const expanded = makeFixtureWithInputs(baseCase, {
          a: Array.from({ length }, (_, index) => (index % 7) - 3),
          b: Array.from({ length }, (_, index) => (index % 5) - 2),
          c: Array.from({ length }, (_, index) => (index % 3) - 1),
        });
        const graph = buildFusionFixture(runtime, expanded);
        const actual = await evaluateCpuWithStats(graph.output);
        await check(
          `length ${length} has exact f32 result and bounds behavior`,
          () => {
            assertFloat32ArraysEqual(
              actual.value,
              evaluateFusionFixtureF32(
                expanded.fixture,
                fixtureInputs(expanded),
              ),
              `bounds ${length}`,
            );
            assert.equal(actual.value.length, length);
          },
        );
      }
      const rounding = makeFixtureWithInputs(baseCase, {
        a: [16777216],
        b: [1],
        c: [3],
      });
      const rounded = await evaluateCpuWithStats(
        buildFusionFixture(runtime, rounding).output,
      );
      await check(
        "intermediate f32 rounding differs from f64 accumulation",
        () => {
          const expected = evaluateFusionFixtureF32(
            rounding.fixture,
            fixtureInputs(rounding),
          );
          assert.equal(expected[0], 50331648);
          assertFloat32ArraysEqual(
            rounded.value,
            expected,
            "f32 precision witness",
          );
          assert.notEqual(Math.fround((16777216 + 1) * 3), expected[0]);
        },
      );
    });

    await recorder.run(
      "T10-cache-topology-data-and-fingerprint",
      async (check) => {
        const firstGraph = buildFusionFixture(runtime, base);
        const firstCandidate = analyzeElementwiseFusion(firstGraph.output);
        assert.ok("decision" in firstCandidate);
        const firstProgram = elementwiseFusionProgram(firstCandidate);
        const changedInputs = makeFixtureWithInputs(base, {
          a: [10, 11, 12, 13],
          b: [1, 2, 3, 4],
          c: [2, 3, 4, 5],
        });
        const secondGraph = buildFusionFixture(runtime, changedInputs);
        const secondCandidate = analyzeElementwiseFusion(secondGraph.output);
        assert.ok("decision" in secondCandidate);
        const secondProgram = elementwiseFusionProgram(secondCandidate);
        await check(
          "same topology reuses generated module without caching Tensor data",
          () => {
            assert.equal(firstProgram.module, secondProgram.module);
            const actual = dispatchElementwiseFusionCpu(
              elementwiseFusionCpuProgram(secondCandidate).cpu,
              secondProgram.inputNames,
              secondCandidate.leaves.map((leaf: any) => leaf.node.value),
              4,
            );
            assertFloat32ArraysEqual(
              actual,
              evaluateFusionFixtureF32(
                base.fixture,
                fixtureInputs(changedInputs),
              ),
              "fresh cache values",
            );
          },
        );
        const longerInputs = makeFixtureWithInputs(base, {
          a: Array.from({ length: 67 }, (_, index) => (index % 11) - 5),
          b: Array.from({ length: 67 }, (_, index) => (index % 7) - 3),
          c: Array.from({ length: 67 }, (_, index) => (index % 5) - 2),
        });
        const longerGraph = buildFusionFixture(runtime, longerInputs);
        const longerCandidate = analyzeElementwiseFusion(longerGraph.output);
        assert.ok("decision" in longerCandidate);
        const longerProgram = elementwiseFusionCpuProgram(longerCandidate);
        const longerActual = dispatchElementwiseFusionCpu(
          longerProgram.cpu,
          longerProgram.inputNames,
          longerCandidate.leaves.map((leaf: any) => leaf.node.value),
          67,
        );
        await check(
          "cached CPU program updates shape length for a different tensor length",
          () => {
            assert.equal(longerProgram.module, firstProgram.module);
            assert.equal(longerActual.length, 67);
            assertFloat32ArraysEqual(
              longerActual,
              evaluateFusionFixtureF32(
                base.fixture,
                fixtureInputs(longerInputs),
              ),
              "cached program with new shape length",
            );
          },
        );
        const reorderedFixture = FUSION_CASES.find(
          (item) => item.id === "reordered-four-op",
        )!;
        const reordered = buildFusionFixture(runtime, reorderedFixture);
        const reorderedCandidate = analyzeElementwiseFusion(reordered.output);
        assert.ok("decision" in reorderedCandidate);
        const reorderedProgram = elementwiseFusionProgram(reorderedCandidate);
        await check(
          "different topology receives a different cache signature",
          () => {
            assert.notEqual(firstProgram.signature, reorderedProgram.signature);
            assert.notEqual(firstProgram.module, reorderedProgram.module);
          },
        );
        await check(
          "compositor identity is keyed by helper fingerprint",
          () => {
            assert.ok(
              firstProgram.signature.startsWith(`${helperFingerprint}:`),
            );
          },
        );
      },
    );

    await recorder.run("T11-conservative-fallbacks", async (check) => {
      const a = runtime.tensor([1, 2, 3, 4], { shape: [4] });
      const b = runtime.tensor([2, 3, 4, 5], { shape: [4] });
      const c = runtime.tensor([1, -1, 2, 3], { shape: [4] });
      const unsupported = a.sub(b).mul(c).relu();
      const unsupportedDecision = inspectElementwiseFusion(unsupported);
      const unsupportedStats = await evaluateCpuWithStats(unsupported);
      await check(
        "unregistered operation declines fusion and keeps nodewise result",
        () => {
          assert.equal(unsupportedDecision.status, "refused");
          assert.equal(unsupportedDecision.code, "unsupported-chain");
          assert.equal(unsupportedStats.fusionProgram, undefined);
          assert.equal(unsupportedStats.dispatches, 3);
          assertFloat32ArraysEqual(
            unsupportedStats.value,
            [0, 1, 0, 0],
            "unsupported fallback",
          );
        },
      );

      const rootA = runtime.tensor([12, 8, 9, 4], { shape: [4] });
      const rootB = runtime.tensor([4, 4, 3, 2], { shape: [4] });
      const rootC = runtime.tensor([3, 2, 5, 6], { shape: [4] });
      const eligibleChild = rootA.add(rootB).mul(rootC).relu();
      const unsupportedRoot = eligibleChild.div(rootA);
      const rootDecision = inspectElementwiseFusion(unsupportedRoot);
      const rootStats = await evaluateCpuWithStats(unsupportedRoot);
      await check(
        "unregistered root keeps eligible child on the full nodewise path",
        () => {
          assert.equal(rootDecision.status, "refused");
          assert.equal(rootDecision.code, "unsupported-chain");
          assert.equal(rootStats.fusionProgram, undefined);
          assert.equal(rootStats.dispatches, 4);
          assertFloat32ArraysEqual(
            rootStats.value,
            [4, 3, 60 / 9, 9],
            "unregistered root fallback",
          );
        },
      );
      const rootEvents: string[] = [];
      const rootGpuA = runtime.tensor([12, 8, 9, 4], { shape: [4] });
      const rootGpuB = runtime.tensor([4, 4, 3, 2], { shape: [4] });
      const rootGpuC = runtime.tensor([3, 2, 5, 6], { shape: [4] });
      const rootGpuOutput = rootGpuA
        .add(rootGpuB)
        .mul(rootGpuC)
        .relu()
        .div(rootGpuA);
      const rootGpu = new GpuGraphExecutor(
        {
          tier: "webgpu",
          device: { limits: { maxStorageBuffersPerShaderStage: 8 } },
          frame: () => ({
            dispatch() {
              rootEvents.push("legacy-dispatch");
            },
            async submit() {},
          }),
        } as any,
        new Map([
          ["add", {}],
          ["mul", {}],
          ["relu", {}],
          ["div", {}],
        ]),
        new DeviceBuffers(),
        undefined,
        {
          onAnalysis(decision: any) {
            rootEvents.push(
              `decision:${decision.status}:${decision.code ?? ""}`,
            );
          },
          onResource(resource: string) {
            rootEvents.push(resource);
          },
        },
      );
      await rootGpu.execute(rootGpuOutput);
      await check(
        "GPU executor does not pack/fuse an eligible child below declined root",
        () => {
          assert.ok(
            rootEvents.includes("decision:refused:unsupported-chain"),
            JSON.stringify(rootEvents),
          );
          assert.equal(rootEvents.includes("fusion-pack"), false);
          assert.equal(rootEvents.includes("output-allocation"), false);
          assert.equal(
            rootEvents.filter((event) => event === "legacy-dispatch").length,
            4,
          );
        },
      );

      const short = runtime.tensor([7], { shape: [1] });
      const broadcast = a.add(short).mul(c).relu();
      const broadcastDecision = inspectElementwiseFusion(broadcast);
      const broadcastStats = await evaluateCpuWithStats(broadcast);
      await check(
        "broadcast graph declines fusion and preserves legacy output",
        () => {
          assert.equal(broadcastDecision.status, "refused");
          assert.equal(broadcastDecision.code, "shape-mismatch");
          assert.equal(broadcastStats.fusionProgram, undefined);
          assertFloat32ArraysEqual(
            broadcastStats.value,
            [8, 0, 20, 33],
            "broadcast fallback",
          );
        },
      );

      const parameter = runtime.createParameter(
        new Float32Array([2, 2, 2, 2]),
        { shape: [4] },
      );
      const trainableProduct = a.add(b).mul(parameter);
      const trainable = trainableProduct.relu();
      assert.equal(
        parameter.requiresGrad,
        true,
        "trainable fixture leaf is marked requiresGrad",
      );
      assert.equal(
        (trainableProduct.node as any).b,
        parameter,
        "trainable fixture uses that leaf as its mul input",
      );
      const gradientDecision = inspectElementwiseFusion(trainable);
      const gradientStats = await evaluateCpuWithStats(trainable);
      await check(
        "gradient graph declines fusion and retains previous semantics",
        () => {
          assert.equal(
            gradientDecision.status,
            "refused",
            JSON.stringify(gradientDecision),
          );
          assert.equal(
            gradientDecision.code,
            "requires-grad",
            JSON.stringify(gradientDecision),
          );
          assert.equal(gradientStats.fusionProgram, undefined);
          assertFloat32ArraysEqual(
            gradientStats.value,
            [6, 10, 14, 18],
            "gradient fallback",
          );
        },
      );
      await trainable.backward(runtime.tensor([1, 1, 1, 1], { shape: [4] }));
      const parameterGradient = parameter.grad;
      assert.ok(parameterGradient, "backward stores a parameter gradient");
      const gradientData = await parameterGradient.data();
      await check(
        "trainable fallback preserves backward parameter gradients",
        () => {
          assertFloat32ArraysEqual(
            gradientData,
            [3, 5, 7, 9],
            "trainable parameter gradient",
          );
        },
      );

      const materialized = a.add(b);
      await materialized.data();
      const materializedOutput = materialized.mul(c).relu();
      const materializedDecision = inspectElementwiseFusion(materializedOutput);
      const materializedStats = await evaluateCpuWithStats(materializedOutput);
      await check("materialized intermediate declines fusion", () => {
        assert.equal(materializedDecision.status, "refused");
        assert.equal(materializedDecision.code, "materialized-intermediate");
        assert.equal(materializedStats.fusionProgram, undefined);
        assertFloat32ArraysEqual(
          materializedStats.value,
          [3, 0, 14, 27],
          "materialized fallback",
        );
      });

      const shared = a.add(b);
      const outsideConsumer = shared.neg();
      assert.ok(outsideConsumer);
      const observable = shared.mul(c).relu();
      const observableDecision = inspectElementwiseFusion(observable);
      const observableStats = await evaluateCpuWithStats(observable);
      await check("externally consumed intermediate declines fusion", () => {
        assert.equal(observableDecision.status, "refused");
        assert.equal(observableDecision.code, "observable-intermediate");
        assert.equal(observableStats.fusionProgram, undefined);
        assertFloat32ArraysEqual(
          observableStats.value,
          [3, 0, 14, 27],
          "observable fallback",
        );
      });

      const alias = a.add(a).mul(c).relu();
      const aliasDecision = inspectElementwiseFusion(alias);
      const aliasStats = await evaluateCpuWithStats(alias);
      await check(
        "repeated read-only Tensor leaf remains eligible and deduplicated",
        () => {
          assert.equal(aliasDecision.status, "fused");
          assert.equal(aliasDecision.operations, 3);
          assert.ok(aliasStats.fusionProgram);
          assert.equal(aliasStats.dispatches, 1);
          assertFloat32ArraysEqual(
            aliasStats.value,
            [2, 0, 12, 24],
            "aliased leaf fallback",
          );
        },
      );

      const internalA = runtime.tensor([2, 3, -1, 4], { shape: [4] });
      const internalB = runtime.tensor([5, -2, 4, -3], { shape: [4] });
      const internalC = runtime.tensor([2, 3, -2, 5], { shape: [4] });
      const internal = internalA.add(internalB);
      const fusedRoot = internal.mul(internalC).relu();
      const internalDecision = inspectElementwiseFusion(fusedRoot);
      const fusedRootStats = await evaluateCpuWithStats(fusedRoot);
      const internalData = await internal.data();
      await check(
        "fused root preserves an internal Tensor read after execution",
        () => {
          assert.equal(internalDecision.status, "fused");
          assert.equal(fusedRootStats.dispatches, 1);
          assertFloat32ArraysEqual(
            fusedRootStats.value,
            [14, 3, 0, 5],
            "fused root output",
          );
          assertFloat32ArraysEqual(
            internalData,
            [7, 1, 3, 1],
            "internal Tensor data after fused root",
          );
        },
      );
    });

    await recorder.run(
      "T12-device-storage-limit-before-resources",
      async (check) => {
        const candidate = baseCandidate;
        const overLimit = analyzeElementwiseFusion(candidate.output, {
          maxStorageBuffers: 3,
        });
        const equalLimit = analyzeElementwiseFusion(candidate.output, {
          maxStorageBuffers: 4,
        });
        const underLimit = analyzeElementwiseFusion(candidate.output, {
          maxStorageBuffers: 5,
        });
        await check(
          "planner rejects below required input-plus-output storage count",
          () => {
            assert.equal(
              "decision" in overLimit ? undefined : overLimit.code,
              "storage-buffer-limit",
            );
          },
        );
        await check(
          "planner accepts exactly-at and above storage count",
          () => {
            assert.ok("decision" in equalLimit);
            assert.ok("decision" in underLimit);
          },
        );

        const events: string[] = [];
        const gpuRuntime = {
          tier: "webgpu",
          device: { limits: { maxStorageBuffersPerShaderStage: 3 } },
          frame() {
            return {
              dispatch() {
                events.push("legacy-dispatch");
              },
              async submit() {
                events.push("submit");
              },
            };
          },
        };
        const executor = new GpuGraphExecutor(
          gpuRuntime as any,
          new Map([
            ["add", {}],
            ["mul", {}],
            ["relu", {}],
          ]),
          new DeviceBuffers(),
          undefined,
          {
            onAnalysis(decision: any) {
              events.push(`decision:${decision.status}:${decision.code ?? ""}`);
            },
            onResource(resource: string) {
              events.push(resource);
            },
          },
        );
        await executor.execute(candidate.output);
        await check(
          "real GPU executor reads device cap and refuses before fused allocation or pack",
          () => {
            assertStorageLimitBeforeResources(events);
            assert.ok(events.includes("legacy-dispatch"));
          },
        );
      },
    );

    await recorder.run("T13-distribution-entrypoint", async (check) => {
      const dist = await import("../dist/index.js");
      const distRuntime = await dist.device("cpu");
      try {
        const graph = buildFusionFixture(distRuntime, base);
        const decision = dist.inspectElementwiseFusion(graph.output);
        const actual = await graph.output.data();
        await check(
          "built dist artifact executes a supported fused graph",
          () => {
            assertFloat32ArraysEqual(
              actual,
              evaluateFusionFixtureF32(base.fixture, fixtureInputs(base)),
              "dist artifact",
            );
            assert.deepEqual(decision, {
              status: "fused",
              operations: 3,
              dispatches: 1,
            });
            assert.equal(distRuntime.lastDispatchCount, 1);
          },
        );
        const unsupportedA = distRuntime.tensor([1, 2, 3, 4], {
          shape: [4],
        });
        const unsupportedB = distRuntime.tensor([2, 3, 4, 5], {
          shape: [4],
        });
        const unsupportedC = distRuntime.tensor([1, -1, 2, 3], {
          shape: [4],
        });
        const unsupported = unsupportedA
          .sub(unsupportedB)
          .mul(unsupportedC)
          .relu();
        const unsupportedDecision = dist.inspectElementwiseFusion(unsupported);
        const unsupportedActual = await unsupported.data();
        await check(
          "built dist artifact refuses unsupported root with nodewise output",
          () => {
            assert.equal(unsupportedDecision.status, "refused");
            assert.equal(unsupportedDecision.code, "unsupported-chain");
            assertFloat32ArraysEqual(
              unsupportedActual,
              [0, 1, 0, 0],
              "dist unsupported fallback",
            );
            assert.equal(distRuntime.lastDispatchCount, 3);
          },
        );
      } finally {
        await distRuntime.dispose();
      }
    });

    const roundingInputs = makeFixtureWithInputs(base, {
      a: [16777216],
      b: [1],
      c: [3],
    });
    const roundingGraph = buildFusionFixture(runtime, roundingInputs);
    const f32Candidate = analyzeElementwiseFusion(roundingGraph.output);
    if (!("decision" in f32Candidate))
      throw new Error("Fault fixture f32 witness is not fusion eligible.");

    const freshInputs = makeFixtureWithInputs(base, {
      a: [9, 8, 7, 6],
      b: [1, 2, 3, 4],
      c: [1, 2, 3, 4],
    });
    const freshGraph = buildFusionFixture(runtime, freshInputs);
    const freshCandidate = analyzeElementwiseFusion(freshGraph.output);
    if (!("decision" in freshCandidate))
      throw new Error(
        "Fault fixture fresh-input graph is not fusion eligible.",
      );

    const diamondCase = FUSION_CASES.find(
      (item) => item.id === "diamond-shared-node",
    )!;
    const diamondGraph = buildFusionFixture(runtime, diamondCase);
    const diamondCandidate = analyzeElementwiseFusion(diamondGraph.output);
    if (!("decision" in diamondCandidate))
      throw new Error("Fault fixture diamond graph is not fusion eligible.");
    const diamondExpected = evaluateFusionFixtureF32(
      diamondCase.fixture,
      fixtureInputs(diamondCase),
    );

    const changedSourceExpected = Float32Array.from(
      base.inputs.a!,
      (_, index) => {
        const a = Math.fround(base.inputs.a![index]!);
        const b = Math.fround(base.inputs.b![index]!);
        const c = Math.fround(base.inputs.c![index]!);
        return Math.fround(Math.max(0, Math.fround(Math.fround(a - b) * c)));
      },
    );

    await verifyFusionMutations({
      check: async (id, label, assertion) => {
        await recorder.run(id, async (record) => record(label, assertion));
      },
      baseCandidate,
      baseProgram: elementwiseFusionProgram(baseCandidate),
      expectedHelpers: [...ELEMENTWISE_FUSION_REGISTRY.helpers.values()],
      helperFingerprint,
      changedSourceExpected,
      f32Candidate,
      f32Inputs: f32Candidate.leaves.map((leaf: any) => leaf.node.value),
      f32Expected: evaluateFusionFixtureF32(
        roundingInputs.fixture,
        fixtureInputs(roundingInputs),
      ),
      freshCandidate,
      freshExpected: evaluateFusionFixtureF32(
        freshInputs.fixture,
        fixtureInputs(freshInputs),
      ),
      diamondCandidate,
      diamondExpected,
      cspMutationOutput: cspMutationOutputForFault,
      cspMutationExpected: cspMutationExpectedForFault,
      evaluateCpuWithStats,
      staleInputs: baseCandidate.leaves.map((leaf: any) => leaf.node.value),
      createRegistry: createElementwiseFusionRegistry,
      generateElementwiseOpIr,
      elementwiseFusionProgram,
      elementwiseFusionCpuProgram,
      dispatchElementwiseFusionCpu,
      compileModuleJs,
      generateModuleJs,
      GpuGraphExecutor,
      DeviceBuffers,
      assertions: {
        moduleHelpers: assertModuleHelpers,
        entryOutput: assertEntryWritesFinalOutput,
        storage: assertOneEntryAndStorage,
        noCodegenFallback: assertNoCodegenFallback,
        output: assertFloat32ArraysEqual,
        storageLimit: assertStorageLimitBeforeResources,
        freshData: assertFloat32ArraysEqual,
        diamondOutput: assertDiamondOutput,
        diamondLocals: assertDiamondLocals,
        emittedCpuExecution: assertEmittedCpuExecution,
      },
    });
  } finally {
    await runtime.dispose();
  }

  const summary = recorder.resultFields();
  return {
    commit: input.commit,
    worktreeFingerprint: input.worktreeFingerprint,
    helperFingerprint,
    gpu: "not-run",
    tests: summary.tests,
    passed: summary.passed,
  };
}
