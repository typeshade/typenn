import assert from "node:assert/strict";
import { compileModule } from "typeshade";
import { device, inspectElementwiseFusion } from "../src/index.js";
import {
  analyzeElementwiseFusion,
  type ElementwiseFusionCandidate,
} from "../src/runtime/elementwise-fusion.js";
import { evaluateCpuUnfusedWithStats } from "../src/runtime/cpu-executor.js";
import {
  dispatchElementwiseFusionCpu,
  elementwiseFusionCpuProgram,
  elementwiseFusionProgram,
} from "../src/runtime/elementwise-fusion-ir.js";

function tensor(
  runtime: Awaited<ReturnType<typeof device>>,
  values: number[],
  shape = [values.length],
) {
  return runtime.tensor(values, { shape });
}

function graph(
  runtime: Awaited<ReturnType<typeof device>>,
  values: number[][],
  order = "forward",
  shape = [values[0]!.length],
) {
  const [aValues, bValues, cValues] = values;
  const a = tensor(runtime, aValues!, shape);
  const b = tensor(runtime, bValues!, shape);
  const c = tensor(runtime, cValues!, shape);
  const add = order === "reverse" ? b.add(a) : a.add(b);
  const product = order === "reverse" ? c.mul(add) : add.mul(c);
  return { a, b, c, add, product, output: product.relu() };
}

function expected(values: number[][]): number[] {
  return values[0]!.map((_, index) =>
    Math.max(0, (values[0]![index]! + values[1]![index]!) * values[2]![index]!),
  );
}

function assertParity(
  actual: ArrayLike<number>,
  reference: number[],
  label: string,
) {
  assert.equal(actual.length, reference.length, `${label} length`);
  for (let index = 0; index < actual.length; index++)
    assert.ok(
      Math.abs(actual[index]! - reference[index]!) <=
        1e-5 * Math.max(1, Math.abs(reference[index]!)),
      `${label}[${index}]: ${actual[index]} != ${reference[index]}`,
    );
}

function sameIrOracle(candidate: ElementwiseFusionCandidate, length: number) {
  const program = elementwiseFusionProgram(candidate);
  const interpreter = compileModule(program.module, { precision: "f32" });
  const values = dispatchElementwiseFusionCpu(
    interpreter,
    program.inputNames,
    candidate.leaves.map((leaf) => leaf.node.value),
    length,
  );
  return { program, values };
}

export async function runElementwiseFusionVerification(): Promise<void> {
  const runtime = await device("cpu");
  const baseValues = [
    [1, 2, 3, 4],
    [5, -3, 2, -8],
    [2, 1, -2, 0.5],
  ];
  const first = graph(runtime, baseValues);
  const firstCandidate = analyzeElementwiseFusion(first.output);
  assert.ok("decision" in firstCandidate);
  assert.deepEqual(firstCandidate.operations, ["add", "mul", "relu"]);
  assert.deepEqual(firstCandidate.leaves, [first.a, first.b, first.c]);
  assert.deepEqual(inspectElementwiseFusion(first.output), {
    status: "fused",
    operations: 3,
    dispatches: 1,
  });
  const sameIr = sameIrOracle(firstCandidate, 4);
  const firstValues = [...(await first.output.data())];
  assertParity(firstValues, expected(baseValues), "TypeNN CPU generated IR");
  assertParity(
    firstValues,
    [...sameIr.values],
    "TypeShade CPU same-IR interpreter",
  );
  assert.equal(
    sameIr.program.module,
    elementwiseFusionProgram(firstCandidate).module,
  );
  assert.equal(runtime.lastFusionDecision?.status, "fused");
  assert.equal(runtime.lastDispatchCount, 1);
  const generatedCpu = elementwiseFusionCpuProgram(firstCandidate);
  assert.ok(
    generatedCpu.cpuBackend === "emitted-js" ||
      generatedCpu.cpuBackend === "csp-interpreter-fallback",
  );

  // The independent reference is TypeNN's original nodewise CPU executor.
  const referenceGraph = graph(runtime, baseValues);
  const unfused = await evaluateCpuUnfusedWithStats(referenceGraph.output);
  assertParity([...unfused.value], firstValues, "TypeNN unfused CPU reference");
  assert.equal(unfused.dispatches, 3);

  // Reversed graph operands change DFS leaf slots; runtime bindings follow them.
  const reverse = graph(runtime, baseValues, "reverse");
  const reverseCandidate = analyzeElementwiseFusion(reverse.output);
  assert.ok("decision" in reverseCandidate);
  assert.deepEqual(reverseCandidate.operations, ["add", "mul", "relu"]);
  assert.deepEqual(reverseCandidate.leaves, [reverse.c, reverse.b, reverse.a]);
  const reversedValues = [...(await reverse.output.data())];
  assertParity(reversedValues, expected(baseValues), "reordered leaf binding");
  const reversedOracle = sameIrOracle(reverseCandidate, 4);
  assertParity(
    reversedValues,
    [...reversedOracle.values],
    "reordered shared IR",
  );
  assert.equal(runtime.lastDispatchCount, 1);

  // The bounds guard must handle rounded workgroups beyond tensor length.
  const longValues = [
    Array.from({ length: 67 }, (_, index) => (index % 5) - 2),
    Array.from({ length: 67 }, (_, index) => (index % 7) - 3),
    Array.from({ length: 67 }, (_, index) => (index % 3) - 1),
  ];
  const longGraph = graph(runtime, longValues, "forward", [67]);
  const longOutput = [...(await longGraph.output.data())];
  assert.deepEqual(longOutput, expected(longValues));
  assert.equal(runtime.lastDispatchCount, 1);

  // A fused result must not replace source intermediates for later reads.
  assert.deepEqual([...(await first.add.data())], [6, -1, 5, -4]);
  assert.deepEqual([...(await first.product.data())], [12, -1, -10, -2]);

  const branch = graph(runtime, baseValues);
  const branchWitness = branch.add.neg();
  assert.ok(branchWitness);
  assert.equal(
    inspectElementwiseFusion(branch.output).code,
    "observable-intermediate",
  );
  assertParity(
    [...(await branch.output.data())],
    expected(baseValues),
    "branch fallback",
  );
  assert.equal(runtime.lastFusionDecision?.status, "refused");
  assert.equal(runtime.lastFusionDecision?.code, "observable-intermediate");
  assert.equal(runtime.lastDispatchCount, 3);

  const materialized = graph(runtime, baseValues);
  assert.deepEqual([...(await materialized.add.data())], [6, -1, 5, -4]);
  assert.equal(
    inspectElementwiseFusion(materialized.output).code,
    "materialized-intermediate",
  );
  assertParity(
    [...(await materialized.output.data())],
    expected(baseValues),
    "materialized intermediate fallback",
  );
  assert.equal(runtime.lastDispatchCount, 2);

  const aliasA = tensor(runtime, baseValues[0]!);
  const aliasC = tensor(runtime, baseValues[2]!);
  const aliasedOutput = aliasA.add(aliasA).mul(aliasC).relu();
  assert.equal(inspectElementwiseFusion(aliasedOutput).code, "aliased-leaf");

  const shapeA = tensor(runtime, baseValues[0]!);
  const shapeShort = tensor(runtime, [1]);
  const shapeC = tensor(runtime, baseValues[2]!);
  assert.equal(
    inspectElementwiseFusion(shapeA.add(shapeShort).mul(shapeC).relu()).code,
    "shape-mismatch",
  );

  const unsupportedA = tensor(runtime, baseValues[0]!);
  const unsupportedB = tensor(runtime, baseValues[1]!);
  const unsupportedC = tensor(runtime, baseValues[2]!);
  assert.equal(
    inspectElementwiseFusion(
      unsupportedA.sub(unsupportedB).mul(unsupportedC).relu(),
    ).code,
    "unsupported-chain",
  );

  const parameterA = tensor(runtime, baseValues[0]!);
  const parameterB = tensor(runtime, baseValues[1]!);
  const parameter = runtime.createParameter(new Float32Array([2, 2, 2, 2]), {
    shape: [4],
  });
  const parameterOutput = parameterA.add(parameterB).mul(parameter).relu();
  assert.equal(inspectElementwiseFusion(parameterOutput).code, "requires-grad");
  assertParity(
    [...(await parameterOutput.data())],
    [12, 0, 10, 0],
    "requiresGrad fallback",
  );
  assert.equal(runtime.lastDispatchCount, 3);

  await runtime.dispose();
  process.stdout.write(
    "Graph-derived shared TypeShade CPU fusion, oracle parity, fallback, and dispatch checks passed.\n",
  );
}
