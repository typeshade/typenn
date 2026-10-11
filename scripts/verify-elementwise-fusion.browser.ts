import { compileModule } from "typeshade";
import { device } from "../src/index.ts";
import { analyzeElementwiseFusion } from "../src/runtime/elementwise-fusion.ts";
import { evaluateCpuUnfusedWithStats } from "../src/runtime/cpu-executor.ts";
import {
  dispatchElementwiseFusionCpu,
  elementwiseFusionCpuProgram,
  elementwiseFusionGpuProgram,
  elementwiseFusionProgram,
} from "../src/runtime/elementwise-fusion-ir.ts";

const result = document.querySelector("#result");
function publishReport(report) {
  const body = JSON.stringify(report, null, 2);
  result.textContent = body;
  const params = new URLSearchParams(location.search);
  if (params.get("report") === "1")
    fetch(
      `/__typenn_test_report?tag=${encodeURIComponent(params.get("tag") ?? "verification")}`,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body,
      },
    ).catch((error) =>
      console.warn("Could not post verification report:", error),
    );
}
const sampleValues = [
  [1, 2, 3, 4],
  [5, -3, 2, -8],
  [2, 1, -2, 0.5],
];
const expectedValues = [12, 0, 0, 0];
const assertParity = (actual, expected, label) => {
  if (
    actual.length !== expected.length ||
    actual.some(
      (value, index) =>
        Math.abs(value - expected[index]) >
        1e-5 * Math.max(1, Math.abs(expected[index])),
    )
  ) {
    throw new Error(
      `${label} parity failed: ${JSON.stringify(actual)} vs ${JSON.stringify(expected)}.`,
    );
  }
};
const expectedFor = (values) =>
  values[0].map((_, index) =>
    Math.max(0, (values[0][index] + values[1][index]) * values[2][index]),
  );
const makeGraph = (target, values = sampleValues, order = "forward") => {
  const shape = [values[0].length];
  const a = target.tensor(values[0], { shape });
  const b = target.tensor(values[1], { shape });
  const c = target.tensor(values[2], { shape });
  const add = order === "reverse" ? b.add(a) : a.add(b);
  const product = order === "reverse" ? c.mul(add) : add.mul(c);
  return { a, b, c, add, product, output: product.relu() };
};
const readAdapterDetails = (adapter) => {
  const info = adapter.info;
  return {
    vendor: info?.vendor ?? "unavailable",
    architecture: info?.architecture ?? "unavailable",
    device: info?.device ?? "unavailable",
    description: info?.description ?? "unavailable",
    isFallbackAdapter:
      adapter.isFallbackAdapter ?? info?.isFallbackAdapter ?? "unavailable",
  };
};

try {
  result.textContent = "Requesting the WebGPU adapter…";
  if (!navigator.gpu) throw new Error("WebGPU is unavailable in this browser.");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter)
    throw new Error("The browser did not provide a WebGPU adapter.");
  const adapterInfo = readAdapterDetails(adapter);

  result.textContent = "Checking generated TypeNN CPU execution…";
  const cpu = await device({ device: "cpu" });
  if (cpu.backend.name !== "cpu")
    throw new Error(`Expected CPU; TypeNN selected ${cpu.backend.name}.`);
  const cpuGraph = makeGraph(cpu);
  const cpuCandidate = analyzeElementwiseFusion(cpuGraph.output);
  if (!("decision" in cpuCandidate))
    throw new Error(
      `CPU graph was not eligible: ${JSON.stringify(cpuCandidate)}.`,
    );

  const programBuildStartedAt = performance.now();
  const sharedProgram = elementwiseFusionProgram(cpuCandidate);
  const cpuProgramBuildMs = performance.now() - programBuildStartedAt;
  const generatedCpuProgram = elementwiseFusionCpuProgram(cpuCandidate);
  const interpreterCompileStartedAt = performance.now();
  const sameIrInterpreter = compileModule(sharedProgram.module, {
    precision: "f32",
  });
  const typeShadeInterpreterCompileMs =
    performance.now() - interpreterCompileStartedAt;
  const oracleValues = dispatchElementwiseFusionCpu(
    sameIrInterpreter,
    sharedProgram.inputNames,
    cpuCandidate.leaves.map((leaf) => leaf.node.value),
    4,
  );

  const cpuFusionStartedAt = performance.now();
  const cpuValues = [...(await cpuGraph.output.data())];
  const cpuFirstFusionMaterializationMs =
    performance.now() - cpuFusionStartedAt;
  const cpuDecision = cpu.lastFusionDecision;
  const cpuDispatchCount = cpu.lastDispatchCount;
  if (
    cpuDecision?.status !== "fused" ||
    cpuDispatchCount !== 1 ||
    (generatedCpuProgram.cpuBackend !== "emitted-js" &&
      generatedCpuProgram.cpuBackend !== "csp-interpreter-fallback")
  ) {
    throw new Error(
      `Expected one shared-IR CPU dispatch; got ${JSON.stringify({ cpuDecision, cpuDispatchCount, cpuBackend: generatedCpuProgram.cpuBackend })}.`,
    );
  }
  assertParity(cpuValues, expectedValues, "TypeNN CPU shared IR");
  assertParity(cpuValues, [...oracleValues], "TypeShade same-IR interpreter");

  const unfusedGraph = makeGraph(cpu);
  const unfused = await evaluateCpuUnfusedWithStats(unfusedGraph.output);
  assertParity([...unfused.value], cpuValues, "TypeNN nodewise CPU reference");
  if (unfused.dispatches !== 3)
    throw new Error(
      `Expected 3 nodewise CPU dispatches; got ${unfused.dispatches}.`,
    );
  await cpu.dispose();

  result.textContent = "Opening an explicitly requested WebGPU device…";
  const fusedDeviceOpenStartedAt = performance.now();
  let gpu = await device({ device: "webgpu" });
  const coldFusedDeviceOpenMs = performance.now() - fusedDeviceOpenStartedAt;
  if (gpu.backend.name !== "webgpu")
    throw new Error(`Expected WebGPU; TypeNN selected ${gpu.backend.name}.`);

  const gpuGraph = makeGraph(gpu);
  result.textContent = "Dispatching the graph-derived shared TypeShade module…";
  const coldFirstFusedStartedAt = performance.now();
  const gpuValues = [...(await gpuGraph.output.data())];
  const coldFirstFusedMaterializationMs =
    performance.now() - coldFirstFusedStartedAt;
  const gpuDecision = gpu.lastFusionDecision;
  const gpuDispatchCount = gpu.lastDispatchCount;
  const gpuProgram = elementwiseFusionGpuProgram(cpuCandidate);
  if (gpuProgram.module !== sharedProgram.module)
    throw new Error("CPU and GPU artifacts did not share one ModuleDecl.");
  if (gpuDecision?.status !== "fused" || gpuDispatchCount !== 1)
    throw new Error(
      `Expected one actual WebGPU dispatch; got ${JSON.stringify({ gpuDecision, gpuDispatchCount })}.`,
    );
  assertParity(gpuValues, expectedValues, "TypeNN WebGPU shared IR");
  assertParity(gpuValues, cpuValues, "WebGPU vs TypeNN generated CPU");
  assertParity(gpuValues, [...oracleValues], "WebGPU vs same-IR interpreter");

  result.textContent = "Checking later source-intermediate materialization…";
  const addValues = [...(await gpuGraph.add.data())];
  const addDispatchCount = gpu.lastDispatchCount;
  const productValues = [...(await gpuGraph.product.data())];
  const productDispatchCount = gpu.lastDispatchCount;
  assertParity(addValues, [6, -1, 5, -4], "later add intermediate");
  assertParity(productValues, [12, -1, -10, -2], "later multiply intermediate");

  await gpu.dispose();
  result.textContent =
    "Checking GPU bounds, operand order, and value rebinding…";
  const longValues = [
    Array.from({ length: 67 }, (_, index) => (index % 5) - 2),
    Array.from({ length: 67 }, (_, index) => (index % 7) - 3),
    Array.from({ length: 67 }, (_, index) => (index % 3) - 1),
  ];
  const reboundValues = [
    [3, -2, 4, 1],
    [2, 5, -1, -3],
    [2, -1, -2, 4],
  ];
  const gpuCases = [
    { label: "length67", values: longValues, order: "forward" },
    { label: "reordered-operands", values: sampleValues, order: "reverse" },
    {
      label: "new-values-same-signature",
      values: reboundValues,
      order: "forward",
    },
  ];
  const nodewiseReferences = new Map();
  const referenceDevice = await device({ device: "cpu" });
  for (const testCase of gpuCases) {
    const reference = await evaluateCpuUnfusedWithStats(
      makeGraph(referenceDevice, testCase.values, testCase.order).output,
    );
    if (reference.dispatches !== 3)
      throw new Error(
        `${testCase.label} CPU reference used ${reference.dispatches} dispatches, expected 3.`,
      );
    assertParity(
      [...reference.value],
      expectedFor(testCase.values),
      `${testCase.label} TypeNN nodewise CPU reference`,
    );
    nodewiseReferences.set(testCase.label, [...reference.value]);
  }
  await referenceDevice.dispose();

  gpu = await device({ device: "webgpu" });
  const gpuComparisons = [];
  for (const testCase of gpuCases) {
    const graph = makeGraph(gpu, testCase.values, testCase.order);
    const candidate = analyzeElementwiseFusion(graph.output);
    if (!("decision" in candidate))
      throw new Error(
        `${testCase.label} GPU graph refused fusion: ${JSON.stringify(candidate)}.`,
      );
    if (
      testCase.label === "reordered-operands" &&
      (candidate.operations.join(",") !== "add,mul,relu" ||
        candidate.leaves[0] !== graph.c ||
        candidate.leaves[1] !== graph.b ||
        candidate.leaves[2] !== graph.a)
    )
      throw new Error(
        "Reordered operands did not retain graph-derived leaf slots.",
      );

    const moduleForGraph = elementwiseFusionGpuProgram(candidate).module;
    if (
      testCase.label !== "reordered-operands" &&
      moduleForGraph !== sharedProgram.module
    )
      throw new Error(
        `${testCase.label} did not reuse the same structural ModuleDecl.`,
      );
    const values = [...(await graph.output.data())];
    const expected = nodewiseReferences.get(testCase.label);
    assertParity(values, expected, `${testCase.label} WebGPU vs nodewise CPU`);
    if (
      gpu.lastFusionDecision?.status !== "fused" ||
      gpu.lastDispatchCount !== 1
    )
      throw new Error(
        `${testCase.label} GPU path used ${gpu.lastDispatchCount} dispatches or refused fusion.`,
      );
    gpuComparisons.push({
      label: testCase.label,
      shape: [testCase.values[0].length],
      operations: candidate.operations,
      leafSlots: candidate.leaves.map((_, slot) => `input${slot}`),
      reusedModule: moduleForGraph === sharedProgram.module,
      output: values,
      cpuNodewiseDispatches: 3,
      gpuDispatches: gpu.lastDispatchCount,
    });
  }
  await gpu.dispose();

  result.textContent = "Measuring a cold refused GPU graph…";
  const unfusedDeviceOpenStartedAt = performance.now();
  gpu = await device({ device: "webgpu" });
  const coldUnfusedDeviceOpenMs =
    performance.now() - unfusedDeviceOpenStartedAt;
  if (gpu.backend.name !== "webgpu")
    throw new Error(`Expected a fresh WebGPU device; got ${gpu.backend.name}.`);
  const coldUnfusedA = gpu.tensor([0, -1, 1, -2], { shape: [4] });
  const coldUnfusedB = gpu.tensor([2, 1, 2, 1], { shape: [4] });
  const coldUnfusedC = gpu.tensor([3, 3, 3, 3], { shape: [4] });
  const coldUnfusedSum = coldUnfusedA.add(coldUnfusedB);
  const branchWitness = coldUnfusedSum.neg();
  const coldUnfusedOutput = coldUnfusedSum.mul(coldUnfusedC).relu();
  const coldUnfusedStartedAt = performance.now();
  const coldUnfusedValues = [...(await coldUnfusedOutput.data())];
  const coldUnfusedFirstMaterializationMs =
    performance.now() - coldUnfusedStartedAt;
  const coldUnfusedDecision = gpu.lastFusionDecision;
  const coldUnfusedDispatchCount = gpu.lastDispatchCount;
  if (!branchWitness) throw new Error("The branch witness was lost.");
  assertParity(coldUnfusedValues, [6, 0, 9, 0], "cold unfused TypeNN");
  if (
    coldUnfusedDecision?.status !== "refused" ||
    coldUnfusedDecision.code !== "observable-intermediate" ||
    coldUnfusedDispatchCount !== 3
  )
    throw new Error(
      `Expected a cold 3-dispatch refusal; got ${JSON.stringify({ coldUnfusedDecision, coldUnfusedDispatchCount })}.`,
    );

  const benchmarkSize = 32_768;
  const benchmarkA = Float32Array.from(
    { length: benchmarkSize },
    (_, index) => (index % 4) - 2,
  );
  const benchmarkB = new Float32Array(benchmarkSize).fill(2);
  const benchmarkC = Float32Array.from({ length: benchmarkSize }, (_, index) =>
    index % 2 === 0 ? 3 : -3,
  );
  async function sample(forceUnfused) {
    const left = gpu.tensor(benchmarkA, { shape: [benchmarkSize] });
    const right = gpu.tensor(benchmarkB, { shape: [benchmarkSize] });
    const scale = gpu.tensor(benchmarkC, { shape: [benchmarkSize] });
    const sum = left.add(right);
    const branch = forceUnfused ? sum.neg() : undefined;
    const value = sum.mul(scale).relu();
    const start = performance.now();
    const output = await value.data();
    assertParity([...output.slice(0, 4)], [0, 0, 6, 0], "benchmark sample");
    return {
      milliseconds: performance.now() - start,
      dispatches: gpu.lastDispatchCount,
      decision: gpu.lastFusionDecision,
      branch,
    };
  }
  for (let index = 0; index < 2; index++) {
    await sample(false);
    await sample(true);
  }
  const fusedSamples = [];
  const unfusedSamples = [];
  for (let index = 0; index < 5; index++) {
    fusedSamples.push(await sample(false));
    unfusedSamples.push(await sample(true));
  }
  const median = (samples) =>
    samples
      .map((sample) => sample.milliseconds)
      .sort((left, right) => left - right)[Math.floor(samples.length / 2)];
  if (
    fusedSamples.some((sample) => sample.dispatches !== 1) ||
    unfusedSamples.some(
      (sample) =>
        sample.dispatches !== 3 ||
        sample.decision?.status !== "refused" ||
        sample.decision.code !== "observable-intermediate",
    )
  )
    throw new Error(
      "The benchmark did not execute a 1-vs-3 dispatch comparison.",
    );

  publishReport({
    status: "passed",
    webgpu: {
      backend: gpu.backend,
      adapter: adapterInfo,
      decision: gpuDecision,
      actualDispatchCount: gpuDispatchCount,
      output: gpuValues,
    },
    cpu: {
      backend: cpuDecision && "cpu",
      decision: cpuDecision,
      actualDispatchCount: cpuDispatchCount,
      generatedCpuBackend: generatedCpuProgram.cpuBackend,
      programBuildMs: cpuProgramBuildMs,
      firstMaterializationMs: cpuFirstFusionMaterializationMs,
      sameModuleInterpreter: [...oracleValues],
      interpreterCompileMs: typeShadeInterpreterCompileMs,
      nodewiseReference: [...unfused.value],
      nodewiseDispatches: unfused.dispatches,
    },
    coldGpu: {
      fused: {
        deviceOpenMs: coldFusedDeviceOpenMs,
        firstMaterializationMs: coldFirstFusedMaterializationMs,
        dispatches: gpuDispatchCount,
      },
      unfused: {
        deviceOpenMs: coldUnfusedDeviceOpenMs,
        firstMaterializationMs: coldUnfusedFirstMaterializationMs,
        decision: coldUnfusedDecision,
        dispatches: coldUnfusedDispatchCount,
        output: coldUnfusedValues,
      },
      note: "Each path uses a fresh explicitly WebGPU TypeNN device. Device-open time includes TypeShade runtime setup and static pipeline preparation; fused first materialization additionally includes dynamic graph-module pack load/compute and readback. The public API exposes no isolated compile timer.",
    },
    laterIntermediates: {
      add: addValues,
      addDispatches: addDispatchCount,
      multiply: productValues,
      multiplyDispatches: productDispatchCount,
    },
    gpuComparisons,
    benchmark: {
      shape: [benchmarkSize],
      warmupsPerPath: 2,
      samplesPerPath: 5,
      fusedMedianMs: median(fusedSamples),
      unfusedMedianMs: median(unfusedSamples),
      fusedDispatches: 1,
      unfusedDispatches: 3,
      note: "Warm materialization timings; browser/device timings are observational.",
    },
  });
  await gpu.dispose();
} catch (error) {
  publishReport({ status: "failed", error: String(error) });
  throw error;
}
