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
const queryParams = new URLSearchParams(location.search);
const extendedHardwareBenchmark = queryParams.get("benchmark") === "1";
const runtimeAdapterOpens = [];
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
  if (actual.length !== expected.length)
    throw new Error(
      `${label} length mismatch: ${actual.length} vs ${expected.length}.`,
    );
  let maxAbsoluteError = 0;
  for (let index = 0; index < actual.length; index++) {
    const value = actual[index];
    const reference = expected[index];
    if (!Number.isFinite(value) || !Number.isFinite(reference))
      throw new Error(
        `${label}[${index}] is nonfinite: ${value} vs ${reference}.`,
      );
    const error = Math.abs(Math.fround(value) - Math.fround(reference));
    maxAbsoluteError = Math.max(maxAbsoluteError, error);
    if (!Object.is(Math.fround(value), Math.fround(reference)))
      throw new Error(
        `${label}[${index}] mismatch: ${value} vs ${reference} (error ${error}).`,
      );
  }
  return { maxAbsoluteError, comparedElementCount: actual.length };
};
const expectedFor = (values) =>
  values[0].map((_, index) => {
    const sum = Math.fround(
      Math.fround(values[0][index]) + Math.fround(values[1][index]),
    );
    const product = Math.fround(sum * Math.fround(values[2][index]));
    return Math.fround(Math.max(0, product));
  });
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
const openWebgpuWithAdapterCapture = async () => {
  const gpuApi = navigator.gpu;
  const originalDescriptor = Object.getOwnPropertyDescriptor(
    gpuApi,
    "requestAdapter",
  );
  if (originalDescriptor && !originalDescriptor.configurable)
    throw new Error(
      "Cannot instrument the GPU adapter request for verification.",
    );
  const originalRequest = gpuApi.requestAdapter;
  const requests = [];
  Object.defineProperty(gpuApi, "requestAdapter", {
    configurable: true,
    enumerable: originalDescriptor?.enumerable ?? false,
    writable: true,
    value: function (...args) {
      return Reflect.apply(originalRequest, this, args).then((adapter) => {
        requests.push({
          argumentCount: args.length,
          options: args[0] ?? null,
          adapter,
        });
        return adapter;
      });
    },
  });
  try {
    const runtimeDevice = await device({ device: "webgpu" });
    return { runtimeDevice, requests };
  } finally {
    if (originalDescriptor)
      Object.defineProperty(gpuApi, "requestAdapter", originalDescriptor);
    else delete gpuApi.requestAdapter;
  }
};

const openRuntimeWebgpu = async (label) => {
  if (!extendedHardwareBenchmark) return await device({ device: "webgpu" });
  const capture = await openWebgpuWithAdapterCapture();
  const runtimeDevice = capture.runtimeDevice;
  if (runtimeDevice.backend.name !== "webgpu")
    throw new Error(
      `Hardware benchmark requires WebGPU; TypeNN selected ${runtimeDevice.backend.name} for ${label}.`,
    );
  if (capture.requests.length !== 1)
    throw new Error(
      `Expected exactly one Runtime adapter request for ${label}; captured ${capture.requests.length}.`,
    );
  const request = capture.requests[0];
  if (!request?.adapter || request.argumentCount !== 0)
    throw new Error(
      `Runtime did not make the expected zero-argument adapter request for ${label}.`,
    );
  const adapter = readAdapterDetails(request.adapter);
  if (
    adapter.vendor.toLowerCase() !== "nvidia" ||
    adapter.isFallbackAdapter !== false
  )
    throw new Error(
      `Opt-in hardware benchmark requires the actual non-fallback NVIDIA adapter for ${label}; got ${JSON.stringify(adapter)}.`,
    );
  runtimeAdapterOpens.push({
    ...adapter,
    label,
    runtimeBackend: runtimeDevice.backend.name,
    requestOptions: request.options,
    requestArgumentCount: request.argumentCount,
  });
  return runtimeDevice;
};

async function runOptInHardwareBenchmark(currentGpu, params) {
  const readIntegerParameter = (name, fallback, minimum, maximum) => {
    const raw = params.get(name);
    if (raw === null) return fallback;
    const value = Number(raw);
    if (!Number.isInteger(value) || value < minimum || value > maximum)
      throw new Error(
        `Query parameter ${name} must be an integer from ${minimum} through ${maximum}.`,
      );
    return value;
  };
  const length = readIntegerParameter("length", 32_768, 32_768, 1_048_576);
  if (length !== 32_768 && length !== 1_048_576)
    throw new Error("Opt-in benchmark length must be 32768 or 1048576.");
  const warmupsPerPath = readIntegerParameter("warmups", 5, 1, 10);
  const samplesPerPath = readIntegerParameter("samples", 20, 20, 40);
  const inputs = [
    Float32Array.from({ length }, (_, index) => (((index * 17) % 19) - 9) / 4),
    Float32Array.from({ length }, (_, index) => (((index * 7) % 13) - 6) / 2),
    Float32Array.from({ length }, (_, index) => (((index * 5) % 11) - 5) / 4),
  ];

  await currentGpu.dispose();
  const referenceDevice = await device({ device: "cpu" });
  const referenceGraph = makeGraph(referenceDevice, inputs);
  const referenceExecution = await evaluateCpuUnfusedWithStats(
    referenceGraph.output,
  );
  if (referenceExecution.dispatches !== 3)
    throw new Error(
      `Expected 3 CPU reference dispatches; got ${referenceExecution.dispatches}.`,
    );
  const referenceValues = [...referenceExecution.value];
  const referenceParity = assertParity(
    referenceValues,
    expectedFor(inputs),
    "independent full-size TypeNN CPU reference",
  );
  await referenceDevice.dispose();

  const gpu = await openRuntimeWebgpu("timed-samples");

  async function measurePath(path) {
    const forceUnfused = path === "nodewise";
    const left = gpu.tensor(inputs[0], { shape: [length] });
    const right = gpu.tensor(inputs[1], { shape: [length] });
    const scale = gpu.tensor(inputs[2], { shape: [length] });
    const sum = left.add(right);
    const branchWitness = forceUnfused ? sum.neg() : undefined;
    const value = sum.mul(scale).relu();
    const startedAt = performance.now();
    const output = await value.data();
    const milliseconds = performance.now() - startedAt;
    const dispatches = gpu.lastDispatchCount;
    const decision = gpu.lastFusionDecision;
    if (forceUnfused && !branchWitness)
      throw new Error("The nodewise branch witness was lost.");
    const parity = assertParity(output, referenceValues, `${path} full output`);
    if (path === "fused") {
      if (decision?.status !== "fused" || dispatches !== 1)
        throw new Error(
          `Fused benchmark sample used ${dispatches} dispatches or was refused.`,
        );
    } else if (
      decision?.status !== "refused" ||
      decision.code !== "observable-intermediate" ||
      dispatches !== 3
    ) {
      throw new Error(
        `Nodewise benchmark sample used ${dispatches} dispatches or did not refuse fusion.`,
      );
    }
    return {
      milliseconds,
      dispatches,
      maxAbsoluteError: parity.maxAbsoluteError,
      comparedElementCount: parity.comparedElementCount,
    };
  }

  const warmupOrder = [];
  for (let index = 0; index < warmupsPerPath; index++) {
    const order =
      index % 2 === 0 ? ["fused", "nodewise"] : ["nodewise", "fused"];
    warmupOrder.push(order);
    for (const path of order) await measurePath(path);
  }
  const fusedSamples = [];
  const nodewiseSamples = [];
  const sampleOrder = [];
  for (let index = 0; index < samplesPerPath; index++) {
    const order =
      index % 2 === 0 ? ["fused", "nodewise"] : ["nodewise", "fused"];
    sampleOrder.push(order);
    for (const path of order) {
      const sample = await measurePath(path);
      (path === "fused" ? fusedSamples : nodewiseSamples).push(sample);
    }
  }
  const median = (samples) => {
    const ordered = samples
      .map((sample) => sample.milliseconds)
      .sort((left, right) => left - right);
    const middle = Math.floor(ordered.length / 2);
    return ordered.length % 2 === 0
      ? (ordered[middle - 1] + ordered[middle]) / 2
      : ordered[middle];
  };

  return {
    gpu,
    report: {
      mode: "opt-in-hardware",
      length,
      warmupsPerPath,
      samplesPerPath,
      adapter: runtimeAdapterOpens.at(-1),
      runtimeAdapterOpens: [...runtimeAdapterOpens],
      reference: {
        backend: "TypeNN original nodewise CPU executor",
        dispatches: referenceExecution.dispatches,
        comparedElementCount: referenceValues.length,
        maxAbsoluteError: referenceParity.maxAbsoluteError,
      },
      warmupOrder,
      sampleOrder,
      fusedSamples,
      nodewiseSamples,
      fusedMedianMs: median(fusedSamples),
      nodewiseMedianMs: median(nodewiseSamples),
      note: "Every timed output is measured through data() with the same inputs and full-output parity checked afterward against one independent CPU nodewise reference per shape. Cold compilation/runtime setup is outside the timed sample window; sample order alternates.",
    },
  };
}

try {
  result.textContent = "Requesting the WebGPU adapter…";
  if (!navigator.gpu) throw new Error("WebGPU is unavailable in this browser.");
  const adapter = extendedHardwareBenchmark
    ? undefined
    : await navigator.gpu.requestAdapter();
  if (!extendedHardwareBenchmark && !adapter)
    throw new Error("The browser did not provide a WebGPU adapter.");
  const adapterInfo = adapter ? readAdapterDetails(adapter) : null;

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
    generatedCpuProgram.cpuBackend !== "emitted-js"
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
  let gpu = await openRuntimeWebgpu("cold-fused-and-primary-check");
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

  gpu = await openRuntimeWebgpu("bounds-and-rebinding-checks");
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
  gpu = await openRuntimeWebgpu("cold-unfused");
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

  let benchmark;
  if (extendedHardwareBenchmark) {
    const benchmarkResult = await runOptInHardwareBenchmark(gpu, queryParams);
    gpu = benchmarkResult.gpu;
    benchmark = benchmarkResult.report;
  } else {
    const benchmarkSize = 32_768;
    const benchmarkA = Float32Array.from(
      { length: benchmarkSize },
      (_, index) => (index % 4) - 2,
    );
    const benchmarkB = new Float32Array(benchmarkSize).fill(2);
    const benchmarkC = Float32Array.from(
      { length: benchmarkSize },
      (_, index) => (index % 2 === 0 ? 3 : -3),
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
    benchmark = {
      shape: [benchmarkSize],
      warmupsPerPath: 2,
      samplesPerPath: 5,
      fusedMedianMs: median(fusedSamples),
      unfusedMedianMs: median(unfusedSamples),
      fusedDispatches: 1,
      unfusedDispatches: 3,
      note: "Warm materialization timings; browser/device timings are observational.",
    };
  }

  publishReport({
    status: "passed",
    webgpu: {
      backend: gpu.backend,
      adapter:
        benchmark?.mode === "opt-in-hardware" ? benchmark.adapter : adapterInfo,
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
    benchmark,
  });
  await gpu.dispose();
} catch (error) {
  publishReport({ status: "failed", error: String(error) });
  throw error;
}
