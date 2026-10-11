import { device, inspectElementwiseFusion } from "../dist/index.js";

const result = document.querySelector<HTMLPreElement>("#result");
function publishReport(report: unknown) {
  const body = JSON.stringify(report, null, 2);
  if (result) result.textContent = body;
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
const inputs = [
  [1, 2, 3, 4],
  [5, -3, 2, -8],
  [2, 1, -2, 0.5],
];
const expected = [12, 0, 0, 0];

function assertValues(actual: Float32Array, wanted: number[], label: string) {
  if (
    actual.length !== wanted.length ||
    [...actual].some((value, index) => Math.abs(value - wanted[index]!) > 1e-5)
  )
    throw new Error(`${label} mismatch: ${JSON.stringify([...actual])}`);
}

function graph(target: Awaited<ReturnType<typeof device>>) {
  const [a, b, c] = inputs.map((values) =>
    target.tensor(values, { shape: [4] }),
  );
  const sum = a!.add(b!);
  const product = sum.mul(c!);
  return { sum, product, output: product.relu() };
}

function readAdapterDetails(adapter: GPUAdapter) {
  const info = adapter.info;
  return {
    vendor: info?.vendor ?? "unavailable",
    architecture: info?.architecture ?? "unavailable",
    device: info?.device ?? "unavailable",
    description: info?.description ?? "unavailable",
    isFallbackAdapter:
      adapter.isFallbackAdapter ?? info?.isFallbackAdapter ?? "unavailable",
  };
}

try {
  if (!result) throw new Error("Missing page result element.");
  if (!navigator.gpu)
    throw new Error("WebGPU is unavailable; production GPU test was not run.");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter was returned.");
  const adapterInfo = readAdapterDetails(adapter);

  const cpu = await device({ device: "cpu" });
  const cpuGraph = graph(cpu);
  const cpuValues = await cpuGraph.output.data();
  assertValues(cpuValues, expected, "published dist CPU fused output");
  if (cpu.lastFusionDecision?.status !== "fused" || cpu.lastDispatchCount !== 1)
    throw new Error("Published dist CPU did not run one fused dispatch.");
  await cpu.dispose();

  const gpu = await device({ device: "webgpu" });
  if (gpu.backend.name !== "webgpu")
    throw new Error(`Explicit WebGPU selected ${gpu.backend.name}.`);
  const gpuGraph = graph(gpu);
  const gpuValues = await gpuGraph.output.data();
  assertValues(gpuValues, expected, "published dist WebGPU fused output");
  if (gpu.lastFusionDecision?.status !== "fused" || gpu.lastDispatchCount !== 1)
    throw new Error("Published dist WebGPU did not run one fused dispatch.");

  const left = gpu.tensor(inputs[0]!, { shape: [4] });
  const right = gpu.tensor(inputs[1]!, { shape: [4] });
  const scale = gpu.tensor(inputs[2]!, { shape: [4] });
  const sum = left.add(right);
  const branchWitness = sum.neg();
  const refused = sum.mul(scale).relu();
  const refusedValues = await refused.data();
  assertValues(refusedValues, expected, "published dist refused graph");
  if (!branchWitness) throw new Error("The branch witness was not retained.");
  if (
    gpu.lastFusionDecision?.status !== "refused" ||
    gpu.lastFusionDecision.code !== "observable-intermediate" ||
    gpu.lastDispatchCount !== 3
  )
    throw new Error("Published dist fallback did not use three dispatches.");

  publishReport({
    status: "passed",
    importedFrom: "dist/index.js",
    adapter: adapterInfo,
    backend: gpu.backend,
    fused: {
      cpu: [...cpuValues],
      gpu: [...gpuValues],
      cpuDispatches: 1,
      gpuDispatches: 1,
    },
    refused: {
      decision: gpu.lastFusionDecision,
      output: [...refusedValues],
      dispatches: gpu.lastDispatchCount,
    },
  });
  await gpu.dispose();
} catch (error) {
  if (result) publishReport({ status: "failed", error: String(error) });
  throw error;
}
