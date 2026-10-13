// End-to-end softmax/gradient benchmark. Default uses SwiftShader for correctness
// investigations; --hardware launches native WebGPU and REJECTS software adapters.
// Run the same script with --dist path/to/baseline/dist to compare two builds.
// Example (hardware GPU + installed Chromium):
//   npm run build
//   TYPESHADE_CHROMIUM=/path/to/chrome node scripts/bench-softmax.mjs --hardware
//   node scripts/bench-softmax.mjs --hardware --axis-sizes 10,64,128,512,2048
//
// Each sample includes graph creation, GPU execution, readback, and graph disposal;
// measurements are wall-clock milliseconds, NOT GPU timestamps or GPU kernel timings.

import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { inPage } from "../tests/gpu/_browser.mjs";

const { values: flags } = parseArgs({
  options: {
    dist: { type: "string", default: "dist" },
    hardware: { type: "boolean", default: false },
    rows: { type: "string", default: "32" },
    "axis-sizes": { type: "string", default: "10,63,64,128,512,2048" },
    steps: { type: "string", default: "10" },
    warmup: { type: "string", default: "3" },
  },
});
const options = {
  rows: Number(flags.rows),
  sizes: flags["axis-sizes"].split(",").map(Number),
  steps: Number(flags.steps),
  warmup: Number(flags.warmup),
  requireHardware: flags.hardware,
};
if (
  !Number.isSafeInteger(options.rows) ||
  options.rows < 1 ||
  !Number.isSafeInteger(options.steps) ||
  options.steps < 1 ||
  !Number.isSafeInteger(options.warmup) ||
  options.warmup < 0 ||
  options.sizes.length === 0 ||
  options.sizes.some((size) => !Number.isSafeInteger(size) || size < 1)
)
  throw new RangeError(
    "rows, axis sizes and steps must be positive safe integers.",
  );

async function measure(arg) {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("WebGPU adapter unavailable.");
  const info = adapter.info;
  const gpu = {
    vendor: info?.vendor || "unknown",
    architecture: info?.architecture || "unknown",
    description: info?.description || "unknown",
    device: info?.device || "unknown",
    fallback: adapter.isFallbackAdapter === true,
  };
  const identity = Object.values(gpu).join(" ").toLowerCase();
  const software = /swiftshader|llvmpipe|software|softpipe/.test(identity);
  if (
    arg.requireHardware &&
    (software ||
      gpu.fallback ||
      (gpu.vendor === "unknown" && gpu.description === "unknown"))
  )
    throw new Error(
      "Native hardware WebGPU was requested, but no verified hardware adapter was found: " +
        JSON.stringify(gpu),
    );

  const nn = await import("/dist/index.js");
  const device = await nn.device("webgpu");
  const samples = [];
  try {
    // A device that fell back to another tier measures that tier, so it is refused here.
    if (device.backend.name !== "webgpu")
      throw new Error(
        `The WebGPU tier was requested, but the device runs on ${device.backend.name}: ` +
          JSON.stringify(device.backend),
      );

    for (const axisSize of arg.sizes) {
      const count = arg.rows * axisSize;
      const input = device.tensor(
        Float32Array.from({ length: count }, (_, i) => Math.sin(i * 0.13) * 6),
        { shape: [arg.rows, axisSize] },
      );
      const upstream = device.tensor(
        Float32Array.from({ length: count }, (_, i) => Math.cos(i * 0.23)),
        { shape: [arg.rows, axisSize] },
      );
      for (const op of ["softmax", "logSoftmax"]) {
        for (const mode of ["forward", "forwardBackward"]) {
          const run = async () => {
            const y = input[op]();
            if (mode === "forward") {
              await y.data();
              await y.disposeGraph([input]);
            } else {
              const [dy] = nn.gradients(y, [input], upstream);
              await dy.data();
              await dy.disposeGraph([input, upstream]);
            }
            const report = device.lastReport;
            return {
              dispatches: report?.dispatches ?? 0,
              submissions: report?.submissions ?? 0,
            };
          };
          for (let k = 0; k < arg.warmup; k++) await run();
          const times = [];
          let plan = {};
          for (let k = 0; k < arg.steps; k++) {
            const before = performance.now();
            plan = await run();
            times.push(performance.now() - before);
          }
          times.sort((a, b) => a - b);
          const n = times.length;
          samples.push({
            axisSize,
            rows: arg.rows,
            op,
            mode,
            medianMs:
              n % 2
                ? times[(n - 1) / 2]
                : (times[n / 2 - 1] + times[n / 2]) / 2,
            minMs: times[0],
            maxMs: times[n - 1],
            ...plan,
          });
        }
      }
      await input.dispose();
      await upstream.dispose();
    }
  } finally {
    await device.dispose();
  }
  return { backend: device.backend.name, gpu, software, samples };
}

const dist = resolve(flags.dist);
const { result, logged } = await inPage(measure, options, {
  dist,
  hardware: flags.hardware,
});
console.log(
  JSON.stringify(
    {
      build: dist,
      requestedHardware: flags.hardware,
      methodology: "end-to-end wall time including GPU readback",
      ...result,
      browserWarnings: logged,
    },
    null,
    2,
  ),
);
