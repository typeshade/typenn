// The calls a WebGPU device is given for the graphs of TypeNN, and the time they take, on a device
// that is software here (Chromium with SwiftShader).
//
//   npm run build
//   TYPESHADE_CHROMIUM=/path/to/chrome node scripts/bench-step.mjs [--dist dist] [--scenario step|chain|counts]
//                                                                  [--steps 20] [--warmup 5] [--length 1048576]
//
// The scenarios:
//   step    one MNIST training step of `examples/mnist` (zeroGrad, forward, loss, backward, Adam) on a
//           batch of 64 images: the median time of `--steps` steps after `--warmup` steps.
//   chain   `relu((a + b) * c)` over `--length` elements, read back whole: the median time.
//   counts  the dispatches and the submissions for one `data()` of the chain, of a dense layer, of the
//           forward pass of the MNIST model, and of its backward pass (four `data()` calls, and the
//           one `step()` of an optimizer). No time.
//
// `--dist` names the `dist` of another build of TypeNN, whose Tensor API is the same, so that two
// builds run in the same harness. A time is host wall time on a software device, from the call of
// `zeroGrad()` to the end of `disposeGraph()`: it says what a step costs here, and not how fast a
// hardware GPU trains. A count is the calls of the page to `GPUQueue.prototype.submit` and
// `GPUComputePassEncoder.prototype.dispatchWorkgroups`, so it includes the submission that reads
// the result back.

import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { inPage } from "../tests/gpu/_browser.mjs";

const { values: flags } = parseArgs({
  options: {
    dist: { type: "string", default: "dist" },
    steps: { type: "string", default: "20" },
    warmup: { type: "string", default: "5" },
    device: { type: "string", default: "webgpu" },
    hardware: { type: "boolean", default: false },
    scenario: { type: "string", default: "step" },
    length: { type: "string", default: "1048576" },
  },
});

/** Runs in the page. */
async function measure({
  steps,
  warmup,
  device: preference,
  scenario,
  length,
  hardware,
}) {
  if (hardware && preference !== "webgpu")
    throw new Error("--hardware requires --device webgpu.");
  // The default harness forces SwiftShader. The hardware opt-in removes those
  // Chromium flags and refuses a software/fallback WebGPU adapter.
  let adapterInfo;
  if (hardware) {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error("A native WebGPU adapter is unavailable.");
    const info = adapter.info;
    const description = [info?.vendor, info?.architecture, info?.description]
      .filter(Boolean)
      .join(" ");
    if (
      adapter.isFallbackAdapter ||
      /swiftshader|llvmpipe|software|softpipe/i.test(description) ||
      description.length === 0
    )
      throw new Error("Expected hardware WebGPU, got: " + description);
    adapterInfo = description;
  }
  const nn = await import("/dist/index.js");
  const device = await nn.device(preference);
  // The calls the device is given while `run` runs.
  const counted = async (run) => {
    const { submit } = GPUQueue.prototype;
    const { dispatchWorkgroups } = GPUComputePassEncoder.prototype;
    const seen = { dispatches: 0, submissions: 0 };
    GPUQueue.prototype.submit = function (...a) {
      seen.submissions++;
      return submit.apply(this, a);
    };
    GPUComputePassEncoder.prototype.dispatchWorkgroups = function (...a) {
      seen.dispatches++;
      return dispatchWorkgroups.apply(this, a);
    };
    try {
      await run();
    } finally {
      GPUQueue.prototype.submit = submit;
      GPUComputePassEncoder.prototype.dispatchWorkgroups = dispatchWorkgroups;
    }
    return seen;
  };
  let state = 7;
  const next = () =>
    (state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 2 ** 32;
  const vector = (size) =>
    device.tensor(Float32Array.from({ length: size }, next), {
      shape: [size],
    });
  // The model of `examples/mnist`, and the mean cross entropy of a batch of 64 images.
  const mnist = () =>
    new nn.Sequential(
      new nn.Flatten(),
      new nn.Linear(784, 128, { seed: 101, weightScale: 1 / Math.sqrt(784) }),
      new nn.ReLU(),
      new nn.Dropout(0.2),
      new nn.Linear(128, 10, { seed: 202, weightScale: 1 / Math.sqrt(128) }),
    ).to(device);
  const images = Float32Array.from({ length: 64 * 784 }, next);
  const labels = Array.from({ length: 64 }, (_, row) => row % 10);
  const lossOf = (model) =>
    new nn.CrossEntropyLoss().forward(
      model.forward(device.tensor(images, { shape: [64, 1, 28, 28] })),
      labels,
    );
  try {
    if (scenario === "counts") {
      const counts = {};
      const [a, b, c] = [vector(1024), vector(1024), vector(1024)];
      const chain = a.add(b).mul(c).relu();
      counts.chain = await counted(() => chain.data());
      const x = device.tensor(Float32Array.from({ length: 64 * 784 }, next), {
        shape: [64, 784],
      });
      const w = device.tensor(Float32Array.from({ length: 784 * 128 }, next), {
        shape: [784, 128],
      });
      const dense = x.matmul(w).add(vector(128)).relu();
      counts.denseLayer = await counted(() => dense.data());
      const model = mnist();
      const loss = lossOf(model);
      counts.mnistForward = await counted(() => loss.data());
      await loss.backward();
      const gradients = model.parameters().map((parameter) => parameter.grad);
      counts.mnistBackwardFourReads = await counted(async () => {
        for (const gradient of gradients) await gradient.data();
      });
      // A second graph, so that the gradients are not computed yet.
      const other = mnist();
      const otherLoss = lossOf(other);
      await otherLoss.data();
      await otherLoss.backward();
      const optimizer = new nn.SGD(other.parameters(), { lr: 0.1 });
      counts.mnistBackwardOptimizerStep = await counted(() => optimizer.step());
      return { backend: device.backend, adapterInfo, counts };
    }
    let step;
    let parameters = [];
    if (scenario === "chain") {
      const [a, b, c] = [vector(length), vector(length), vector(length)];
      step = async () => {
        const y = a.add(b).mul(c).relu();
        await y.data();
        await y.disposeGraph([a, b, c]);
      };
    } else {
      const model = mnist();
      parameters = model.parameters();
      const optimizer = nn.optim.Adam(parameters, { lr: 0.001 });
      step = async () => {
        await optimizer.zeroGrad();
        const loss = lossOf(model);
        await loss.data();
        await loss.backward();
        await optimizer.step();
        await loss.disposeGraph(parameters);
      };
    }
    for (let i = 0; i < warmup; i++) await step();
    const seen = await counted(step);
    const times = [];
    for (let i = 0; i < steps; i++) {
      const start = performance.now();
      await step();
      times.push(performance.now() - start);
    }
    const sorted = [...times].sort((p, q) => p - q);
    const median =
      steps % 2
        ? sorted[(steps - 1) / 2]
        : (sorted[steps / 2 - 1] + sorted[steps / 2]) / 2;
    return {
      backend: device.backend,
      adapterInfo,
      seen,
      medianMs: median,
      minMs: sorted[0],
      maxMs: sorted[steps - 1],
      times,
    };
  } finally {
    await device.dispose();
  }
}

const dist = resolve(flags.dist);
const { result } = await inPage(
  measure,
  {
    steps: Number(flags.steps),
    warmup: Number(flags.warmup),
    device: flags.device,
    scenario: flags.scenario,
    length: Number(flags.length),
    hardware: flags.hardware,
  },
  { dist, hardware: flags.hardware },
);
if (flags.scenario === "counts") {
  console.log(
    `The calls the device is given for one data(), on ${result.backend.name} (${flags.hardware ? result.adapterInfo : "SwiftShader"}), build ${dist}`,
  );
  for (const [name, seen] of Object.entries(result.counts))
    console.log(
      `${name}: ${seen.dispatches} dispatches in ${seen.submissions} submissions`,
    );
  console.log(JSON.stringify({ dist, ...result }));
} else {
  console.log(
    `${flags.scenario === "chain" ? `chain of ${flags.length} elements` : "MNIST training step, batch 64"}, ${flags.steps} runs after ${flags.warmup} warm-up runs, on ${result.backend.name} (${flags.hardware ? result.adapterInfo : "SwiftShader"}), build ${dist}`,
  );
  console.log(
    `median ${result.medianMs.toFixed(1)} ms, least ${result.minMs.toFixed(1)} ms, most ${result.maxMs.toFixed(1)} ms; the device was given ${result.seen.dispatches} dispatches in ${result.seen.submissions} submissions for one run`,
  );
  console.log(
    JSON.stringify({
      dist,
      ...result,
      times: result.times.map((t) => Number(t.toFixed(2))),
    }),
  );
}
