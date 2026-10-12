// The time of one MNIST training step on a WebGPU device that is software here (Chromium with
// SwiftShader), and the calls the device is given for it. The step is the one of
// `examples/mnist`: zeroGrad, forward, loss, backward, Adam, on a batch of 64 images.
//
//   npm run build
//   TYPESHADE_CHROMIUM=/path/to/chrome node scripts/bench-step.mjs [--dist dist] [--steps 20] [--warmup 5]
//
// `--dist` names the `dist` of another build of TypeNN, whose Tensor API is the same, so that two
// builds run in the same harness. What it measures is host wall time on a software device, from the
// call of `zeroGrad()` to the end of `disposeGraph()`: it says what a step costs here, and not how
// fast a hardware GPU trains.

import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { inPage } from "../tests/gpu/_browser.mjs";

const { values: flags } = parseArgs({
  options: {
    dist: { type: "string", default: "dist" },
    steps: { type: "string", default: "20" },
    warmup: { type: "string", default: "5" },
    device: { type: "string", default: "webgpu" },
  },
});

/** Runs in the page. */
async function measure({ steps, warmup, device: preference }) {
  const nn = await import("/dist/index.js");
  const device = await nn.device(preference);
  try {
    const model = new nn.Sequential(
      new nn.Flatten(),
      new nn.Linear(784, 128, { seed: 101, weightScale: 1 / Math.sqrt(784) }),
      new nn.ReLU(),
      new nn.Dropout(0.2),
      new nn.Linear(128, 10, { seed: 202, weightScale: 1 / Math.sqrt(128) }),
    ).to(device);
    const parameters = model.parameters();
    const optimizer = nn.optim.Adam(parameters, { lr: 0.001 });
    const criterion = new nn.CrossEntropyLoss();
    let state = 7;
    const next = () =>
      (state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 2 ** 32;
    const images = Float32Array.from({ length: 64 * 784 }, next);
    const labels = Array.from({ length: 64 }, (_, row) => row % 10);
    const step = async () => {
      await optimizer.zeroGrad();
      const logits = model.forward(
        device.tensor(images, { shape: [64, 1, 28, 28] }),
      );
      const loss = criterion.forward(logits, labels);
      await loss.data();
      await loss.backward();
      await optimizer.step();
      await loss.disposeGraph(parameters);
    };
    for (let i = 0; i < warmup; i++) await step();
    // The calls the device is given for one step.
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
      await step();
    } finally {
      GPUQueue.prototype.submit = submit;
      GPUComputePassEncoder.prototype.dispatchWorkgroups = dispatchWorkgroups;
    }
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
    const info = device.backend;
    return {
      backend: info,
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
  },
  { dist },
);
console.log(
  `MNIST training step, batch 64, ${flags.steps} steps after ${flags.warmup} warm-up steps, on ${result.backend.name} (SwiftShader), build ${dist}`,
);
console.log(
  `median ${result.medianMs.toFixed(1)} ms, least ${result.minMs.toFixed(1)} ms, most ${result.maxMs.toFixed(1)} ms; the device was given ${result.seen.dispatches} dispatches in ${result.seen.submissions} submissions for one step`,
);
console.log(
  JSON.stringify({
    dist,
    ...result,
    times: result.times.map((t) => Number(t.toFixed(2))),
  }),
);
