import { loadMNIST } from "./data.js";
import { train } from "./loop.js";

// The environment of Node, Bun and Deno.
declare const process: { env: Record<string, string | undefined> };

// The MNIST script: download the IDX files, train for five epochs, and print the test accuracy.
// MNIST_STEPS=20 stops after 20 training steps, MNIST_TEST_SAMPLES=1000 evaluates on the first
// 1,000 test images, and MNIST_DEVICE=cpu|webgpu|webgl2 picks the tier.
const steps = process.env.MNIST_STEPS;
const testSamples = process.env.MNIST_TEST_SAMPLES;
const device = process.env.MNIST_DEVICE;

await train(await loadMNIST(true), await loadMNIST(false), {
  ...(steps ? { steps: Number(steps) } : {}),
  ...(testSamples ? { testSamples: Number(testSamples) } : {}),
  ...(device ? { device: device as "cpu" | "webgpu" | "webgl2" } : {}),
  log: console.log,
});
