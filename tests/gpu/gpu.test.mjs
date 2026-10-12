import test from "node:test";
import assert from "node:assert/strict";
import { inPage } from "./_browser.mjs";

// TypeNN on a real WebGPU device: headless Chromium with SwiftShader (see `_browser.mjs`). The
// numerics cases of `_cases.mjs` run on the CPU tier and on the GPU tier of one page and are held
// to the reference and to each other at the tolerances the cases carry (exact for the operations
// that are exact on a GPU, 1e-5 relative for the rest). The plans of the MNIST model are the ones
// `plan.test.mjs` asserts on a fake device, and the counts a report gives are held to the calls
// the page counts on `GPUQueue.prototype.submit` and `GPUComputePassEncoder.prototype.dispatchWorkgroups`.
//
//   npm run build
//   TYPESHADE_CHROMIUM=/path/to/chrome npm run test:gpu

/**
 * Cases whose values SwiftShader gives differently from the f32 reference by more than 1e-5
 * relative, on WebGPU and on WebGL2 alike. They are measured, not excused: WGSL allows `sin` and
 * `cos` an absolute error of 2^-11, and SwiftShader's worst over 4,000 inputs in [-3, 3] is
 * 1.9e-4 for both, 9.9e-5 for `tan` on [-1.2, 1.2] and 6.7e-5 for `acos` on [-0.99, 0.99]. The
 * comparison stays at 1e-5 and the cases stay red here, so any other case that differs fails the
 * test, and so does a driver that gets these right (the list is then stale).
 */
const SWIFTSHADER_TRIG = [
  "gradient of cos",
  "gradient of sin",
  "gradient of tan",
  "map acos",
  "map cos",
  "map sin",
  "map tan",
];

/** Runs in the page: every case that is not CPU-only on the CPU tier and on `tier`. */
async function compareTiers(tier) {
  const nn = await import("/dist/index.js");
  const lists = await import("/tests/_cases.mjs");
  const { difference } = await import("/tests/_reference.mjs");
  const cases = [
    ...lists.elementwiseCases(),
    ...lists.reductionAndMatmulCases(),
    ...lists.softmaxAndLossCases(),
    ...lists.spatialCases(),
  ].filter((testCase) => !testCase.cpuOnly);
  const run = async (preference) => {
    const device = await nn.device(preference);
    const results = new Map();
    try {
      for (const testCase of cases) {
        try {
          results.set(testCase.name, await testCase.run(nn, device));
        } catch (error) {
          results.set(testCase.name, { error: String(error) });
        }
      }
    } finally {
      await device.dispose();
    }
    return { name: device.backend.name, results };
  };
  const cpu = await run("cpu");
  const gpu = await run(tier);
  const failing = [];
  for (const testCase of cases) {
    const onGpu = gpu.results.get(testCase.name);
    const onCpu = cpu.results.get(testCase.name);
    if (onGpu.error || onCpu.error) {
      failing.push({ name: testCase.name, error: onGpu.error ?? onCpu.error });
      continue;
    }
    const reference = difference(
      onGpu.actual,
      onGpu.expected,
      testCase.gpuTolerance,
    );
    const againstCpu = difference(
      onGpu.actual,
      onCpu.actual,
      testCase.gpuTolerance,
    );
    if (reference || againstCpu)
      failing.push({ name: testCase.name, reference, againstCpu });
  }
  return { backend: gpu.name, cases: cases.length, failing };
}

for (const tier of ["webgpu", "webgl2"]) {
  test(`the numerics cases on ${tier} equal the CPU tier's and the reference`, async () => {
    const { result, logged } = await inPage(compareTiers, tier);
    assert.equal(result.backend, tier);
    assert.ok(result.cases > 400, `${result.cases} cases`);
    assert.deepEqual(
      result.failing.map((failure) => failure.name).sort(),
      SWIFTSHADER_TRIG,
      JSON.stringify(result.failing, null, 1),
    );
    // A WebGL2 driver says "GPU stall due to ReadPixels" about every read-back.
    assert.deepEqual(
      logged.filter((line) => !line.includes("GL Driver Message")),
      [],
    );
  });
}

/** Runs in the page: the plans of the chain, a dense layer and the MNIST model on a real device,
 *  and the calls the device was given. */
async function plansOnDevice() {
  const nn = await import("/dist/index.js");
  const { mnistStep, stated } = await import("/tests/_plans.mjs");
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
  const device = await nn.device("webgpu");
  const out = { backend: device.backend };
  try {
    const vector = (values) =>
      device.tensor(values, { shape: [values.length] });
    const chain = vector([1, 2, -3, 4])
      .add(vector([5, -3, 2, -8]))
      .mul(vector([2, 1, -2, 0.5]))
      .relu();
    let values;
    out.chainSeen = await counted(async () => {
      values = Array.from(await chain.data());
    });
    out.chain = { values, report: device.lastReport };

    const layer = device
      .tensor(new Float32Array(64 * 784), { shape: [64, 784] })
      .matmul(device.tensor(new Float32Array(784 * 128), { shape: [784, 128] }))
      .add(vector(new Float32Array(128)))
      .relu();
    out.layerSeen = await counted(async () => {
      await layer.data();
    });
    out.layer = stated(device.lastReport);

    const { loss, parameters } = mnistStep(nn, device);
    out.forwardSeen = await counted(async () => {
      await loss.data();
    });
    out.forward = stated(device.lastReport);
    await loss.backward();
    const before = device.reports.length;
    // The step of an optimizer reads the gradient of every parameter.
    out.backwardSeen = await counted(async () => {
      await new nn.SGD(parameters, { lr: 0.1 }).step();
    });
    out.backward = device.reports.slice(before).map(stated);
  } finally {
    await device.dispose();
  }
  return out;
}

test("the plans on a real device are the plans on the fake one, and the device is given what the reports count", async () => {
  const { MNIST_BACKWARD, MNIST_FORWARD } = await import("../_plans.mjs");
  const { result, logged } = await inPage(plansOnDevice);
  assert.deepEqual(result.backend, { provider: "typeshade", name: "webgpu" });
  // The chain: one fused dispatch, and the submission of the plan beside the one of the read-back.
  assert.deepEqual(result.chain, {
    values: [12, 0, 2, 0],
    report: {
      tier: "webgpu",
      dispatches: 1,
      submissions: 1,
      groups: [{ calls: ["add", "mul", "relu"], fused: true }],
    },
  });
  assert.deepEqual(result.chainSeen, { dispatches: 1, submissions: 2 });
  // A dense layer, `matmul`, the bias and `relu`, is one dispatch as well.
  assert.deepEqual(result.layer.groups, [
    { calls: ["matmul", "addExpand", "relu"], fused: true },
  ]);
  assert.deepEqual(result.layerSeen, { dispatches: 1, submissions: 2 });
  assert.deepEqual(result.forward, MNIST_FORWARD);
  assert.deepEqual(result.forwardSeen, { dispatches: 5, submissions: 2 });
  assert.deepEqual(result.backward, [MNIST_BACKWARD]);
  // One plan and the read-back of each of the four gradients.
  assert.deepEqual(result.backwardSeen, { dispatches: 10, submissions: 5 });
  assert.deepEqual(logged, []);
});

/** Runs in the page: the MNIST example for 30 steps on the CPU tier and on WebGPU. */
async function trainBoth() {
  const { train } = await import("/.typenn-build/mnist/loop-browser.mjs");
  const { syntheticMnist } = await import("/tests/_mnist.mjs");
  const data = [syntheticMnist(320, 21), syntheticMnist(100, 22)];
  const options = { batchSize: 32, steps: 30, shuffle: false };
  const cpu = await train(...data, { ...options, device: "cpu" });
  const gpu = await train(...data, { ...options, device: "webgpu" });
  return { cpu, gpu };
}

test("the MNIST example trains on WebGPU as it trains on the CPU tier", async () => {
  const { result, logged } = await inPage(trainBoth);
  const { cpu, gpu } = result;
  assert.deepEqual(gpu.backend, { provider: "typeshade", name: "webgpu" });
  assert.equal(gpu.losses.length, 30);
  assert.ok(gpu.losses.every(Number.isFinite));
  // The two runs make the same 30 steps of Adam on the same 30 batches. The losses agree to 1e-5
  // at every step; the worst over the 30 on SwiftShader, measured, is 3e-7.
  const worst = Math.max(
    ...gpu.losses.map(
      (loss, step) => Math.abs(loss - cpu.losses[step]) / cpu.losses[step],
    ),
  );
  assert.ok(
    worst <= 1e-5,
    `the losses of the two tiers differ by ${worst} relative at the worst step`,
  );
  const mean = (values) =>
    values.reduce((sum, value) => sum + value, 0) / values.length;
  assert.ok(
    mean(gpu.losses.slice(-5)) < 0.7 * mean(gpu.losses.slice(0, 5)),
    `${gpu.losses}`,
  );
  assert.ok(gpu.accuracy > 0.6, `the accuracy ${gpu.accuracy}`);
  assert.equal(gpu.lastReport.tier, "webgpu");
  assert.deepEqual(logged, []);
});
