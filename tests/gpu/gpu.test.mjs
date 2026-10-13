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
    const grads = parameters.map((parameter) => parameter.grad);
    const before = device.reports.length;
    // The gradients of all parameters are read in one plan, as the backward pass of the model.
    out.backwardSeen = await counted(async () => {
      await device.materializeAll(grads);
    });
    out.backward = device.reports.slice(before).map(stated);
    // The step computes the same gradients inside its update plan and reads nothing back.
    const beforeStep = device.reports.length;
    out.stepSeen = await counted(async () => {
      await new nn.SGD(parameters, { lr: 0.1 }).step();
    });
    out.step = device.reports.slice(beforeStep).map(stated);
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
  // The step is one plan with no read-back: its submission is the only one.
  assert.equal(result.step.length, 1);
  assert.equal(result.stepSeen.submissions, 1);
  assert.equal(result.stepSeen.dispatches, result.step[0].dispatches);
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

/** The loss of 20 steps of the two models on WebGPU (SwiftShader), as `origin/main` 3a6ba27 computed
 * them: the read before `backward()` and the Adam step of that commit. */
const RECORDED_GPU_MLP = [
  2.3285720348358154, 2.2661049365997314, 2.215897798538208, 2.1727631092071533,
  2.097892999649048, 2.0505495071411133, 1.9738157987594604, 1.9186973571777344,
  1.8722509145736694, 1.8110764026641846, 1.737915277481079, 1.7072733640670776,
  1.6429568529129028, 1.5745967626571655, 1.4944705963134766,
  1.4696835279464722, 1.407051920890808, 1.3253984451293945, 1.2298675775527954,
  1.1899969577789307,
];
const RECORDED_GPU_CNN = [
  2.3246567249298096, 2.264599561691284, 2.2140798568725586, 2.1559009552001953,
  2.097506284713745, 2.041372537612915, 1.9855619668960571, 1.928479790687561,
  1.8704768419265747, 1.8126192092895508, 1.755347490310669, 1.6983989477157593,
  1.6415375471115112, 1.5848430395126343, 1.5285427570343018,
  1.4727919101715088, 1.4176115989685059, 1.3630743026733398, 1.309317708015442,
  1.2564539909362793,
];

/** Runs in the page: `steps` steps of one model with the loop `alsoRead` names, on WebGPU. */
async function trainOnWebGPU({ model, steps, alsoRead }) {
  const nn = await import("/dist/index.js");
  const device = await nn.device("webgpu");
  try {
    let state = 7;
    const next = () =>
      (state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 2 ** 32;
    const images = Float32Array.from({ length: 64 * 784 }, next);
    const labels = Array.from({ length: 64 }, (_, row) => row % 10);
    const net =
      model === "mlp"
        ? new nn.Sequential(
            new nn.Flatten(),
            new nn.Linear(784, 128, {
              seed: 101,
              weightScale: 1 / Math.sqrt(784),
            }),
            new nn.ReLU(),
            new nn.Dropout(0.2),
            new nn.Linear(128, 10, {
              seed: 202,
              weightScale: 1 / Math.sqrt(128),
            }),
          ).to(device)
        : new nn.Sequential(
            new nn.Conv2d(1, 8, { kernelSize: 3, seed: 5 }),
            new nn.ReLU(),
            new nn.MaxPool2d({ filterSize: 2 }),
            new nn.Flatten(),
            new nn.Linear(13 * 13 * 8, 10, { seed: 6 }),
          ).to(device);
    const shape = model === "mlp" ? [64, 1, 28, 28] : [64, 28, 28, 1];
    const parameters = net.parameters();
    const optimizer = nn.optim.Adam(parameters, { lr: 0.001 });
    const criterion = new nn.CrossEntropyLoss();
    const losses = [];
    for (let step = 0; step < steps; step++) {
      await optimizer.zeroGrad();
      const loss = criterion.forward(
        net.forward(device.tensor(images, { shape })),
        labels,
      );
      if (alsoRead) {
        await loss.backward();
        const [values] = await optimizer.step({ alsoRead: [loss] });
        losses.push(values[0]);
      } else {
        losses.push((await loss.data())[0]);
        await loss.backward();
        await optimizer.step();
      }
      await loss.disposeGraph(parameters);
    }
    return losses;
  } finally {
    await device.dispose();
  }
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

test("a training step on WebGPU, with the pool and with alsoRead, gives the losses of origin/main, bit for bit", async () => {
  for (const [model, recorded] of [
    ["mlp", RECORDED_GPU_MLP],
    ["cnn", RECORDED_GPU_CNN],
  ]) {
    for (const alsoRead of [false, true]) {
      const { result, logged } = await inPage(trainOnWebGPU, {
        model,
        steps: 20,
        alsoRead,
      });
      assert.deepEqual(result, recorded, `${model}, alsoRead ${alsoRead}`);
      assert.deepEqual(logged, []);
    }
  }
});
