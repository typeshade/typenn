import test, { after } from "node:test";
import assert from "node:assert/strict";
import { fakeGpu } from "./_fake-gpu.mjs";
import { closeNn, loadNn } from "./_nn.mjs";

// Arrays and plans over time: what a tensor holds after a read, a failure and a disposal.

const nn = await loadNn();
after(closeNn);

/** Run `body` on a TypeNN device over a fake WebGPU device, and dispose of both. */
async function onFakeGpu(body, options = {}) {
  const { poolBytes, ...gpuOptions } = options;
  const gpu = fakeGpu(gpuOptions);
  const device = await nn.device({
    device: "webgpu",
    gpuDevice: gpu.device,
    ...(poolBytes === undefined ? {} : { poolBytes }),
  });
  try {
    await body(device, gpu);
  } finally {
    await device.dispose();
  }
}

/** The exact count of arrays below is the count without the pool, which frees each array at once. */
const onFakeGpuUnpooled = (body, options = {}) =>
  onFakeGpu(body, { ...options, poolBytes: 0 });

const vector = (device, values) =>
  device.tensor(values, { shape: [values.length] });

test("reads that overlap, started together, are computed once and give what one at a time gives", async () => {
  const device = await nn.device("cpu");
  try {
    const a = vector(device, [1, -2, 3, -4]);
    const b = vector(device, [5, 6, -7, 8]);
    const shared = a.add(b);
    const left = shared.mul(a).relu();
    const right = shared.sub(b).neg();
    const [x, y, z] = await Promise.all([
      left.data(),
      right.data(),
      shared.data(),
    ]);
    assert.deepEqual(Array.from(x), [6, 0, 0, 0]);
    assert.deepEqual(Array.from(y), [-1, 2, -3, 4]);
    assert.deepEqual(Array.from(z), [6, 4, -4, 4]);
    // `shared` was computed by the first plan, and the later reads took it as a leaf.
    const calls = device.reports.flatMap((report) =>
      report.groups.flatMap((group) => group.calls),
    );
    assert.equal(calls.filter((name) => name === "add").length, 1);
    // One `data()` for each tensor that had no array, and none for the last (`shared`).
    assert.equal(device.reports.length, 2);
  } finally {
    await device.dispose();
  }
});

test("a plan that fails leaves no array behind, and the same read runs again", async () => {
  await onFakeGpu(
    async (device) => {
      const y = vector(device, [1, 2, 3]).relu().neg();
      await assert.rejects(() => y.data(), /planted validation error/);
      assert.equal(device.reports.length, 0);
      // The tensors have no array: the next read computes them again, and the device takes it.
      assert.deepEqual(Array.from(await y.data()), [0, 0, 0]);
      assert.equal(device.reports.length, 1);
      assert.deepEqual(device.lastReport.groups, [
        { calls: ["relu", "negMap"], fused: true },
      ]);
    },
    { validation: "planted validation error", validationAt: 1 },
  );
});

test("a training step leaves no array behind", async () => {
  await onFakeGpuUnpooled(async (device, gpu) => {
    const model = new nn.Sequential(
      new nn.Flatten(),
      new nn.Linear(16, 8, { seed: 1 }),
      new nn.ReLU(),
      new nn.Linear(8, 4, { seed: 2 }),
    ).to(device);
    const parameters = model.parameters();
    const optimizer = nn.optim.Adam(parameters, { lr: 0.01 });
    const criterion = new nn.CrossEntropyLoss();
    const live = () => gpu.made.filter((buffer) => !buffer.destroyed).length;
    const counts = [];
    for (let step = 0; step < 5; step++) {
      await optimizer.zeroGrad();
      const logits = model.forward(
        device.tensor(new Float32Array(5 * 16), { shape: [5, 4, 4] }),
      );
      const loss = criterion.forward(logits, [0, 1, 2, 3, 0]);
      await loss.data();
      await loss.backward();
      await optimizer.step();
      await loss.disposeGraph(parameters);
      counts.push(live());
    }
    // The same buffers after every step: the parameters, their gradients, and what the compiler
    // keeps for a read.
    assert.deepEqual(
      new Set(counts).size,
      1,
      `live buffers after each step: ${counts}`,
    );
    await optimizer.zeroGrad();
    // Four parameters, and the two moments of each on the device, and nothing else.
    assert.equal(live(), 4 + 2 * 4);
  });
  // A disposed device has freed everything it made.
  const gpu = fakeGpu();
  const device = await nn.device({ device: "webgpu", gpuDevice: gpu.device });
  await vector(device, [1, 2]).relu().data();
  await device.dispose();
  assert.equal(gpu.made.filter((buffer) => !buffer.destroyed).length, 0);
});

test("a parameter written by the optimizer is what the next plan reads", async () => {
  const device = await nn.device("cpu");
  try {
    const weight = device.createParameter(new Float32Array([1, 2]), {
      shape: [2],
    });
    const input = vector(device, [10, 10]);
    const product = weight.mul(input);
    assert.deepEqual(Array.from(await product.data()), [10, 20]);
    await weight.assign([3, 4]);
    // `product` holds the value it computed, and a graph built after the write reads the new one.
    assert.deepEqual(Array.from(await product.data()), [10, 20]);
    assert.deepEqual(Array.from(await weight.mul(input).data()), [30, 40]);
    assert.deepEqual(Array.from(await weight.data()), [3, 4]);
  } finally {
    await device.dispose();
  }
});
