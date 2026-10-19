import test, { after } from "node:test";
import assert from "node:assert/strict";
import { closeNn, loadNn } from "./_nn.mjs";

// What the Tensor API refuses, and says. The messages are part of what a caller reads.

const nn = await loadNn();
after(closeNn);

/** Run `body` on a CPU device, and dispose of it. */
async function onCpu(body) {
  const device = await nn.device({ device: "cpu" });
  try {
    await body(device);
  } finally {
    await device.dispose();
  }
}

test("a device is one of auto, webgpu, webgl2 and cpu, and two are open at a time", async () => {
  await assert.rejects(() => nn.device("metal"), {
    name: "TypeError",
    message: "Unknown TypeNN device preference: metal.",
  });
  await assert.rejects(() => nn.device({ device: "cpu", gpuDevice: {} }), {
    name: "TypeError",
  });
  await onCpu(async (device) => {
    assert.deepEqual(device.backend, { provider: "typeshade", name: "cpu" });
    // Two devices coexist (TypeShade change 0071): each owns its arrays and its plans.
    const other = await nn.device("cpu");
    try {
      const a = device.tensor([1, -2], { shape: [2] });
      const b = other.tensor([-3, 4], { shape: [2] });
      assert.deepEqual([...(await a.relu().data())], [1, 0]);
      assert.deepEqual([...(await b.relu().data())], [0, 4]);
      // A tensor of one device is refused by an operation of the other, before any plan.
      assert.throws(() => a.add(b), {
        message:
          "Tensor operations require tensors from the same TypeNN device.",
      });
    } finally {
      await other.dispose();
    }
  });
  await onCpu(async () => {});
});

test("without a GPU, auto falls back to the CPU tier and says why", async () => {
  const device = await nn.device("auto");
  try {
    assert.equal(device.backend.name, "cpu");
    assert.match(
      device.backend.fallbackReason,
      /webgpu: there is no WebGPU device/,
    );
  } finally {
    await device.dispose();
  }
  await assert.rejects(
    () => nn.device("webgpu"),
    /webgpu: there is no WebGPU device/,
  );
});

test("a tensor needs a shape that matches its values", async () => {
  await onCpu((device) => {
    assert.throws(() => device.tensor([1, 2, 3], { shape: [2] }), {
      name: "RangeError",
      message: "Tensor data has 3 values; shape requires 2.",
    });
    assert.throws(() => device.tensor([1], { shape: [0] }), {
      message: "Shape dimension 0 must be a positive safe integer.",
    });
  });
});

test("operands of one device, in shapes that broadcast", async () => {
  await onCpu((device) => {
    const a = device.tensor([1, 2, 3, 4, 5, 6], { shape: [2, 3] });
    assert.throws(() => a.add(device.tensor([1, 2], { shape: [2] })), {
      name: "RangeError",
      message:
        "Binary operation shapes [2, 3] and [2] are not broadcast-compatible.",
    });
    assert.throws(() => a.reshape([4]), {
      message: "reshape cannot change tensor size from 6 to 4.",
    });
    assert.throws(() => a.leakyRelu(-1), RangeError);
    assert.throws(() => a.clipByValue(2, 1), RangeError);
  });
});

test("a disposed tensor and a disposed device refuse more work", async () => {
  const device = await nn.device("cpu");
  const a = device.tensor([1, 2], { shape: [2] });
  const b = a.relu();
  await b.data();
  await b.dispose();
  await assert.rejects(() => b.data(), {
    message: "Tensor has been disposed.",
  });
  await device.dispose();
  assert.throws(() => a.relu(), {
    message: "TypeNN Device has been disposed.",
  });
});

test("an activation without a gradient rule says so when the gradient is asked for", async () => {
  await onCpu((device) => {
    const x = device.tensor([1, -2], { shape: [2] });
    const upstream = device.tensor([1, 1], { shape: [2] });
    for (const activation of [
      "gelu",
      "relu6",
      "softplus",
      "elu",
      "leakyRelu",
    ]) {
      const y = x[activation]();
      assert.throws(() => nn.gradients(y, [x], upstream), {
        message: `Autograd for activation ${activation} is not implemented yet.`,
      });
    }
    // The gradient of a gradient is not implemented either.
    const [gradient] = nn.gradients(x.relu(), [x], upstream);
    assert.throws(() => nn.gradients(gradient, [x], upstream), {
      message: "Autograd for TypeNN reluGrad nodes is not implemented yet.",
    });
  });
});

test("gradients asks for an upstream gradient of the output's shape", async () => {
  await onCpu((device) => {
    const x = device.tensor([1, 2], { shape: [2] });
    assert.throws(() => nn.gradients(x.relu(), [x]), {
      message:
        "gradients requires an upstream tensor when output is not scalar.",
    });
    assert.throws(
      () => nn.gradients(x.relu(), [x], device.tensor([1], { shape: [1] })),
      { message: "The upstream gradient must have the output shape." },
    );
  });
});

test("a tensor that is not in the graph has a zero gradient", async () => {
  await onCpu(async (device) => {
    const x = device.tensor([1, 2], { shape: [2] });
    const other = device.tensor([5, 6], { shape: [2] });
    const [gradient] = nn.gradients(
      x.relu(),
      [other],
      device.tensor([1, 1], { shape: [2] }),
    );
    assert.deepEqual(Array.from(await gradient.data()), [0, 0]);
  });
});

test("backward needs a parameter in the graph, and sums into .grad", async () => {
  await onCpu(async (device) => {
    const x = device.tensor([1, 2], { shape: [2] });
    await assert.rejects(() => x.relu().backward(x), {
      message: "backward() found no trainable parameters in the graph.",
    });
    const parameter = device.createParameter(new Float32Array([3, -4]), {
      shape: [2],
    });
    const upstream = device.tensor([1, 1], { shape: [2] });
    const y = parameter.mul(parameter);
    await y.backward(upstream);
    await y.backward(upstream);
    // 2 * p, added twice.
    assert.deepEqual(Array.from(await parameter.grad.data()), [12, -16]);
    await parameter.zeroGrad();
    assert.equal(parameter.grad, undefined);
  });
});
