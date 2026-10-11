import test from "node:test";
import assert from "node:assert/strict";
import * as nn from "../dist/index.js";

const close = (actual, expected, tolerance = 1e-5) => {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++) {
    const absolute = Math.abs(actual[i] - expected[i]);
    const relative = absolute / Math.max(Math.abs(expected[i]), 1e-8);
    assert.ok(
      absolute <= tolerance || relative <= tolerance,
      `index ${i}: got ${actual[i]}, expected ${expected[i]}, abs=${absolute}, rel=${relative}`,
    );
  }
};

test("CPU broadcast, matmul, activation and reduction match reference values", async () => {
  const device = await nn.device({ device: "cpu" });
  try {
    const a = device.tensor([1, 2, 3, 4], { shape: [2, 2] });
    const b = device.tensor([10, 20], { shape: [2] });
    close(await a.add(b).data(), [11, 22, 13, 24]);
    close(
      await a.matmul(device.tensor([5, 6, 7, 8], { shape: [2, 2] })).data(),
      [19, 22, 43, 50],
    );
    close(
      await a
        .sub(device.tensor([2, 3, 2, 5], { shape: [2, 2] }))
        .relu()
        .sum()
        .data(),
      [1],
    );
  } finally {
    await device.dispose();
  }
});

test("Tensor graph gradients pass a finite-difference check", async () => {
  const device = await nn.device({ device: "cpu" });
  try {
    const values = new Float32Array([0.4, -0.7]);
    const parameter = device.createParameter(values, { shape: [2] });
    const objective = parameter.square().sum();
    await objective.backward();
    const analytic = await parameter.grad.data();
    const h = 1e-3;
    const numeric = [];
    for (let i = 0; i < values.length; i++) {
      const plus = values.slice();
      const minus = values.slice();
      plus[i] += h;
      minus[i] -= h;
      const fPlus = plus.reduce((sum, x) => sum + x * x, 0);
      const fMinus = minus.reduce((sum, x) => sum + x * x, 0);
      numeric.push((fPlus - fMinus) / (2 * h));
    }
    close(analytic, numeric, 2e-4);
    await objective.disposeGraph([parameter, parameter.grad]);
  } finally {
    await device.dispose();
  }
});

test("CPU Adam uses bias correction and keeps the existing step API", async () => {
  const device = await nn.device({ device: "cpu" });
  try {
    const parameter = device.createParameter(new Float32Array([1, -2]), {
      shape: [2],
    });
    const optimizer = new nn.Adam([parameter], { lr: 0.1 });
    const loss = parameter.square().sum();
    await loss.backward();
    await optimizer.step();
    close(await parameter.data(), [0.9, -1.9], 1e-5);
    await optimizer.zeroGrad();
    await loss.disposeGraph([parameter]);
  } finally {
    await device.dispose();
  }
});

test("backward accumulates gradients and CPU SGD updates parameters", async () => {
  const device = await nn.device({ device: "cpu" });
  try {
    const parameter = device.createParameter(new Float32Array([1, -2]), {
      shape: [2],
    });
    const objective = parameter.square().sum();
    await objective.backward();
    await objective.backward();
    close(await parameter.grad.data(), [4, -8]);
    const optimizer = new nn.SGD([parameter], { lr: 0.05 });
    await optimizer.step();
    close(await parameter.data(), [0.8, -1.6]);
    await objective.disposeGraph([parameter, parameter.grad]);
  } finally {
    await device.dispose();
  }
});
