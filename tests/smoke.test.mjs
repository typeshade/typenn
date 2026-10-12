import test from "node:test";
import assert from "node:assert/strict";
import * as nn from "../dist/index.js";

test("add, mul and relu chain on the CPU tier", async () => {
  const device = await nn.device({ device: "cpu" });
  try {
    const a = device.tensor([1, 2, -3, 4], { shape: [4] });
    const b = device.tensor([5, -3, 2, -8], { shape: [4] });
    const c = device.tensor([2, 1, -2, 0.5], { shape: [4] });
    const y = a.add(b).mul(c).relu();
    assert.deepEqual(Array.from(await y.data()), [12, 0, 2, 0]);
  } finally {
    await device.dispose();
  }
});
