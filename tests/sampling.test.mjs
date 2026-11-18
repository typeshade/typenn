import test, { after } from "node:test";
import assert from "node:assert/strict";
import { closeNn, loadNn } from "./_nn.mjs";

const nn = await loadNn();
after(closeNn);

async function onCpu(body) {
  const device = await nn.device({ device: "cpu" });
  try {
    await body(device);
  } finally {
    await device.dispose();
  }
}

function close(actual, expected, epsilon = 2e-6) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < expected.length; i++) {
    assert.ok(
      Math.abs(actual[i] - expected[i]) <= epsilon,
      `index ${i}: got ${actual[i]}, expected ${expected[i]}`,
    );
  }
}

test("gridSampleNHWC matches PyTorch bilinear zero-padding coordinate modes", async () => {
  await onCpu(async (device) => {
    const image = device.tensor([1, 2, 3, 4, 5, 6], { shape: [1, 2, 3, 1] });
    const coordinates = device.tensor(
      [0, 0, 1, 1, -1, -1, 0.5, -0.5, 1.25, 0, -1.5, 0],
      { shape: [1, 1, 6, 2] },
    );
    // CPU oracle: torch.nn.functional.grid_sample, mode=bilinear,
    // padding_mode=zeros, with both align_corners settings.
    close(
      await nn
        .gridSampleNHWC(image, coordinates, { alignCorners: true })
        .data(),
      [3.5, 6, 1, 3.25, 3.375, 1.25],
    );
    close(
      await image.gridSampleNHWC(coordinates).data(),
      [3.5, 1.5, 0.25, 2.75, 0.5625, 0],
    );
  });
});

test("gridSampleNHWC preserves NHWC channel order", async () => {
  await onCpu(async (device) => {
    const image = device.tensor([1, 10, 2, 20, 3, 30, 4, 40], {
      shape: [1, 2, 2, 2],
    });
    const coordinates = device.tensor([0, 0, -1, -1], {
      shape: [1, 1, 2, 2],
    });
    close(
      await image.gridSampleNHWC(coordinates, { alignCorners: true }).data(),
      [2.5, 25, 1, 10],
    );
  });
});

test("sampling preserves batch offsets across channels", async () => {
  await onCpu(async (device) => {
    const image = device.tensor([1, 10, 3, 30, 101, 1010, 103, 1030], {
      shape: [2, 1, 2, 2],
    });
    const coordinates = device.tensor([0, 0, 0, 0], {
      shape: [2, 1, 1, 2],
    });
    close(
      await image.gridSampleNHWC(coordinates, { alignCorners: true }).data(),
      [2, 20, 102, 1020],
    );
    close(await image.resizeBilinear(1, 1).data(), [2, 20, 102, 1020]);
  });
});

test("gridSampleNHWC handles singleton dimensions and propagates non-finite grids", async () => {
  await onCpu(async (device) => {
    const image = device.tensor([7, 11], { shape: [1, 1, 2, 1] });
    const coordinates = device.tensor([-1, 0, 0, 0, 1, 0, 2, 0], {
      shape: [1, 1, 4, 2],
    });
    close(
      await image.gridSampleNHWC(coordinates, { alignCorners: true }).data(),
      [7, 9, 11, 5.5],
    );
    close(await image.gridSampleNHWC(coordinates).data(), [3.5, 9, 5.5, 0]);

    const nonFinite = device.tensor(
      [Number.NaN, 0, Number.POSITIVE_INFINITY, 0, Number.NEGATIVE_INFINITY, 0],
      { shape: [1, 1, 3, 2] },
    );
    const result = await image.gridSampleNHWC(nonFinite).data();
    assert.ok([...result].every(Number.isNaN));
  });
});

test("resizeBilinear matches PyTorch interpolate including edge clamping", async () => {
  await onCpu(async (device) => {
    const image = device.tensor([1, 2, 3, 4, 5, 6], { shape: [1, 2, 3, 1] });
    close(
      await nn.resizeBilinear(image, 3, 5, { alignCorners: true }).data(),
      [1, 1.5, 2, 2.5, 3, 2.5, 3, 3.5, 4, 4.5, 4, 4.5, 5, 5.5, 6],
    );
    close(
      await image.resizeBilinear(3, 5).data(),
      [1, 1.4, 2, 2.6, 3, 2.5, 2.9, 3.5, 4.1, 4.5, 4, 4.4, 5, 5.6, 6],
    );
    close(await image.resizeBilinear(1, 1).data(), [3.5]);
  });
});

test("sampling validates shapes and reports its inference-only gradient policy", async () => {
  await onCpu(async (device) => {
    const image = device.tensor([1, 2, 3, 4], { shape: [1, 2, 2, 1] });
    const wrongGrid = device.tensor([0, 0], { shape: [1, 1, 1, 2] });
    const badGrid = device.tensor([0, 0, 0], { shape: [1, 1, 1, 3] });
    assert.throws(() => nn.gridSampleNHWC(image, badGrid), RangeError);
    assert.throws(() => image.resizeBilinear(0, 2), RangeError);
    assert.throws(
      () => image.gridSampleNHWC(wrongGrid, { alignCorners: 1 }),
      TypeError,
    );

    const x = device.tensor([1, 2, 3, 4], { shape: [1, 2, 2, 1] });
    const grid = device.tensor([0, 0], { shape: [1, 1, 1, 2] });
    const sampled = x.gridSampleNHWC(grid);
    assert.throws(() => nn.gradients(sampled, [x]), {
      message: "Autograd for gridSampleNHWC is not implemented yet.",
    });
    assert.throws(() => nn.gradients(x.resizeBilinear(1, 1), [x]), {
      message: "Autograd for resizeBilinear is not implemented yet.",
    });
  });
});
