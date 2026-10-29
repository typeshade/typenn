import test from "node:test";
import assert from "node:assert/strict";
import { inPage } from "./_browser.mjs";

async function sampleOnWebGPU() {
  const nn = await import("/dist/index.js");
  const device = await nn.device({ device: "webgpu" });
  try {
    const image = device.tensor([1, 2, 3, 4, 5, 6], {
      shape: [1, 2, 3, 1],
    });
    const coordinates = device.tensor(
      [0, 0, 1, 1, -1, -1, 0.5, -0.5, 1.25, 0, -1.5, 0],
      { shape: [1, 1, 6, 2] },
    );
    const alignedGrid = image.gridSampleNHWC(coordinates, {
      alignCorners: true,
    });
    const alignCornersTrue = Array.from(await alignedGrid.data());
    const gridReport = device.lastReport;
    const defaultGrid = await image.gridSampleNHWC(coordinates).data();
    const resizedAligned = await image
      .resizeBilinear(3, 5, {
        alignCorners: true,
      })
      .data();
    const resizedDefault = await image.resizeBilinear(3, 5).data();
    const nonFinite = device.tensor(
      [Number.NaN, 0, Number.POSITIVE_INFINITY, 0, Number.NEGATIVE_INFINITY, 0],
      { shape: [1, 1, 3, 2] },
    );
    const nonFiniteResult = await image.gridSampleNHWC(nonFinite).data();
    const multiChannel = device.tensor([1, 10, 2, 20, 3, 30, 4, 40], {
      shape: [1, 2, 2, 2],
    });
    const multiChannelGrid = device.tensor([0, 0, -1, -1], {
      shape: [1, 1, 2, 2],
    });
    const multiChannelResult = await multiChannel
      .gridSampleNHWC(multiChannelGrid, { alignCorners: true })
      .data();
    const batched = device.tensor([1, 10, 3, 30, 101, 1010, 103, 1030], {
      shape: [2, 1, 2, 2],
    });
    const batchGrid = device.tensor([0, 0, 0, 0], { shape: [2, 1, 1, 2] });
    const batchGridResult = await batched
      .gridSampleNHWC(batchGrid, { alignCorners: true })
      .data();
    const batchResizeResult = await batched.resizeBilinear(1, 1).data();
    return {
      backend: device.backend.name,
      alignCornersTrue,
      alignCornersFalse: Array.from(defaultGrid),
      resizeAlignCornersTrue: Array.from(resizedAligned),
      resizeAlignCornersFalse: Array.from(resizedDefault),
      nonFinite: Array.from(nonFiniteResult, Number.isNaN),
      multiChannel: Array.from(multiChannelResult),
      batchGrid: Array.from(batchGridResult),
      batchResize: Array.from(batchResizeResult),
      gridReport,
    };
  } finally {
    await device.dispose();
  }
}

function close(actual, expected, epsilon = 2e-5) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < expected.length; i++) {
    assert.ok(
      Math.abs(actual[i] - expected[i]) <= epsilon,
      `index ${i}: got ${actual[i]}, expected ${expected[i]}`,
    );
  }
}

test("sampling kernels match PyTorch finite-coordinate goldens and TypeNN non-finite policy on WebGPU", async () => {
  const hardware = process.env.TYPECV_HARDWARE_GPU === "1";
  const { result, logged } = await inPage(sampleOnWebGPU, null, { hardware });
  assert.equal(result.backend, "webgpu");
  close(result.alignCornersTrue, [3.5, 6, 1, 3.25, 3.375, 1.25]);
  close(result.alignCornersFalse, [3.5, 1.5, 0.25, 2.75, 0.5625, 0]);
  close(
    result.resizeAlignCornersTrue,
    [1, 1.5, 2, 2.5, 3, 2.5, 3, 3.5, 4, 4.5, 4, 4.5, 5, 5.5, 6],
  );
  close(
    result.resizeAlignCornersFalse,
    [1, 1.4, 2, 2.6, 3, 2.5, 2.9, 3.5, 4.1, 4.5, 4, 4.4, 5, 5.6, 6],
  );
  assert.deepEqual(result.nonFinite, [true, true, true]);
  close(result.multiChannel, [2.5, 25, 1, 10]);
  close(result.batchGrid, [2, 20, 102, 1020]);
  close(result.batchResize, [2, 20, 102, 1020]);
  assert.equal(result.gridReport.tier, "webgpu");
  assert.deepEqual(
    result.gridReport.groups.map((group) => group.calls),
    [["gridSampleNHWC"]],
  );
  assert.deepEqual(logged, []);
});
