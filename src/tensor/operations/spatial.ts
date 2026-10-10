import { sameShape } from "../shape.js";
import type { Tensor } from "../tensor.js";
import { poolingWindow, type Pool2DOptions } from "../spatial.js";
import { assertCompatible } from "./common.js";

export function pool2d(
  input: Tensor,
  op: "maxPool" | "avgPool",
  options: Pool2DOptions,
): Tensor {
  input.assertUsable();
  if (input.shape.length !== 4)
    throw new RangeError(`${op} expects an NHWC rank-4 tensor.`);
  const [batch, height, width, channels] = input.shape as readonly [
    number,
    number,
    number,
    number,
  ];
  const window = poolingWindow(height, width, options, op);
  return input.device.createOperation(
    [batch, window.outputHeight, window.outputWidth, channels],
    {
      kind: "pool2d",
      op,
      input,
      filterHeight: window.filterHeight,
      filterWidth: window.filterWidth,
      outputHeight: window.outputHeight,
      outputWidth: window.outputWidth,
      strideY: window.strideY,
      strideX: window.strideX,
      padTop: window.padTop,
      padLeft: window.padLeft,
    },
  );
}

export function pool2dGrad(
  input: Tensor,
  op: "maxPoolGrad" | "avgPoolGrad",
  dy: Tensor,
  options: Pool2DOptions,
): Tensor {
  assertCompatible(input, dy);
  if (input.shape.length !== 4)
    throw new RangeError(`${op} expects an NHWC rank-4 tensor.`);
  const [batch, height, width] = input.shape as readonly [
    number,
    number,
    number,
    number,
  ];
  const window = poolingWindow(height, width, options, op);
  const expectedShape = [
    batch,
    window.outputHeight,
    window.outputWidth,
    input.shape[3]!,
  ];
  if (!sameShape(expectedShape, dy.shape))
    throw new RangeError(
      `${op} output gradient shape must be [${expectedShape.join(", ")}].`,
    );
  return input.device.createOperation(input.shape, {
    kind: "pool2dGrad",
    op,
    input,
    dy,
    filterHeight: window.filterHeight,
    filterWidth: window.filterWidth,
    outputHeight: window.outputHeight,
    outputWidth: window.outputWidth,
    strideY: window.strideY,
    strideX: window.strideX,
    padTop: window.padTop,
    padLeft: window.padLeft,
  });
}
