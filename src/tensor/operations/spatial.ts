import {
  avgPool,
  avgPoolGrad,
  maxPool,
  maxPoolGrad,
} from "../../kernels/pool.shade.ts";
import { sameShape } from "../shape.js";
import type { Tensor } from "../tensor.js";
import { poolingWindow, type Pool2DOptions } from "../spatial.js";
import { assertCompatible, operation } from "./common.js";

/** What a pooling kernel function takes after its arrays: the input and the window. */
function windowValues(
  input: Tensor,
  window: ReturnType<typeof poolingWindow>,
): number[] {
  const [, height, width, channels] = input.shape as readonly number[] &
    [number, number, number, number];
  return [
    height!,
    width!,
    channels!,
    window.filterHeight,
    window.filterWidth,
    window.outputHeight,
    window.outputWidth,
    window.strideY,
    window.strideX,
    window.padTop,
    window.padLeft,
  ];
}

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
  return operation(
    [batch, window.outputHeight, window.outputWidth, channels],
    op,
    op === "maxPool" ? maxPool : avgPool,
    [input],
    windowValues(input, window),
    (upstream) => [
      pool2dGrad(
        input,
        op === "maxPool" ? "maxPoolGrad" : "avgPoolGrad",
        upstream,
        options,
      ),
    ],
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
  const values = windowValues(input, window);
  // The gradient of an average does not read the input, only the gradient of the output.
  return op === "maxPoolGrad"
    ? operation(input.shape, "pool2dGrad", maxPoolGrad, [input, dy], values)
    : operation(input.shape, "pool2dGrad", avgPoolGrad, [dy], values);
}
