import { sizeOf } from "../shape.js";
import type { Tensor } from "../tensor.js";

export function softmax(
  input: Tensor,
  op: "softmax" | "logSoftmax",
  axis = -1,
): Tensor {
  input.assertUsable();
  if (input.shape.length === 0)
    throw new RangeError(
      `${op} requires a tensor with at least one dimension.`,
    );
  const normalizedAxis = axis < 0 ? axis + input.shape.length : axis;
  if (
    !Number.isInteger(axis) ||
    normalizedAxis < 0 ||
    normalizedAxis >= input.shape.length
  )
    throw new RangeError(
      `Axis ${axis} is out of range for rank-${input.shape.length} tensor.`,
    );

  return input.device.createOperation(input.shape, {
    kind: "softmax",
    op,
    input,
    axis: normalizedAxis,
    outer: sizeOf(input.shape.slice(0, normalizedAxis)),
    axisSize: input.shape[normalizedAxis]!,
    inner: sizeOf(input.shape.slice(normalizedAxis + 1)),
  });
}
