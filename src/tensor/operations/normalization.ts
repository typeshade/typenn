import {
  logSoftmax,
  logSoftmaxGrad,
  softmax as softmaxKernel,
  softmaxGrad,
} from "../../kernels/softmax.shade.ts";
import { sizeOf } from "../shape.js";
import type { Tensor } from "../tensor.js";
import { operation } from "./common.js";

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
  // The input is [outer, axisSize, inner].
  const axisSize = input.shape[normalizedAxis]!;
  const inner = sizeOf(input.shape.slice(normalizedAxis + 1));
  return operation(
    input.shape,
    op,
    op === "softmax" ? softmaxKernel : logSoftmax,
    [input],
    [axisSize, inner],
    (upstream, value) => [
      operation(
        input.shape,
        "softmaxGrad",
        op === "softmax" ? softmaxGrad : logSoftmaxGrad,
        [value, upstream],
        [axisSize, inner],
      ),
    ],
  );
}
