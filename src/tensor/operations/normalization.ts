import {
  logSoftmax,
  logSoftmaxFromStatistics,
  logSoftmaxGrad,
  logSoftmaxGradFromReduction,
  logSoftmaxGradReduction,
  softmax as softmaxKernel,
  softmaxFromStatistics,
  softmaxGrad,
  softmaxGradFromReduction,
  softmaxGradReduction,
  softmaxStatistics,
} from "../../kernels/softmax.shade.ts";
import { sizeOf } from "../shape.js";
import type { Tensor } from "../tensor.js";
import { operation } from "./common.js";

// The extra dispatch pays off only once the axis is long enough. Benchmark
// the crossover on hardware with scripts/bench-softmax.mjs.
const TWO_PASS_MIN_AXIS = 64;

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
  const twoPass = axisSize >= TWO_PASS_MIN_AXIS;
  // Two entries per slice: the peak, then the sum of exp(x - peak).
  const statistics = twoPass
    ? operation(
        [(2 * sizeOf(input.shape)) / axisSize],
        "softmaxStatistics",
        softmaxStatistics,
        [input],
        [axisSize, inner],
      )
    : undefined;

  return operation(
    input.shape,
    op,
    twoPass
      ? op === "softmax"
        ? softmaxFromStatistics
        : logSoftmaxFromStatistics
      : op === "softmax"
        ? softmaxKernel
        : logSoftmax,
    statistics ? [input, statistics] : [input],
    [axisSize, inner],
    (upstream, value) => {
      if (!twoPass)
        return [
          operation(
            input.shape,
            "softmaxGrad",
            op === "softmax" ? softmaxGrad : logSoftmaxGrad,
            [value, upstream],
            [axisSize, inner],
          ),
        ];

      const softmaxPath = op === "softmax";
      const reduced = operation(
        [sizeOf(input.shape) / axisSize],
        softmaxPath ? "softmaxGradReduction" : "logSoftmaxGradReduction",
        softmaxPath ? softmaxGradReduction : logSoftmaxGradReduction,
        softmaxPath ? [value, upstream] : [upstream],
        [axisSize, inner],
      );
      const gradient = operation(
        input.shape,
        "softmaxGrad",
        softmaxPath ? softmaxGradFromReduction : logSoftmaxGradFromReduction,
        [value, upstream, reduced],
        [axisSize, inner],
      );
      // Statistics are a forward-only cache, not another differentiable input.
      return [gradient, undefined];
    },
  );
}
