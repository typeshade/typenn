import {
  logSoftmax,
  logSoftmaxFromLogSumExp,
  logSoftmaxGrad,
  softmax as softmaxKernel,
  softmaxFromLogSumExp,
  softmaxGrad,
  softmaxGradFromReduction,
  softmaxGradReduction,
  softmaxLogSumExp,
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
  const statistics = twoPass
    ? operation(
        [sizeOf(input.shape) / axisSize],
        "softmaxLogSumExp",
        softmaxLogSumExp,
        [input],
        [axisSize, inner],
      )
    : undefined;

  return operation(
    input.shape,
    op,
    twoPass
      ? op === "softmax"
        ? softmaxFromLogSumExp
        : logSoftmaxFromLogSumExp
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

      const weighted = op === "softmax" ? 1 : 0;
      const reduced = operation(
        [sizeOf(input.shape) / axisSize],
        "softmaxGradReduction",
        softmaxGradReduction,
        [value, upstream],
        [axisSize, inner, weighted],
      );
      const gradient = operation(
        input.shape,
        "softmaxGrad",
        softmaxGradFromReduction,
        [value, upstream, reduced],
        [axisSize, inner, weighted],
      );
      // Statistics are a forward-only cache, not another differentiable input.
      return [gradient, undefined];
    },
  );
}
