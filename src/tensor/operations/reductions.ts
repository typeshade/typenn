import {
  argMaxAxis,
  argMinAxis,
  maxAxis,
  meanAxis,
  minAxis,
  prodAxis,
  reduceGrad,
  sumAxis,
} from "../../kernels/reduce.shade.ts";
import type { Kernel } from "../node.js";
import { sizeOf } from "../shape.js";
import type { Tensor } from "../tensor.js";
import { operation } from "./common.js";

const REDUCTIONS = {
  sum: sumAxis,
  mean: meanAxis,
  prod: prodAxis,
  max: maxAxis,
  min: minAxis,
  argMax: argMaxAxis,
  argMin: argMinAxis,
} satisfies Record<string, Kernel>;

export type ReductionOperation = keyof typeof REDUCTIONS;

export function reduce(
  input: Tensor,
  op: ReductionOperation,
  axis?: number,
): Tensor {
  input.assertUsable();
  if (input.shape.length === 0)
    throw new RangeError(
      `${op} requires a tensor with at least one dimension.`,
    );
  const normalizedAxis = normalizeAxis(axis, input.shape.length);
  const outputShape =
    normalizedAxis === undefined
      ? []
      : input.shape.filter((_, index) => index !== normalizedAxis);
  // The input is [outer, axisSize, inner] and the output [outer, inner].
  const axisSize =
    normalizedAxis === undefined
      ? sizeOf(input.shape)
      : input.shape[normalizedAxis]!;
  const inner =
    normalizedAxis === undefined
      ? 1
      : sizeOf(input.shape.slice(normalizedAxis + 1));
  return operation(
    outputShape,
    op,
    REDUCTIONS[op],
    [input],
    [axisSize, inner],
    (upstream) => {
      if (op !== "sum" && op !== "mean")
        throw new Error(
          `Autograd for ${op} reductions is not implemented yet.`,
        );
      return [
        operation(
          input.shape,
          "reduceGrad",
          reduceGrad,
          [upstream],
          [axisSize, inner, op === "mean" ? 1 / axisSize : 1],
        ),
      ];
    },
  );
}

function normalizeAxis(
  axis: number | undefined,
  rank: number,
): number | undefined {
  if (axis === undefined) return undefined;
  const normalized = axis < 0 ? axis + rank : axis;
  if (!Number.isInteger(axis) || normalized < 0 || normalized >= rank)
    throw new RangeError(
      `Axis ${axis} is out of range for rank-${rank} tensor.`,
    );
  return normalized;
}
