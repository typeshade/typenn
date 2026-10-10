import { sizeOf } from "../shape.js";
import type { ReductionOperation } from "../nodes/reduce.js";
import type { Tensor } from "../tensor.js";

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
  const axisSize =
    normalizedAxis === undefined
      ? sizeOf(input.shape)
      : input.shape[normalizedAxis]!;
  return input.device.createOperation(outputShape, {
    kind: "reduce",
    op,
    input,
    outer:
      normalizedAxis === undefined
        ? 1
        : sizeOf(input.shape.slice(0, normalizedAxis)),
    axisSize,
    inner:
      normalizedAxis === undefined
        ? 1
        : sizeOf(input.shape.slice(normalizedAxis + 1)),
  });
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
