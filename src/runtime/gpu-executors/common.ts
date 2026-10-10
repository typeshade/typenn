import { sizeOf } from "../../tensor/shape.js";
import type { GpuExecutionContext } from "./types.js";

export function elementwiseWorkgroups(
  context: Pick<GpuExecutionContext, "tensor">,
): number {
  return Math.ceil(sizeOf(context.tensor.shape) / 64);
}

export function outputLength(
  context: Pick<GpuExecutionContext, "tensor">,
): number {
  return sizeOf(context.tensor.shape);
}
