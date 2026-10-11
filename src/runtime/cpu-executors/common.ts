import { sizeOf } from "../../tensor/shape.js";
import type { Tensor } from "../../tensor/tensor.js";

export function createOutput(tensor: Tensor): Float32Array {
  return new Float32Array(sizeOf(tensor.shape));
}

export function workgroupsFor(length: number): number {
  return Math.ceil(length / 64);
}

export type CpuKernel<Bindings> = (
  bindings: Bindings,
  workgroups: number | readonly [number, number, number],
) => void | Promise<void>;

export async function dispatchCpuKernel<Bindings>(
  kernel: CpuKernel<Bindings>,
  bindings: Bindings,
  logicalSize: number,
  onDispatch?: () => void,
): Promise<void> {
  await kernel(bindings, workgroupsFor(logicalSize));
  onDispatch?.();
}
