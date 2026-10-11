import type { Tensor } from "../../tensor/tensor.js";

export function sum(input: Tensor, axis?: number): Tensor {
  return input.sum(axis);
}
export function mean(input: Tensor, axis?: number): Tensor {
  return input.mean(axis);
}
export function prod(input: Tensor, axis?: number): Tensor {
  return input.prod(axis);
}
export function max(input: Tensor, axis?: number): Tensor {
  return input.max(axis);
}
export function min(input: Tensor, axis?: number): Tensor {
  return input.min(axis);
}
export function argMax(input: Tensor, axis = 0): Tensor {
  return input.argMax(axis);
}
export function argMin(input: Tensor, axis = 0): Tensor {
  return input.argMin(axis);
}
