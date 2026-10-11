import type { Tensor } from "../../tensor/tensor.js";

export function softmax(input: Tensor, axis = -1): Tensor {
  return input.softmax(axis);
}
export function logSoftmax(input: Tensor, axis = -1): Tensor {
  return input.logSoftmax(axis);
}
