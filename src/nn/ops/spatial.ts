import type { Pool2DOptions, Tensor } from "../../tensor/tensor.js";

export function maxPool(input: Tensor, options: Pool2DOptions): Tensor {
  return input.maxPool(options);
}
export function avgPool(input: Tensor, options: Pool2DOptions): Tensor {
  return input.avgPool(options);
}
export function maxPoolGrad(
  input: Tensor,
  outputGradient: Tensor,
  options: Pool2DOptions,
): Tensor {
  return input.maxPoolGrad(outputGradient, options);
}
export function avgPoolGrad(
  input: Tensor,
  outputGradient: Tensor,
  options: Pool2DOptions,
): Tensor {
  return input.avgPoolGrad(outputGradient, options);
}
