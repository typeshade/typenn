import type { Conv2DOptions, Tensor } from "../../tensor/tensor.js";

export function matMul(
  a: Tensor,
  b: Tensor,
  options: {
    readonly transposeA?: boolean;
    readonly transposeB?: boolean;
  } = {},
): Tensor {
  return a.matmul(b, options);
}
export function conv2d(
  input: Tensor,
  filter: Tensor,
  options: Conv2DOptions = {},
): Tensor {
  return input.conv2d(filter, options);
}
export function conv2dGradients(
  input: Tensor,
  filter: Tensor,
  outputGradient: Tensor,
  options: Conv2DOptions = {},
): readonly [inputGradient: Tensor, filterGradient: Tensor] {
  return input.conv2dGradients(filter, outputGradient, options);
}
