import type { Shape } from "../../tensor/shape.js";
import type { Tensor } from "../../tensor/tensor.js";

export function reshape(input: Tensor, shape: Shape): Tensor {
  return input.reshape(shape);
}
