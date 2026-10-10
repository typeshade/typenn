import { sameShape, sizeOf } from "../tensor/shape.js";
import type { Tensor } from "../tensor/tensor.js";

export function constantLike(tensor: Tensor, value: number): Tensor {
  return tensor.device.tensor(
    new Float32Array(sizeOf(tensor.shape)).fill(value),
    { shape: tensor.shape },
  );
}

export function scalarLike(tensor: Tensor, value: number): Tensor {
  return tensor.device.tensor([value], { shape: [] });
}

export function addGradient(
  gradients: Map<Tensor, Tensor>,
  tensor: Tensor,
  contribution: Tensor,
): void {
  const previous = gradients.get(tensor);
  if (!previous) {
    gradients.set(tensor, contribution);
    return;
  }
  if (!sameShape(previous.shape, contribution.shape))
    throw new RangeError(
      "Gradient contributions for the same tensor must have identical shapes.",
    );
  gradients.set(tensor, previous.add(contribution));
}
