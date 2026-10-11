import { checkedShape, sizeOf } from "../shape.js";
import type { Tensor } from "../tensor.js";

export function reshape(input: Tensor, shape: readonly number[]): Tensor {
  input.assertUsable();
  const outputShape = checkedShape(shape);
  const inputSize = sizeOf(input.shape);
  const outputSize = sizeOf(outputShape);
  if (outputSize !== inputSize)
    throw new RangeError(
      `reshape cannot change tensor size from ${inputSize} to ${outputSize}.`,
    );
  return input.device.createOperation(outputShape, { kind: "reshape", input });
}
