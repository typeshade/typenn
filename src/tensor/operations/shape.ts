import { copy } from "../../kernels/map.shade.ts";
import { checkedShape, sizeOf } from "../shape.js";
import type { Tensor } from "../tensor.js";
import { operation } from "./common.js";

/** A reshape is a copy: the new tensor has an array of its own. */
export function reshape(input: Tensor, shape: readonly number[]): Tensor {
  input.assertUsable();
  const outputShape = checkedShape(shape);
  const inputSize = sizeOf(input.shape);
  const outputSize = sizeOf(outputShape);
  if (outputSize !== inputSize)
    throw new RangeError(
      `reshape cannot change tensor size from ${inputSize} to ${outputSize}.`,
    );
  return operation(outputShape, "reshape", copy, [input], [], (upstream) => [
    upstream.reshape(input.shape),
  ]);
}
