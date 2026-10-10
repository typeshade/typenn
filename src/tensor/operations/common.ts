import type { Tensor } from "../tensor.js";

export function assertCompatible(left: Tensor, right: Tensor): void {
  left.assertUsable();
  right.assertUsable();
  if (left.device !== right.device)
    throw new TypeError(
      "Tensor operations require tensors from the same TypeNN device.",
    );
}
