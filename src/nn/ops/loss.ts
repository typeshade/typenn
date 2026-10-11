import type { Tensor } from "../../tensor/tensor.js";

export function softmaxCrossEntropyWithLogits(
  logits: Tensor,
  labels: Tensor,
  options: { readonly reduction?: "none" | "sum" | "mean" } = {},
): Tensor {
  return logits.softmaxCrossEntropyWithLogits(labels, options);
}
