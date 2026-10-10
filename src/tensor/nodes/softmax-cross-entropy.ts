import type { Tensor } from "../tensor.js";

/** @internal */
export interface SoftmaxCrossEntropyNode {
  readonly kind: "softmaxCrossEntropy";
  readonly logits: Tensor;
  readonly labels: Tensor;
  readonly rows: number;
  readonly columns: number;
}
