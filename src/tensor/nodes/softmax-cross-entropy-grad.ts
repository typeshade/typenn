import type { Tensor } from "../tensor.js";

/** @internal */
export interface SoftmaxCrossEntropyGradNode {
  readonly kind: "softmaxCrossEntropyGrad";
  readonly op: "crossEntropyLogitsGrad" | "crossEntropyLabelsGrad";
  readonly logits: Tensor;
  readonly labels: Tensor;
  readonly dy: Tensor;
  readonly rows: number;
  readonly columns: number;
}
