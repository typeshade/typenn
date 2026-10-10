import type { Tensor } from "../tensor.js";

/** @internal */
export interface ReluGradNode {
  readonly kind: "reluGrad";
  readonly input: Tensor;
  readonly dy: Tensor;
}
