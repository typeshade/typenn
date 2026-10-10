import type { Tensor } from "../tensor.js";

/** @internal */
export interface ReluNode {
  readonly kind: "relu";
  readonly input: Tensor;
}
