import type { Tensor } from "../tensor.js";

/** @internal */
export interface ReshapeNode {
  readonly kind: "reshape";
  readonly input: Tensor;
}
