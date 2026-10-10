import type { Tensor } from "../tensor.js";

/** @internal */
export interface ReduceGradNode {
  readonly kind: "reduceGrad";
  readonly input: Tensor;
  readonly dy: Tensor;
  readonly outer: number;
  readonly axisSize: number;
  readonly inner: number;
  readonly meanScale: number;
}
