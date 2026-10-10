import type { Tensor } from "../tensor.js";

/** @internal */
export interface SoftmaxGradNode {
  readonly kind: "softmaxGrad";
  readonly op: "softmaxGrad" | "logSoftmaxGrad";
  readonly value: Tensor;
  readonly dy: Tensor;
  readonly outer: number;
  readonly axisSize: number;
  readonly inner: number;
}
