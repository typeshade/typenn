import type { Tensor } from "../tensor.js";

/** @internal */
export interface SoftmaxNode {
  readonly kind: "softmax";
  readonly op: "softmax" | "logSoftmax";
  readonly input: Tensor;
  readonly axis: number;
  readonly outer: number;
  readonly axisSize: number;
  readonly inner: number;
}
