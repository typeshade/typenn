import type { Tensor } from "../tensor.js";

/** @internal */
export interface Pool2dGradNode {
  readonly kind: "pool2dGrad";
  readonly op: "maxPoolGrad" | "avgPoolGrad";
  readonly input: Tensor;
  readonly dy: Tensor;
  readonly filterHeight: number;
  readonly filterWidth: number;
  readonly outputHeight: number;
  readonly outputWidth: number;
  readonly strideY: number;
  readonly strideX: number;
  readonly padTop: number;
  readonly padLeft: number;
}
