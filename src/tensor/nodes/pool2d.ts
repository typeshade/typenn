import type { Tensor } from "../tensor.js";

/** @internal */
export interface Pool2dNode {
  readonly kind: "pool2d";
  readonly op: "maxPool" | "avgPool";
  readonly input: Tensor;
  readonly filterHeight: number;
  readonly filterWidth: number;
  readonly outputHeight: number;
  readonly outputWidth: number;
  readonly strideY: number;
  readonly strideX: number;
  readonly padTop: number;
  readonly padLeft: number;
}
