import type { Tensor } from "../tensor.js";

/** @internal */
export interface MatmulNode {
  readonly kind: "matmul";
  readonly a: Tensor;
  readonly b: Tensor;
  readonly inner: number;
  readonly transposeA: boolean;
  readonly transposeB: boolean;
}
