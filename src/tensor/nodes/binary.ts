import type { Tensor } from "../tensor.js";

export type BinaryOperation =
  "add" | "sub" | "mul" | "div" | "maximum" | "minimum" | "pow" | "prelu";

/** @internal */
export interface BinaryNode {
  readonly kind: "binary";
  readonly op: BinaryOperation;
  readonly a: Tensor;
  readonly b: Tensor;
  readonly dimensions: number[];
  readonly aStrides: number[];
  readonly bStrides: number[];
}
