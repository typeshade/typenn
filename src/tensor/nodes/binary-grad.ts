import type { BinaryOperation } from "./binary.js";
import type { Tensor } from "../tensor.js";

/** @internal */
export interface BinaryGradNode {
  readonly kind: "binaryGrad";
  readonly op: BinaryOperation;
  readonly a: Tensor;
  readonly b: Tensor;
  readonly dy: Tensor;
  readonly dimensions: number[];
  readonly aStrides: number[];
  readonly bStrides: number[];
  readonly wrtA: boolean;
}
