import type { Tensor } from "../tensor.js";

export type SpecialOperation =
  "clipByValue" | "gelu" | "leakyRelu" | "elu" | "relu6" | "softplus";

/** @internal */
export interface SpecialNode {
  readonly kind: "special";
  readonly op: SpecialOperation;
  readonly input: Tensor;
  readonly alpha: number;
  readonly minimum: number;
  readonly maximum: number;
}
