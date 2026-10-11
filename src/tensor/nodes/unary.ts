import type { Tensor } from "../tensor.js";

export type UnaryOperation =
  | "acos"
  | "acosh"
  | "asin"
  | "asinh"
  | "atan"
  | "neg"
  | "abs"
  | "square"
  | "sqrt"
  | "exp"
  | "log"
  | "sigmoid"
  | "tanh"
  | "floor"
  | "ceil"
  | "round"
  | "sign"
  | "reciprocal"
  | "rsqrt"
  | "sin"
  | "cos"
  | "tan"
  | "expm1"
  | "log1p"
  | "sinh"
  | "cosh"
  | "atanh"
  | "step";

/** @internal */
export interface UnaryNode {
  readonly kind: "unary";
  readonly op: UnaryOperation;
  readonly input: Tensor;
}
