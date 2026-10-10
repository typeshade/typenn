import type { TensorNode } from "../../tensor/node.js";
import type { Tensor } from "../../tensor/tensor.js";

export type DifferentiableNodeKind =
  | "relu"
  | "unary"
  | "special"
  | "binary"
  | "softmax"
  | "matmul"
  | "conv2d"
  | "pool2d"
  | "reduce"
  | "softmaxCrossEntropy";

export type GradientRuleContext<
  K extends DifferentiableNodeKind = DifferentiableNodeKind,
> = {
  readonly value: Tensor;
  readonly node: Extract<TensorNode, { kind: K }>;
  readonly upstream: Tensor;
  readonly gradients: Map<Tensor, Tensor>;
};

export interface GradientExecutionContext {
  readonly value: Tensor;
  readonly node: TensorNode;
  readonly upstream: Tensor;
  readonly gradients: Map<Tensor, Tensor>;
}

export type GradientRule<K extends DifferentiableNodeKind> = (
  context: GradientRuleContext<K>,
) => void;

export type GradientRuleMap = {
  [K in DifferentiableNodeKind]: GradientRule<K>;
};
