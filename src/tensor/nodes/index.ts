import type { InputNode } from "./input.js";
import type { ReluNode } from "./relu.js";
import type { ReluGradNode } from "./relu-grad.js";
import type { UnaryNode } from "./unary.js";
import type { SpecialNode } from "./special.js";
import type { BinaryNode } from "./binary.js";
import type { BinaryGradNode } from "./binary-grad.js";
import type { SoftmaxNode } from "./softmax.js";
import type { SoftmaxGradNode } from "./softmax-grad.js";
import type { MatmulNode } from "./matmul.js";
import type { Conv2dNode } from "./conv2d.js";
import type { Pool2dNode } from "./pool2d.js";
import type { Conv2dGradNode } from "./conv2d-grad.js";
import type { Pool2dGradNode } from "./pool2d-grad.js";
import type { ReduceNode } from "./reduce.js";
import type { ReduceGradNode } from "./reduce-grad.js";
import type { SoftmaxCrossEntropyNode } from "./softmax-cross-entropy.js";
import type { SoftmaxCrossEntropyGradNode } from "./softmax-cross-entropy-grad.js";
import type { ReshapeNode } from "./reshape.js";

export type { BinaryOperation, BinaryNode } from "./binary.js";
export type { BinaryGradNode } from "./binary-grad.js";
export type { Conv2dNode } from "./conv2d.js";
export type { Conv2dGradNode } from "./conv2d-grad.js";
export type { InputNode } from "./input.js";
export type { MatmulNode } from "./matmul.js";
export type { Pool2dNode } from "./pool2d.js";
export type { Pool2dGradNode } from "./pool2d-grad.js";
export type { ReduceNode, ReductionOperation } from "./reduce.js";
export type { ReduceGradNode } from "./reduce-grad.js";
export type { ReshapeNode } from "./reshape.js";
export type { ReluNode } from "./relu.js";
export type { ReluGradNode } from "./relu-grad.js";
export type { SoftmaxNode } from "./softmax.js";
export type { SoftmaxGradNode } from "./softmax-grad.js";
export type { SoftmaxCrossEntropyNode } from "./softmax-cross-entropy.js";
export type { SoftmaxCrossEntropyGradNode } from "./softmax-cross-entropy-grad.js";
export type { SpecialNode, SpecialOperation } from "./special.js";
export type { UnaryNode, UnaryOperation } from "./unary.js";

/** @internal */
export type TensorNode =
  | InputNode
  | ReluNode
  | ReluGradNode
  | UnaryNode
  | SpecialNode
  | BinaryNode
  | BinaryGradNode
  | SoftmaxNode
  | SoftmaxGradNode
  | MatmulNode
  | Conv2dNode
  | Pool2dNode
  | Conv2dGradNode
  | Pool2dGradNode
  | ReduceNode
  | ReduceGradNode
  | SoftmaxCrossEntropyNode
  | SoftmaxCrossEntropyGradNode
  | ReshapeNode;
