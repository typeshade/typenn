import {
  binaryGradientRule,
  reluGradient,
  specialGradient,
  unaryGradientRule,
} from "./elementwise.js";
import { matmulGradient } from "./linear-algebra.js";
import { conv2dGradient, pool2dGradient } from "./spatial.js";
import { reductionGradient } from "./reductions.js";
import { softmaxGradient } from "./normalization.js";
import { softmaxCrossEntropyGradient } from "./loss.js";
import { reshapeGradient } from "./shape.js";
import type {
  DifferentiableNodeKind,
  GradientExecutionContext,
  GradientRuleMap,
} from "./types.js";

const GRADIENT_RULES: GradientRuleMap = {
  relu: reluGradient,
  unary: unaryGradientRule,
  special: specialGradient,
  binary: binaryGradientRule,
  softmax: softmaxGradient,
  matmul: matmulGradient,
  conv2d: conv2dGradient,
  pool2d: pool2dGradient,
  reduce: reductionGradient,
  softmaxCrossEntropy: softmaxCrossEntropyGradient,
  reshape: reshapeGradient,
};

export function applyGradientRule(context: GradientExecutionContext): void {
  const rule = GRADIENT_RULES[context.node.kind as DifferentiableNodeKind] as
    ((context: GradientExecutionContext) => void) | undefined;
  if (rule) {
    rule(context);
    return;
  }
  throw new Error(
    `Autograd for TypeNN ${context.node.kind} nodes is not implemented yet.`,
  );
}
