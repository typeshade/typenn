import type { Pack } from "typeshade/runtime";
import reluPack from "../kernels/relu.shade.ts";
import reluGradPack from "../kernels/relu-grad.shade.ts";
import binaryGradPack from "../kernels/binary-grad.shade.ts";
import softmaxGradPack from "../kernels/softmax-grad.shade.ts";
import addPack from "../kernels/add.shade.ts";
import subPack from "../kernels/sub.shade.ts";
import mulPack from "../kernels/mul.shade.ts";
import divPack from "../kernels/div.shade.ts";
import maximumPack from "../kernels/maximum.shade.ts";
import minimumPack from "../kernels/minimum.shade.ts";
import powPack from "../kernels/pow.shade.ts";
import preluPack from "../kernels/prelu.shade.ts";
import sumPack from "../kernels/sum.shade.ts";
import meanPack from "../kernels/mean.shade.ts";
import prodPack from "../kernels/prod.shade.ts";
import argMaxPack from "../kernels/arg-max.shade.ts";
import argMinPack from "../kernels/arg-min.shade.ts";
import maxPack from "../kernels/max.shade.ts";
import minPack from "../kernels/min.shade.ts";
import negPack from "../kernels/neg.shade.ts";
import absPack from "../kernels/abs.shade.ts";
import squarePack from "../kernels/square.shade.ts";
import sqrtPack from "../kernels/sqrt.shade.ts";
import expPack from "../kernels/exp.shade.ts";
import logPack from "../kernels/log.shade.ts";
import sigmoidPack from "../kernels/sigmoid.shade.ts";
import tanhPack from "../kernels/tanh.shade.ts";
import floorPack from "../kernels/floor.shade.ts";
import ceilPack from "../kernels/ceil.shade.ts";
import roundPack from "../kernels/round.shade.ts";
import signPack from "../kernels/sign.shade.ts";
import reciprocalPack from "../kernels/reciprocal.shade.ts";
import rsqrtPack from "../kernels/rsqrt.shade.ts";
import sinPack from "../kernels/sin.shade.ts";
import cosPack from "../kernels/cos.shade.ts";
import tanPack from "../kernels/tan.shade.ts";
import expm1Pack from "../kernels/expm1.shade.ts";
import log1pPack from "../kernels/log1p.shade.ts";
import sinhPack from "../kernels/sinh.shade.ts";
import coshPack from "../kernels/cosh.shade.ts";
import clipByValuePack from "../kernels/clip-by-value.shade.ts";
import geluPack from "../kernels/gelu.shade.ts";
import leakyReluPack from "../kernels/leaky-relu.shade.ts";
import eluPack from "../kernels/elu.shade.ts";
import relu6Pack from "../kernels/relu6.shade.ts";
import softplusPack from "../kernels/softplus.shade.ts";
import softmaxCrossEntropyWithLogitsPack from "../kernels/softmax-cross-entropy-with-logits.shade.ts";
import linearForwardPack from "../kernels/linear-forward.shade.ts";
import linearObjectivePack from "../kernels/linear-objective.shade.ts";
import linearReducePack from "../kernels/linear-reduce.shade.ts";
import linearBackwardPack from "../kernels/linear-backward.shade.ts";
import linearUpdatePack from "../kernels/linear-update.shade.ts";
import matmulPack from "../kernels/matmul.shade.ts";
import conv2dPack from "../kernels/conv2d.shade.ts";
import maxPoolPack from "../kernels/max-pool.shade.ts";
import avgPoolPack from "../kernels/avg-pool.shade.ts";
import conv2dInputGradPack from "../kernels/conv2d-input-grad.shade.ts";
import conv2dFilterGradPack from "../kernels/conv2d-filter-grad.shade.ts";
import maxPoolGradPack from "../kernels/max-pool-grad.shade.ts";
import avgPoolGradPack from "../kernels/avg-pool-grad.shade.ts";
import reduceGradPack from "../kernels/reduce-grad.shade.ts";
import crossEntropyLogitsGradPack from "../kernels/cross-entropy-logits-grad.shade.ts";
import crossEntropyLabelsGradPack from "../kernels/cross-entropy-labels-grad.shade.ts";
import softmaxPack from "../kernels/softmax.shade.ts";
import logSoftmaxPack from "../kernels/log-softmax.shade.ts";
import acosPack from "../kernels/acos.shade.ts";
import acoshPack from "../kernels/acosh.shade.ts";
import asinPack from "../kernels/asin.shade.ts";
import asinhPack from "../kernels/asinh.shade.ts";
import atanPack from "../kernels/atan.shade.ts";
import atanhPack from "../kernels/atanh.shade.ts";
import stepPack from "../kernels/step.shade.ts";
import reshapePack from "../kernels/reshape.shade.ts";
import type { FuncDecl } from "typeshade";
import {
  ELEMENTWISE_OP_DECLARATIONS,
  ELEMENTWISE_OP_IR_FINGERPRINT,
} from "./elementwise-op-declarations.generated.js";

type BinaryOperation =
  "add" | "sub" | "mul" | "div" | "maximum" | "minimum" | "pow" | "prelu";
type ReductionOperation =
  "sum" | "mean" | "prod" | "max" | "min" | "argMax" | "argMin";
type UnaryOperation =
  | "acos"
  | "acosh"
  | "asin"
  | "asinh"
  | "atan"
  | "atanh"
  | "step"
  | "reshape"
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
  | "cosh";
type SpecialOperation =
  "clipByValue" | "gelu" | "leakyRelu" | "elu" | "relu6" | "softplus";
export type LinearTrainingEntry =
  | "linearForward"
  | "linearObjective"
  | "linearReduce"
  | "linearBackward"
  | "linearUpdate";
export type Operation =
  | "relu"
  | "reluGrad"
  | "binaryGrad"
  | "softmaxGrad"
  | "logSoftmaxGrad"
  | BinaryOperation
  | ReductionOperation
  | UnaryOperation
  | SpecialOperation
  | "matmul"
  | "conv2d"
  | "maxPool"
  | "avgPool"
  | "conv2dInputGrad"
  | "conv2dFilterGrad"
  | "maxPoolGrad"
  | "avgPoolGrad"
  | "reduceGrad"
  | "crossEntropyLogitsGrad"
  | "crossEntropyLabelsGrad"
  | "softmax"
  | "logSoftmax"
  | "softmaxCrossEntropyWithLogits"
  | LinearTrainingEntry;

export const PACKS: Record<Operation, Pack> = {
  relu: reluPack,
  reluGrad: reluGradPack,
  binaryGrad: binaryGradPack,
  softmaxGrad: softmaxGradPack,
  logSoftmaxGrad: softmaxGradPack,
  add: addPack,
  sub: subPack,
  mul: mulPack,
  div: divPack,
  maximum: maximumPack,
  minimum: minimumPack,
  pow: powPack,
  prelu: preluPack,
  sum: sumPack,
  mean: meanPack,
  prod: prodPack,
  argMax: argMaxPack,
  argMin: argMinPack,
  max: maxPack,
  min: minPack,
  neg: negPack,
  abs: absPack,
  square: squarePack,
  sqrt: sqrtPack,
  exp: expPack,
  log: logPack,
  sigmoid: sigmoidPack,
  tanh: tanhPack,
  floor: floorPack,
  ceil: ceilPack,
  round: roundPack,
  sign: signPack,
  reciprocal: reciprocalPack,
  rsqrt: rsqrtPack,
  sin: sinPack,
  cos: cosPack,
  tan: tanPack,
  expm1: expm1Pack,
  log1p: log1pPack,
  sinh: sinhPack,
  cosh: coshPack,
  clipByValue: clipByValuePack,
  gelu: geluPack,
  leakyRelu: leakyReluPack,
  elu: eluPack,
  relu6: relu6Pack,
  softplus: softplusPack,
  softmaxCrossEntropyWithLogits: softmaxCrossEntropyWithLogitsPack,
  linearForward: linearForwardPack,
  linearObjective: linearObjectivePack,
  linearReduce: linearReducePack,
  linearBackward: linearBackwardPack,
  linearUpdate: linearUpdatePack,
  matmul: matmulPack,
  conv2d: conv2dPack,
  maxPool: maxPoolPack,
  avgPool: avgPoolPack,
  conv2dInputGrad: conv2dInputGradPack,
  conv2dFilterGrad: conv2dFilterGradPack,
  maxPoolGrad: maxPoolGradPack,
  avgPoolGrad: avgPoolGradPack,
  reduceGrad: reduceGradPack,
  crossEntropyLogitsGrad: crossEntropyLogitsGradPack,
  crossEntropyLabelsGrad: crossEntropyLabelsGradPack,
  softmax: softmaxPack,
  logSoftmax: logSoftmaxPack,
  acos: acosPack,
  acosh: acoshPack,
  asin: asinPack,
  asinh: asinhPack,
  atan: atanPack,
  atanh: atanhPack,
  step: stepPack,
  reshape: reshapePack,
};

export const ENTRYPOINT: Record<Operation, string> = {
  relu: "reluKernel",
  reluGrad: "reluGradKernel",
  binaryGrad: "binaryGradKernel",
  softmaxGrad: "softmaxGradKernel",
  logSoftmaxGrad: "softmaxGradKernel",
  add: "addKernel",
  sub: "subKernel",
  mul: "mulKernel",
  div: "divKernel",
  maximum: "maximumKernel",
  minimum: "minimumKernel",
  pow: "powKernel",
  prelu: "preluKernel",
  sum: "sumKernel",
  mean: "meanKernel",
  prod: "prodKernel",
  argMax: "argMaxKernel",
  argMin: "argMinKernel",
  max: "maxKernel",
  min: "minKernel",
  neg: "negKernel",
  abs: "absKernel",
  square: "squareKernel",
  sqrt: "sqrtKernel",
  exp: "expKernel",
  log: "logKernel",
  sigmoid: "sigmoidKernel",
  tanh: "tanhKernel",
  floor: "floorKernel",
  ceil: "ceilKernel",
  round: "roundKernel",
  sign: "signKernel",
  reciprocal: "reciprocalKernel",
  rsqrt: "rsqrtKernel",
  sin: "sinKernel",
  cos: "cosKernel",
  tan: "tanKernel",
  expm1: "expm1Kernel",
  log1p: "log1pKernel",
  sinh: "sinhKernel",
  cosh: "coshKernel",
  clipByValue: "clipByValueKernel",
  gelu: "geluKernel",
  leakyRelu: "leakyReluKernel",
  elu: "eluKernel",
  relu6: "relu6Kernel",
  softplus: "softplusKernel",
  softmaxCrossEntropyWithLogits: "softmaxCrossEntropyWithLogitsKernel",
  linearForward: "linearForwardKernel",
  linearObjective: "linearObjectiveKernel",
  linearReduce: "linearReduceKernel",
  linearBackward: "linearBackwardKernel",
  linearUpdate: "linearUpdateKernel",
  matmul: "matmulKernel",
  conv2d: "conv2dKernel",
  maxPool: "maxPoolKernel",
  avgPool: "avgPoolKernel",
  conv2dInputGrad: "conv2dInputGradKernel",
  conv2dFilterGrad: "conv2dFilterGradKernel",
  maxPoolGrad: "maxPoolGradKernel",
  avgPoolGrad: "avgPoolGradKernel",
  reduceGrad: "reduceGradKernel",
  crossEntropyLogitsGrad: "crossEntropyLogitsGradKernel",
  crossEntropyLabelsGrad: "crossEntropyLabelsGradKernel",
  softmax: "softmaxKernel",
  logSoftmax: "logSoftmaxKernel",
  acos: "acosKernel",
  acosh: "acoshKernel",
  asin: "asinKernel",
  asinh: "asinhKernel",
  atan: "atanKernel",
  atanh: "atanhKernel",
  step: "stepKernel",
  reshape: "reshapeKernel",
};

/** Tensor node fields consumed by the helper's named scalar inputs. */
export interface ElementwiseFusionOperation {
  readonly name: string;
  readonly nodeKind: "binary" | "relu";
  readonly inputFields: readonly string[];
  readonly helperName: string;
  readonly helperInputNames: readonly string[];
}

export interface ElementwiseFusionRegistry {
  readonly fingerprint: string;
  readonly operations: Readonly<Record<string, ElementwiseFusionOperation>>;
  readonly helpers: ReadonlyMap<string, FuncDecl>;
}

/**
 * Build-time extracted TypeShade functions are the scalar operation implementations used by
 * both standalone kernels and the graph composer. The graph planner only uses this port table
 * to connect Tensor edges; it never selects an arithmetic formula.
 */
export const ELEMENTWISE_FUSION_OPS: Readonly<
  Record<string, ElementwiseFusionOperation>
> = {
  add: {
    name: "add",
    nodeKind: "binary",
    inputFields: ["a", "b"],
    helperName: "typennAddF32",
    helperInputNames: ["left", "right"],
  },
  mul: {
    name: "mul",
    nodeKind: "binary",
    inputFields: ["a", "b"],
    helperName: "typennMulF32",
    helperInputNames: ["left", "right"],
  },
  relu: {
    name: "relu",
    nodeKind: "relu",
    inputFields: ["input"],
    helperName: "typennReluF32",
    helperInputNames: ["inputValue"],
  },
};

export function createElementwiseFusionRegistry(input: {
  readonly declarations: readonly FuncDecl[];
  readonly fingerprint: string;
}): ElementwiseFusionRegistry {
  return {
    fingerprint: input.fingerprint,
    operations: ELEMENTWISE_FUSION_OPS,
    helpers: new Map(
      input.declarations.map((declaration) => [declaration.name, declaration]),
    ),
  };
}

export const ELEMENTWISE_FUSION_REGISTRY = createElementwiseFusionRegistry({
  declarations: ELEMENTWISE_OP_DECLARATIONS,
  fingerprint: ELEMENTWISE_OP_IR_FINGERPRINT,
});

export const ELEMENTWISE_OP_HELPERS = ELEMENTWISE_FUSION_REGISTRY.helpers;
