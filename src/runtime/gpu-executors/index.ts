import {
  executeBinary,
  executeBinaryGrad,
  executeRelu,
  executeReluGrad,
  executeSpecial,
  executeUnary,
} from "./elementwise.js";
import { executeMatmul } from "./linear-algebra.js";
import { executeReshape } from "./shape.js";
import { executeSoftmax, executeSoftmaxGrad } from "./normalization.js";
import { executeReduction, executeReductionGrad } from "./reductions.js";
import {
  executeSoftmaxCrossEntropy,
  executeSoftmaxCrossEntropyGrad,
} from "./loss.js";
import {
  executeConv2d,
  executeConv2dGrad,
  executePool2d,
  executePool2dGrad,
} from "./spatial.js";
import type {
  GpuNodeExecutor,
  GpuNodeExecutorMap,
  GpuNodeKind,
} from "./types.js";

export const GPU_NODE_EXECUTORS: GpuNodeExecutorMap = {
  relu: executeRelu,
  reluGrad: executeReluGrad,
  unary: executeUnary,
  special: executeSpecial,
  binary: executeBinary,
  binaryGrad: executeBinaryGrad,
  softmax: executeSoftmax,
  softmaxGrad: executeSoftmaxGrad,
  matmul: executeMatmul,
  conv2d: executeConv2d,
  pool2d: executePool2d,
  conv2dGrad: executeConv2dGrad,
  pool2dGrad: executePool2dGrad,
  reduceGrad: executeReductionGrad,
  softmaxCrossEntropyGrad: executeSoftmaxCrossEntropyGrad,
  reduce: executeReduction,
  softmaxCrossEntropy: executeSoftmaxCrossEntropy,
  reshape: executeReshape,
};

export function executorFor<K extends GpuNodeKind>(
  kind: K,
): GpuNodeExecutor<K> {
  return GPU_NODE_EXECUTORS[kind] as GpuNodeExecutor<K>;
}
