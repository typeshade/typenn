import type { TensorNode } from "../tensor/node.js";
import { Tensor } from "../tensor/tensor.js";
import {
  executeBinary,
  executeBinaryGrad,
  executeInput,
  executeRelu,
  executeReluGrad,
  executeSpecial,
  executeUnary,
} from "./cpu-executors/elementwise.js";
import { executeMatmul } from "./cpu-executors/linear-algebra.js";
import {
  executeSoftmaxCrossEntropy,
  executeSoftmaxCrossEntropyGrad,
} from "./cpu-executors/loss.js";
import {
  executeSoftmax,
  executeSoftmaxGrad,
} from "./cpu-executors/normalization.js";
import {
  executeReduction,
  executeReductionGrad,
} from "./cpu-executors/reductions.js";
import {
  executeConv2d,
  executeConv2dGrad,
  executePool2d,
  executePool2dGrad,
} from "./cpu-executors/spatial.js";
import type {
  CpuNodeExecutor,
  CpuNodeExecutorMap,
  CpuNodeKind,
} from "./cpu-executors/types.js";

const NODE_EXECUTORS: CpuNodeExecutorMap = {
  input: executeInput,
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
};

const pendingEvaluations = new WeakMap<Tensor, Promise<Float32Array>>();

/** Execute a deferred Tensor graph through the CPU implementations of Shade kernels. */
export async function evaluateCpu(tensor: Tensor): Promise<Float32Array> {
  const materialized = tensor.getMaterialized();
  if (materialized instanceof Float32Array) return materialized;

  const pending = pendingEvaluations.get(tensor);
  if (pending) return pending;

  const task = executeNode(tensor, tensor.node)
    .then((value) => {
      tensor.setMaterialized(value);
      return value;
    })
    .finally(() => pendingEvaluations.delete(tensor));
  pendingEvaluations.set(tensor, task);
  return task;
}

function executeNode<K extends CpuNodeKind>(
  tensor: Tensor,
  node: Extract<TensorNode, { kind: K }>,
): Promise<Float32Array> {
  const execute = NODE_EXECUTORS[node.kind] as CpuNodeExecutor<K>;
  return execute(tensor, node, evaluateCpu);
}
