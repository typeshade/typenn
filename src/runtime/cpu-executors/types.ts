import type { TensorNode } from "../../tensor/node.js";
import type { Tensor } from "../../tensor/tensor.js";

export type CpuEvaluator = (tensor: Tensor) => Promise<Float32Array>;
export type CpuNodeKind = TensorNode["kind"];
export type CpuNode<K extends CpuNodeKind> = Extract<TensorNode, { kind: K }>;

export type CpuNodeExecutor<K extends CpuNodeKind> = (
  tensor: Tensor,
  node: CpuNode<K>,
  evaluate: CpuEvaluator,
  onDispatch?: () => void,
) => Promise<Float32Array>;

export type CpuNodeExecutorMap = {
  [K in CpuNodeKind]: CpuNodeExecutor<K>;
};
