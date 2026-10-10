import type { Resident } from "typeshade/runtime";
import type { TensorNode } from "../../tensor/node.js";
import type { Tensor } from "../../tensor/tensor.js";
import type { Operation } from "../kernel-registry.js";

export type GpuNodeKind = Exclude<TensorNode["kind"], "input">;
export type GpuNode<K extends GpuNodeKind = GpuNodeKind> = Extract<
  TensorNode,
  { kind: K }
>;
export type GpuScheduler = (tensor: Tensor) => Promise<Resident<Float32Array>>;

export interface GpuExecutionContext<K extends GpuNodeKind = GpuNodeKind> {
  readonly tensor: Tensor;
  readonly node: GpuNode<K>;
  readonly output: Resident<Float32Array>;
  readonly schedule: GpuScheduler;
}

export interface GpuDispatch {
  readonly operation: Operation;
  readonly bindings: Record<string, unknown>;
  readonly workgroups: number;
}

export type GpuNodeExecutor<K extends GpuNodeKind> = (
  context: GpuExecutionContext<K>,
) => Promise<GpuDispatch>;

export type GpuNodeExecutorMap = {
  [K in GpuNodeKind]: GpuNodeExecutor<K>;
};
