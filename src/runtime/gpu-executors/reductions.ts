import { sizeOf } from "../../tensor/shape.js";
import type { GpuNodeExecutor } from "./types.js";

export const executeReduction: GpuNodeExecutor<"reduce"> = async ({
  tensor,
  node,
  output,
  schedule,
}) => ({
  operation: node.op,
  bindings: {
    shape: {
      outputLength: sizeOf(tensor.shape),
      outer: node.outer,
      axisSize: node.axisSize,
      inner: node.inner,
    },
    input: await schedule(node.input),
    output,
  },
  workgroups: Math.ceil(sizeOf(tensor.shape) / 64),
});

export const executeReductionGrad: GpuNodeExecutor<"reduceGrad"> = async ({
  tensor,
  node,
  output,
  schedule,
}) => ({
  operation: "reduceGrad",
  bindings: {
    reduceShape: {
      axisSize: node.axisSize,
      inner: node.inner,
      inputLength: sizeOf(tensor.shape),
      meanScale: node.meanScale,
    },
    reduceDy: await schedule(node.dy),
    reduceOutput: output,
  },
  workgroups: Math.ceil(sizeOf(tensor.shape) / 64),
});
