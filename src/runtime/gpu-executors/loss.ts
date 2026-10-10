import { sizeOf } from "../../tensor/shape.js";
import type { GpuNodeExecutor } from "./types.js";

export const executeSoftmaxCrossEntropy: GpuNodeExecutor<
  "softmaxCrossEntropy"
> = async ({ node, output, schedule }) => {
  const [logits, labels] = await Promise.all([
    schedule(node.logits),
    schedule(node.labels),
  ]);
  return {
    operation: "softmaxCrossEntropyWithLogits",
    bindings: {
      shape: { rows: node.rows, columns: node.columns },
      logits,
      labels,
      output,
    },
    workgroups: Math.ceil(node.rows / 64),
  };
};

export const executeSoftmaxCrossEntropyGrad: GpuNodeExecutor<
  "softmaxCrossEntropyGrad"
> = async ({ tensor, node, output, schedule }) => {
  const [logits, labels, upstream] = await Promise.all([
    schedule(node.logits),
    schedule(node.labels),
    schedule(node.dy),
  ]);
  return {
    operation: node.op,
    bindings: {
      crossEntropyShape: {
        rows: node.rows,
        columns: node.columns,
        length: sizeOf(tensor.shape),
      },
      logits,
      labels,
      upstream,
      gradient: output,
    },
    workgroups: Math.ceil(sizeOf(tensor.shape) / 64),
  };
};
