import { crossEntropyLabelsGradKernel } from "../../kernels/cross-entropy-labels-grad.shade.ts";
import { crossEntropyLogitsGradKernel } from "../../kernels/cross-entropy-logits-grad.shade.ts";
import { softmaxCrossEntropyWithLogitsKernel } from "../../kernels/softmax-cross-entropy-with-logits.shade.ts";
import { createOutput, dispatchCpuKernel } from "./common.js";
import type { CpuNodeExecutor } from "./types.js";

const CROSS_ENTROPY_GRADIENTS = {
  crossEntropyLogitsGrad: crossEntropyLogitsGradKernel,
  crossEntropyLabelsGrad: crossEntropyLabelsGradKernel,
};

export const executeSoftmaxCrossEntropy: CpuNodeExecutor<
  "softmaxCrossEntropy"
> = async (tensor, node, evaluate, onDispatch) => {
  const [logits, labels] = await Promise.all([
    evaluate(node.logits),
    evaluate(node.labels),
  ]);
  const output = createOutput(tensor);
  await dispatchCpuKernel(
    softmaxCrossEntropyWithLogitsKernel,
    {
      shape: { rows: node.rows, columns: node.columns },
      logits,
      labels,
      output,
    },
    node.rows,
    onDispatch,
  );
  return output;
};

export const executeSoftmaxCrossEntropyGrad: CpuNodeExecutor<
  "softmaxCrossEntropyGrad"
> = async (tensor, node, evaluate, onDispatch) => {
  const [logits, labels, upstream] = await Promise.all([
    evaluate(node.logits),
    evaluate(node.labels),
    evaluate(node.dy),
  ]);
  const output = createOutput(tensor);
  await dispatchCpuKernel(
    CROSS_ENTROPY_GRADIENTS[node.op],
    {
      crossEntropyShape: {
        rows: node.rows,
        columns: node.columns,
        length: output.length,
      },
      logits,
      labels,
      upstream,
      gradient: output,
    },
    output.length,
    onDispatch,
  );
  return output;
};
