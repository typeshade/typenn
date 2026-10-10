import { logSoftmaxKernel } from "../../kernels/log-softmax.shade.ts";
import { softmaxGradKernel } from "../../kernels/softmax-grad.shade.ts";
import { softmaxKernel } from "../../kernels/softmax.shade.ts";
import { createOutput, dispatchCpuKernel } from "./common.js";
import type { CpuNodeExecutor } from "./types.js";

const NORMALIZATION_KERNELS = {
  softmax: softmaxKernel,
  logSoftmax: logSoftmaxKernel,
};

export const executeSoftmax: CpuNodeExecutor<"softmax"> = async (
  tensor,
  node,
  evaluate,
) => {
  const input = await evaluate(node.input);
  const output = createOutput(tensor);
  const sliceCount = node.outer * node.inner;
  await dispatchCpuKernel(
    NORMALIZATION_KERNELS[node.op],
    {
      shape: {
        outer: node.outer,
        axisSize: node.axisSize,
        inner: node.inner,
      },
      input,
      output,
    },
    sliceCount,
  );
  return output;
};

export const executeSoftmaxGrad: CpuNodeExecutor<"softmaxGrad"> = async (
  tensor,
  node,
  evaluate,
) => {
  const [value, dy] = await Promise.all([
    evaluate(node.value),
    evaluate(node.dy),
  ]);
  const output = createOutput(tensor);
  const sliceCount = node.outer * node.inner;
  await dispatchCpuKernel(
    softmaxGradKernel,
    {
      shape: {
        outer: node.outer,
        axisSize: node.axisSize,
        inner: node.inner,
        logMode: node.op === "logSoftmaxGrad" ? 1 : 0,
      },
      value,
      dy,
      output,
    },
    sliceCount,
  );
  return output;
};
