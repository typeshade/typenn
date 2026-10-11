import { argMaxKernel } from "../../kernels/arg-max.shade.ts";
import { argMinKernel } from "../../kernels/arg-min.shade.ts";
import { maxKernel } from "../../kernels/max.shade.ts";
import { meanKernel } from "../../kernels/mean.shade.ts";
import { minKernel } from "../../kernels/min.shade.ts";
import { prodKernel } from "../../kernels/prod.shade.ts";
import { reduceGradKernel } from "../../kernels/reduce-grad.shade.ts";
import { sumKernel } from "../../kernels/sum.shade.ts";
import { createOutput, dispatchCpuKernel } from "./common.js";
import type { CpuNodeExecutor } from "./types.js";

const REDUCTION_KERNELS = {
  sum: sumKernel,
  mean: meanKernel,
  prod: prodKernel,
  max: maxKernel,
  min: minKernel,
  argMax: argMaxKernel,
  argMin: argMinKernel,
};

export const executeReduction: CpuNodeExecutor<"reduce"> = async (
  tensor,
  node,
  evaluate,
  onDispatch,
) => {
  const input = await evaluate(node.input);
  const output = createOutput(tensor);
  await dispatchCpuKernel(
    REDUCTION_KERNELS[node.op],
    {
      shape: {
        outputLength: output.length,
        axisSize: node.axisSize,
        inner: node.inner,
      },
      input,
      output,
    },
    output.length,
    onDispatch,
  );
  return output;
};

export const executeReductionGrad: CpuNodeExecutor<"reduceGrad"> = async (
  tensor,
  node,
  evaluate,
  onDispatch,
) => {
  const dy = await evaluate(node.dy);
  const output = createOutput(tensor);
  await dispatchCpuKernel(
    reduceGradKernel,
    {
      reduceShape: {
        axisSize: node.axisSize,
        inner: node.inner,
        inputLength: output.length,
        meanScale: node.meanScale,
      },
      reduceDy: dy,
      reduceOutput: output,
    },
    output.length,
    onDispatch,
  );
  return output;
};
