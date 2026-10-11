import { reshapeKernel } from "../../kernels/reshape.shade.ts";
import { createOutput, dispatchCpuKernel } from "./common.js";
import type { CpuNodeExecutor } from "./types.js";

export const executeReshape: CpuNodeExecutor<"reshape"> = async (
  tensor,
  node,
  evaluate,
  onDispatch,
) => {
  const input = await evaluate(node.input);
  const output = createOutput(tensor);
  await dispatchCpuKernel(
    reshapeKernel,
    { shape: { length: output.length }, input, output },
    output.length,
    onDispatch,
  );
  return output;
};
