import { matmulKernel } from "../../kernels/matmul.shade.ts";
import { createOutput, dispatchCpuKernel } from "./common.js";
import type { CpuNodeExecutor } from "./types.js";

export const executeMatmul: CpuNodeExecutor<"matmul"> = async (
  tensor,
  node,
  evaluate,
) => {
  const [a, b] = await Promise.all([evaluate(node.a), evaluate(node.b)]);
  const output = createOutput(tensor);
  const [rows, columns] = tensor.shape as readonly [number, number];
  await dispatchCpuKernel(
    matmulKernel,
    {
      shape: {
        rows,
        inner: node.inner,
        columns,
        transposeA: node.transposeA ? 1 : 0,
        transposeB: node.transposeB ? 1 : 0,
      },
      a,
      b,
      output,
    },
    rows * columns,
  );
  return output;
};
