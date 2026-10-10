import { sizeOf } from "../../tensor/shape.js";
import type { GpuNodeExecutor } from "./types.js";

export const executeMatmul: GpuNodeExecutor<"matmul"> = async ({
  tensor,
  node,
  output,
  schedule,
}) => {
  const [a, b] = await Promise.all([schedule(node.a), schedule(node.b)]);
  const [rows, columns] = tensor.shape as readonly [number, number];
  return {
    operation: "matmul",
    bindings: {
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
    workgroups: Math.ceil((rows * columns) / 64),
  };
};
