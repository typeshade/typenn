import { sizeOf } from "../../tensor/shape.js";
import type { GpuNodeExecutor } from "./types.js";

export const executeReshape: GpuNodeExecutor<"reshape"> = async ({
  tensor,
  node,
  output,
  schedule,
}) => ({
  operation: "reshape",
  bindings: {
    shape: { length: sizeOf(tensor.shape) },
    input: await schedule(node.input),
    output,
  },
  workgroups: Math.ceil(sizeOf(tensor.shape) / 64),
});
