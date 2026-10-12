import type { Backward, Kernel, KernelValue } from "../node.js";
import type { Shape } from "../shape.js";
import type { Tensor } from "../tensor.js";

export function assertCompatible(left: Tensor, right: Tensor): void {
  left.assertUsable();
  right.assertUsable();
  if (left.device !== right.device)
    throw new TypeError(
      "Tensor operations require tensors from the same TypeNN device.",
    );
}

/**
 * A tensor that `kernel(out, ...inputs, ...values)` computes. The graph keeps the call and, if
 * the operation has a gradient rule, `backward`. Nothing runs until `data()` reads a result.
 */
export function operation(
  shape: Shape,
  op: string,
  kernel: Kernel,
  inputs: readonly Tensor[],
  values: readonly KernelValue[] = [],
  backward?: Backward,
): Tensor {
  return inputs[0]!.device.createOperation(shape, {
    kind: "op",
    op,
    kernel,
    inputs,
    values,
    ...(backward ? { backward } : {}),
  });
}
