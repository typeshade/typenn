import { sameShape, sizeOf } from "./tensor/shape.js";
import { Tensor } from "./tensor/tensor.js";
import { constantLike } from "./autograd/accumulate.js";
import { topologicalOrder } from "./autograd/graph.js";
import { applyGradientRule } from "./autograd/rules/index.js";

/**
 * Build deferred reverse-mode gradients for the supported TypeNN Tensor graph
 * operations. Non-scalar outputs require an explicit upstream gradient.
 */
export function gradients(
  output: Tensor,
  wrt: readonly Tensor[],
  upstream?: Tensor,
): readonly Tensor[] {
  output.assertUsable();
  if (wrt.length === 0) return [];

  for (const tensor of wrt) {
    tensor.assertUsable();
    if (tensor.device !== output.device)
      throw new TypeError(
        "gradients requires output and requested tensors to share a TypeNN device.",
      );
  }

  const seed = makeUpstream(output, upstream);
  const accumulated = new Map<Tensor, Tensor>([[output, seed]]);
  for (const value of topologicalOrder(output)) {
    const gradient = accumulated.get(value);
    if (!gradient) continue;

    const node = value.node;
    if (node.kind === "input") continue;
    applyGradientRule({
      value,
      node,
      upstream: gradient,
      gradients: accumulated,
    });
  }

  return wrt.map(
    (tensor) => accumulated.get(tensor) ?? constantLike(tensor, 0),
  );
}

function makeUpstream(output: Tensor, upstream?: Tensor): Tensor {
  if (upstream) {
    upstream.assertUsable();
    if (upstream.device !== output.device)
      throw new TypeError("The upstream gradient must use the output device.");
    if (!sameShape(upstream.shape, output.shape))
      throw new RangeError("The upstream gradient must have the output shape.");
    return upstream;
  }

  if (sizeOf(output.shape) !== 1)
    throw new RangeError(
      "gradients requires an upstream tensor when output is not scalar.",
    );
  return output.device.tensor([1], { shape: output.shape });
}
