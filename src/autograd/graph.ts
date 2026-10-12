import type { Tensor } from "../tensor/tensor.js";

/** The tensors a tensor is computed from: the inputs of its kernel call. */
function parents(tensor: Tensor): readonly Tensor[] {
  return tensor.node.kind === "op" ? tensor.node.inputs : [];
}

/**
 * The tensors `output` is computed from, each after the tensors it reads and `output` last.
 * A tensor for which `stop` is true ends the walk: it is not in the list, and nor is anything
 * only it reads. The walk is iterative, so a deep graph does not exhaust the call stack.
 */
export function postOrder(
  output: Tensor,
  stop: (tensor: Tensor) => boolean = () => false,
): Tensor[] {
  const order: Tensor[] = [];
  const seen = new Map<Tensor, "open" | "done">();
  const stack: Tensor[] = [output];
  while (stack.length > 0) {
    const tensor = stack[stack.length - 1]!;
    const state = seen.get(tensor);
    if (state === "open") order.push(tensor);
    if (state !== undefined || stop(tensor)) {
      seen.set(tensor, "done");
      stack.pop();
      continue;
    }
    seen.set(tensor, "open");
    // Pushed last first, so that the first input is walked first.
    const inputs = parents(tensor);
    for (let index = inputs.length - 1; index >= 0; index--)
      if (!seen.has(inputs[index]!)) stack.push(inputs[index]!);
  }
  return order;
}

/** The tensors `output` is computed from, `output` first and each before the tensors it reads. */
export function topologicalOrder(output: Tensor): Tensor[] {
  return postOrder(output).reverse();
}
