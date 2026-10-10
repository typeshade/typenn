import type { TensorNode } from "../tensor/node.js";
import type { Tensor } from "../tensor/tensor.js";

type NodeKind = TensorNode["kind"];
type Node<K extends NodeKind> = Extract<TensorNode, { kind: K }>;
type ParentSelectorMap = {
  [K in NodeKind]: (node: Node<K>) => readonly Tensor[];
};

const NODE_PARENTS: ParentSelectorMap = {
  input: () => [],
  relu: ({ input }) => [input],
  reluGrad: ({ input, dy }) => [input, dy],
  unary: ({ input }) => [input],
  special: ({ input }) => [input],
  binary: ({ a, b }) => [a, b],
  binaryGrad: ({ a, b, dy }) => [a, b, dy],
  softmax: ({ input }) => [input],
  softmaxGrad: ({ value, dy }) => [value, dy],
  matmul: ({ a, b }) => [a, b],
  conv2d: ({ input, filter }) => [input, filter],
  pool2d: ({ input }) => [input],
  conv2dGrad: ({ input, filter, dy }) => [input, filter, dy],
  pool2dGrad: ({ input, dy }) => [input, dy],
  reduceGrad: ({ input, dy }) => [input, dy],
  softmaxCrossEntropyGrad: ({ logits, labels, dy }) => [logits, labels, dy],
  reduce: ({ input }) => [input],
  softmaxCrossEntropy: ({ logits, labels }) => [logits, labels],
};

function parents(node: TensorNode): readonly Tensor[] {
  const selector = NODE_PARENTS[node.kind] as (
    node: TensorNode,
  ) => readonly Tensor[];
  return selector(node);
}

export function topologicalOrder(output: Tensor): Tensor[] {
  const order: Tensor[] = [];
  const visited = new Set<Tensor>();

  function visit(tensor: Tensor): void {
    if (visited.has(tensor)) return;
    visited.add(tensor);
    for (const parent of parents(tensor.node)) visit(parent);
    order.push(tensor);
  }

  visit(output);
  return order.reverse();
}
