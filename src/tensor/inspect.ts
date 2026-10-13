import { postOrder } from "../autograd/graph.js";
import type { Shape } from "./shape.js";
import type { Tensor } from "./tensor.js";

/** One tensor of a graph, as `graph()` reports it. */
export interface GraphNode {
  /** Stable for the life of the Tensor object. */
  readonly id: number;
  readonly kind: "input" | "op";
  /** The operation name, or `"input"` for an input tensor. */
  readonly op: string;
  readonly shape: readonly number[];
  /** Ids of the input tensors, in order. Empty for an input tensor. */
  readonly inputs: readonly number[];
  /** True for a trainable parameter (`requiresGrad`). */
  readonly parameter: boolean;
  /**
   * True when the tensor already holds an array. An input tensor always does, so `computed` is
   * true for every input Tensor. An operation holds one after a plan computed it, so a later
   * `data()` does not run it.
   */
  readonly computed: boolean;
}

/** The graph under a tensor. `nodes` is in topological order: inputs before their users. */
export interface GraphSnapshot {
  readonly output: number;
  readonly nodes: readonly GraphNode[];
}

const ids = new WeakMap<Tensor, number>();
let nextId = 0;

/** @internal The graph id of a tensor. The first call assigns it. */
export function graphId(tensor: Tensor): number {
  let id = ids.get(tensor);
  if (id === undefined) {
    id = nextId++;
    ids.set(tensor, id);
  }
  return id;
}

/** The graph under `tensor`: each reachable tensor once, inputs before the tensors that use them. */
export function graph(tensor: Tensor): GraphSnapshot {
  const nodes = postOrder(tensor).map((value): GraphNode => ({
    id: graphId(value),
    kind: value.node.kind,
    op: value.node.kind === "input" ? "input" : value.node.op,
    shape: value.shape.slice() as Shape,
    inputs: value.node.kind === "op" ? value.node.inputs.map(graphId) : [],
    parameter: value.requiresGrad,
    computed: value.node.kind === "input" || value.storage !== undefined,
  }));
  return { output: graphId(tensor), nodes };
}
