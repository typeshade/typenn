import type { FuseReport, FuseScope } from "typeshade/fuse";
import { type Resident } from "typeshade/runtime";
import { postOrder } from "../autograd/graph.js";
import { copy } from "../kernels/map.shade.ts";
import type { OpNode } from "../tensor/node.js";
import { sizeOf } from "../tensor/shape.js";
import { graphId } from "../tensor/inspect.js";
import type { Tensor } from "../tensor/tensor.js";

/** What the executor asks of the device that owns the arrays. */
export interface Memory {
  /** A resident array that holds a copy of `values`, owned by the device until it is recycled. */
  keep(values: Float32Array): Resident<Float32Array>;
  /**
   * A resident array of `length` values for a kernel to write. Its contents are whatever it held
   * before: a kernel call writes every element of its output, so the device need not zero it.
   */
  output(length: number): Resident<Float32Array>;
  /** Free an array made by `keep` or `output` now. */
  drop(array: Resident<Float32Array>): void;
  /** Give back an array no Tensor holds. The device reuses it, or frees it. */
  recycle(array: Resident<Float32Array>): void;
  /** Keep the report of a plan, and the graph id of each of its kernel calls, in call order. */
  record(report: FuseReport, nodes: readonly number[]): void;
  /** Run `body` as one plan on the device that owns the arrays (`device.fuse()`, change 0071). */
  fuse(body: (scope: FuseScope) => void): Promise<FuseReport>;
}

/**
 * Compute tensors and read them back. The operations under them that have no array yet are the
 * calls of one `fuse` scope, in the order of the graph, and the compiler runs the scope as one
 * plan on the tier `configure({ prefer })` selects: on WebGPU one submission, with the calls the
 * compiler's facts allow in one dispatch. A tensor that already has an array is a leaf of the
 * scope. Every operation keeps its array, so a later read, and the gradient graph, find it.
 *
 * `reads` are the tensors whose values are returned. They are computed in the same plan as
 * `roots`, so a read adds no plan when the plan computes it anyway. By default they are `roots`.
 */
export async function execute(
  roots: readonly Tensor[],
  memory: Memory,
  reads: readonly Tensor[] = roots,
): Promise<Float32Array[]> {
  await compute(reads === roots ? roots : [...roots, ...reads], memory);
  return Promise.all(reads.map(readValues));
}

async function readValues(tensor: Tensor): Promise<Float32Array> {
  // A tensor the caller made holds its values on the host; a device update may have left
  // a newer value in its array, which one readback brings to the host first.
  return tensor.node.kind === "input"
    ? (await tensor.hostValues()).slice()
    : tensor.storage!.read();
}

/** Compute the tensors under `roots` that have no array, as one plan. Nothing is read back. */
export async function compute(
  roots: readonly Tensor[],
  memory: Memory,
): Promise<void> {
  const pending = postOrder(
    roots,
    (tensor) => tensor.storage !== undefined || tensor.node.kind === "input",
  );
  if (pending.length > 0) await plan(pending, memory);
}

async function plan(pending: readonly Tensor[], memory: Memory): Promise<void> {
  // The arrays are made before the body runs, and the body is synchronous.
  for (const tensor of pending) {
    for (const input of (tensor.node as OpNode).inputs)
      if (input.storage === undefined && input.node.kind === "input")
        input.storage = memory.keep(input.node.value);
    tensor.storage = memory.output(sizeOf(tensor.shape));
  }
  // The graph id of each kernel call of the scope, in call order. A scratch copy takes the id of
  // the operation that reads it.
  const nodes: number[] = [];
  try {
    memory.record(
      await memory.fuse((scope) => {
        for (const tensor of pending) {
          const node = tensor.node as OpNode;
          const id = graphId(tensor);
          const inputs = distinct(scope, node.inputs, () => nodes.push(id));
          nodes.push(id);
          node.kernel(
            tensor.storage,
            ...inputs,
            ...(node.arrays ?? []),
            ...node.values,
          );
        }
      }),
      nodes,
    );
  } catch (error) {
    // The arrays hold nothing a later read can use: the next `data()` computes them again.
    for (const tensor of pending) {
      memory.drop(tensor.storage!);
      tensor.storage = undefined;
    }
    throw error;
  }
}

/**
 * The array of each input. One array must not be two parameters of one call (`a.add(a)`), so a
 * repeated input is read from a copy. The copy is a scratch array of the scope, and joins the
 * dispatch of the call that reads it.
 */
function distinct(
  scope: FuseScope,
  inputs: readonly Tensor[],
  onCopy: () => void,
): Resident<Float32Array>[] {
  const seen = new Set<Resident<Float32Array>>();
  return inputs.map((input) => {
    const array = input.storage!;
    if (!seen.has(array)) {
      seen.add(array);
      return array;
    }
    const twin = scope.scratch(sizeOf(input.shape));
    onCopy();
    copy(twin, array);
    return twin;
  });
}
