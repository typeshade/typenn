import { fuse, type FuseReport, type FuseScope } from "typeshade/fuse";
import { type Resident } from "typeshade/runtime";
import { postOrder } from "../autograd/graph.js";
import { copy } from "../kernels/map.shade.ts";
import type { OpNode } from "../tensor/node.js";
import { sizeOf } from "../tensor/shape.js";
import type { Tensor } from "../tensor/tensor.js";

/** What the executor asks of the device that owns the arrays. */
export interface Memory {
  /** A resident array that holds a copy of `values`, freed with the device. */
  keep(values: Float32Array): Resident<Float32Array>;
  /** Free an array made by `keep`. */
  drop(array: Resident<Float32Array>): void;
  /** Keep the report of a plan. */
  record(report: FuseReport): void;
}

/**
 * Compute a tensor and read it back. The operations under it that have no array yet are the
 * calls of one `fuse` scope, in the order of the graph, and the compiler runs the scope as one
 * plan on the tier `configure({ prefer })` selects: on WebGPU one submission, with the calls the
 * compiler's facts allow in one dispatch. A tensor that already has an array is a leaf of the
 * scope. Every operation keeps its array, so a later read, and the gradient graph, find it.
 */
export async function execute(
  root: Tensor,
  memory: Memory,
): Promise<Float32Array> {
  // A tensor the caller made holds its values on the host.
  if (root.node.kind === "input") return root.node.value.slice();
  const pending = postOrder(
    root,
    (tensor) => tensor.storage !== undefined || tensor.node.kind === "input",
  );
  if (pending.length > 0) await plan(pending, memory);
  return root.storage!.read();
}

async function plan(pending: readonly Tensor[], memory: Memory): Promise<void> {
  // The arrays are made before the body runs, and the body is synchronous.
  for (const tensor of pending) {
    for (const input of (tensor.node as OpNode).inputs)
      if (input.storage === undefined && input.node.kind === "input")
        input.storage = memory.keep(input.node.value);
    tensor.storage = memory.keep(new Float32Array(sizeOf(tensor.shape)));
  }
  try {
    memory.record(
      await fuse((scope) => {
        for (const tensor of pending) {
          const node = tensor.node as OpNode;
          node.kernel(
            tensor.storage,
            ...distinct(scope, node.inputs),
            ...node.values,
          );
        }
      }),
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
): Resident<Float32Array>[] {
  const seen = new Set<Resident<Float32Array>>();
  return inputs.map((input) => {
    const array = input.storage!;
    if (!seen.has(array)) {
      seen.add(array);
      return array;
    }
    const twin = scope.scratch(sizeOf(input.shape));
    copy(twin, array);
    return twin;
  });
}
