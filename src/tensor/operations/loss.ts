import {
  crossEntropyLabelsGrad,
  crossEntropyLogitsGrad,
  softmaxCrossEntropy,
  sparseCrossEntropyGrad,
  sparseSoftmaxCrossEntropy,
} from "../../kernels/loss.shade.ts";
import { sameShape } from "../shape.js";
import type { Tensor } from "../tensor.js";
import { assertCompatible, operation } from "./common.js";

export function softmaxCrossEntropyWithLogits(
  logits: Tensor,
  labels: Tensor,
  options: { readonly reduction?: "none" | "sum" | "mean" } = {},
): Tensor {
  assertCompatible(logits, labels);
  if (logits.shape.length !== 2 || !sameShape(logits.shape, labels.shape))
    throw new RangeError(
      "softmaxCrossEntropyWithLogits requires equal rank-2 logits and label tensors.",
    );
  const reduction = options.reduction ?? "none";
  if (reduction !== "none" && reduction !== "sum" && reduction !== "mean")
    throw new TypeError(
      `Unknown softmaxCrossEntropyWithLogits reduction: ${String(reduction)}.`,
    );
  const [rows, columns] = logits.shape as readonly [number, number];
  const perExample = operation(
    [rows],
    "softmaxCrossEntropy",
    softmaxCrossEntropy,
    [logits, labels],
    [columns],
    (upstream) => [
      operation(
        logits.shape,
        "softmaxCrossEntropyGrad",
        crossEntropyLogitsGrad,
        [logits, labels, upstream],
        [columns],
      ),
      operation(
        labels.shape,
        "softmaxCrossEntropyGrad",
        crossEntropyLabelsGrad,
        [logits, upstream],
        [columns],
      ),
    ],
  );
  if (reduction === "none") return perExample;
  return reduction === "sum" ? perExample.sum() : perExample.mean();
}

/**
 * Softmax cross entropy of rank-2 logits and one f32 class id per row (`classes`, shape
 * `[rows]`). The gradient flows to the logits only. The row values equal those of the one-hot
 * form for the same classes.
 */
export function sparseSoftmaxCrossEntropyWithLogits(
  logits: Tensor,
  classes: Tensor,
  options: { readonly reduction?: "none" | "sum" | "mean" } = {},
): Tensor {
  assertCompatible(logits, classes);
  if (
    logits.shape.length !== 2 ||
    classes.shape.length !== 1 ||
    classes.shape[0] !== logits.shape[0]
  )
    throw new RangeError(
      "Class-id cross entropy requires rank-2 logits and one class id per row.",
    );
  const reduction = options.reduction ?? "none";
  if (reduction !== "none" && reduction !== "sum" && reduction !== "mean")
    throw new TypeError(
      `Unknown softmaxCrossEntropyWithLogits reduction: ${String(reduction)}.`,
    );
  const [rows, columns] = logits.shape as readonly [number, number];
  const perExample = operation(
    [rows],
    "sparseSoftmaxCrossEntropy",
    sparseSoftmaxCrossEntropy,
    [logits, classes],
    [columns],
    (upstream) => [
      operation(
        logits.shape,
        "sparseSoftmaxCrossEntropyGrad",
        sparseCrossEntropyGrad,
        [logits, classes, upstream],
        [columns],
      ),
      undefined,
    ],
  );
  if (reduction === "none") return perExample;
  return reduction === "sum" ? perExample.sum() : perExample.mean();
}
