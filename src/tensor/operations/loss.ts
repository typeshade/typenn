import {
  crossEntropyLabelsGrad,
  crossEntropyLogitsGrad,
  softmaxCrossEntropy,
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
