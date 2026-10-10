import { addGradient } from "../accumulate.js";
import type { GradientRule } from "./types.js";

export const softmaxCrossEntropyGradient: GradientRule<
  "softmaxCrossEntropy"
> = ({ value, node, upstream, gradients }) => {
  const common = {
    logits: node.logits,
    labels: node.labels,
    dy: upstream,
    rows: node.rows,
    columns: node.columns,
  };
  addGradient(
    gradients,
    node.logits,
    value.device.createOperation(node.logits.shape, {
      kind: "softmaxCrossEntropyGrad",
      op: "crossEntropyLogitsGrad",
      ...common,
    }),
  );
  addGradient(
    gradients,
    node.labels,
    value.device.createOperation(node.labels.shape, {
      kind: "softmaxCrossEntropyGrad",
      op: "crossEntropyLabelsGrad",
      ...common,
    }),
  );
};
