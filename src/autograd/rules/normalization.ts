import { addGradient } from "../accumulate.js";
import type { GradientRule } from "./types.js";

export const softmaxGradient: GradientRule<"softmax"> = ({
  value,
  node,
  upstream,
  gradients,
}) => {
  addGradient(
    gradients,
    node.input,
    value.device.createOperation(node.input.shape, {
      kind: "softmaxGrad",
      op: node.op === "softmax" ? "softmaxGrad" : "logSoftmaxGrad",
      value,
      dy: upstream,
      outer: node.outer,
      axisSize: node.axisSize,
      inner: node.inner,
    }),
  );
};
