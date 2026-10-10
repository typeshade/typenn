import { addGradient } from "../accumulate.js";
import type { GradientRule } from "./types.js";

export const reductionGradient: GradientRule<"reduce"> = ({
  value,
  node,
  upstream,
  gradients,
}) => {
  if (node.op !== "sum" && node.op !== "mean")
    throw new Error(
      `Autograd for ${node.op} reductions is not implemented yet.`,
    );

  const inputGradient = value.device.createOperation(node.input.shape, {
    kind: "reduceGrad",
    input: node.input,
    dy: upstream,
    outer: node.outer,
    axisSize: node.axisSize,
    inner: node.inner,
    meanScale: node.op === "mean" ? 1 / node.axisSize : 1,
  });
  addGradient(gradients, node.input, inputGradient);
};
