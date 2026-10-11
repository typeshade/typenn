import { addGradient } from "../accumulate.js";
import type { GradientRule } from "./types.js";

export const reshapeGradient: GradientRule<"reshape"> = ({
  node,
  upstream,
  gradients,
}) => {
  addGradient(gradients, node.input, upstream.reshape(node.input.shape));
};
