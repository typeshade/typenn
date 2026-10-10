import { addGradient } from "../accumulate.js";
import type { Tensor } from "../../tensor/tensor.js";
import type { GradientRule } from "./types.js";

function paddingMode(
  input: Tensor,
  outputHeight: number,
  outputWidth: number,
  strideY: number,
  strideX: number,
): "same" | "valid" {
  const height = input.shape[1]!;
  const width = input.shape[2]!;
  return outputHeight === Math.ceil(height / strideY) &&
    outputWidth === Math.ceil(width / strideX)
    ? "same"
    : "valid";
}

export const conv2dGradient: GradientRule<"conv2d"> = ({
  node,
  upstream,
  gradients,
}) => {
  const padding = paddingMode(
    node.input,
    node.outputHeight,
    node.outputWidth,
    node.strideY,
    node.strideX,
  );
  const [inputGradient, filterGradient] = node.input.conv2dGradients(
    node.filter,
    upstream,
    { strides: [node.strideY, node.strideX], padding },
  );
  addGradient(gradients, node.input, inputGradient);
  addGradient(gradients, node.filter, filterGradient);
};

export const pool2dGradient: GradientRule<"pool2d"> = ({
  node,
  upstream,
  gradients,
}) => {
  const options = {
    filterSize: [node.filterHeight, node.filterWidth] as const,
    strides: [node.strideY, node.strideX] as const,
    padding: paddingMode(
      node.input,
      node.outputHeight,
      node.outputWidth,
      node.strideY,
      node.strideX,
    ),
  };
  const inputGradient =
    node.op === "maxPool"
      ? node.input.maxPoolGrad(upstream, options)
      : node.input.avgPoolGrad(upstream, options);
  addGradient(gradients, node.input, inputGradient);
};
