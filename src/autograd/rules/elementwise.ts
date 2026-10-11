import { addGradient, constantLike, scalarLike } from "../accumulate.js";
import type { GradientRule } from "./types.js";
import type { UnaryOperation } from "../../tensor/nodes/unary.js";
import type { Tensor } from "../../tensor/tensor.js";

type UnaryGradient = (
  input: Tensor,
  output: Tensor,
  upstream: Tensor,
) => Tensor;

const UNARY_GRADIENTS: Record<UnaryOperation, UnaryGradient> = {
  acos: (input, _output, upstream) =>
    upstream.div(scalarLike(input, 1).sub(input.square()).sqrt().neg()),
  acosh: (input, _output, upstream) =>
    upstream.div(input.square().sub(scalarLike(input, 1)).sqrt()),
  asin: (input, _output, upstream) =>
    upstream.div(scalarLike(input, 1).sub(input.square()).sqrt()),
  asinh: (input, _output, upstream) =>
    upstream.div(input.square().add(scalarLike(input, 1)).sqrt()),
  atan: (input, _output, upstream) =>
    upstream.div(input.square().add(scalarLike(input, 1))),
  atanh: (input, _output, upstream) =>
    upstream.div(scalarLike(input, 1).sub(input.square())),
  step: (input) => constantLike(input, 0),
  neg: (_input, _output, upstream) => upstream.neg(),
  abs: (input, _output, upstream) => upstream.mul(input.sign()),
  square: (input, _output, upstream) =>
    upstream.mul(input).mul(scalarLike(input, 2)),
  sqrt: (_input, output, upstream) =>
    upstream.div(output.mul(scalarLike(output, 2))),
  exp: (_input, output, upstream) => upstream.mul(output),
  log: (input, _output, upstream) => upstream.div(input),
  reciprocal: (_input, output, upstream) => upstream.mul(output.square()).neg(),
  rsqrt: (_input, output, upstream) =>
    upstream.mul(output.square()).mul(output).mul(scalarLike(output, -0.5)),
  sin: (input, _output, upstream) => upstream.mul(input.cos()),
  cos: (input, _output, upstream) => upstream.mul(input.sin()).neg(),
  tan: (_input, output, upstream) =>
    upstream.mul(scalarLike(output, 1).add(output.square())),
  expm1: (input, _output, upstream) => upstream.mul(input.exp()),
  log1p: (input, _output, upstream) =>
    upstream.div(scalarLike(input, 1).add(input)),
  sinh: (input, _output, upstream) => upstream.mul(input.cosh()),
  cosh: (input, _output, upstream) => upstream.mul(input.sinh()),
  sigmoid: (_input, output, upstream) =>
    upstream.mul(output).mul(scalarLike(output, 1).sub(output)),
  tanh: (_input, output, upstream) =>
    upstream.mul(scalarLike(output, 1).sub(output.square())),
  floor: (input) => constantLike(input, 0),
  ceil: (input) => constantLike(input, 0),
  round: (input) => constantLike(input, 0),
  sign: (input) => constantLike(input, 0),
};

function binaryGradient(
  output: Tensor,
  node: Parameters<GradientRule<"binary">>[0]["node"],
  upstream: Tensor,
  wrtA: boolean,
  target: Tensor,
): Tensor {
  return output.device.createOperation(target.shape, {
    kind: "binaryGrad",
    op: node.op,
    a: node.a,
    b: node.b,
    dy: upstream,
    dimensions: node.dimensions,
    aStrides: node.aStrides,
    bStrides: node.bStrides,
    wrtA,
  });
}

export const reluGradient: GradientRule<"relu"> = ({
  node,
  upstream,
  gradients,
}) => {
  addGradient(gradients, node.input, node.input.reluGrad(upstream));
};

export const unaryGradientRule: GradientRule<"unary"> = ({
  value,
  node,
  upstream,
  gradients,
}) => {
  addGradient(
    gradients,
    node.input,
    UNARY_GRADIENTS[node.op](node.input, value, upstream),
  );
};

export const binaryGradientRule: GradientRule<"binary"> = ({
  value,
  node,
  upstream,
  gradients,
}) => {
  addGradient(
    gradients,
    node.a,
    binaryGradient(value, node, upstream, true, node.a),
  );
  addGradient(
    gradients,
    node.b,
    binaryGradient(value, node, upstream, false, node.b),
  );
};

export const specialGradient: GradientRule<"special"> = ({ node }) => {
  throw new Error(`Autograd for activation ${node.op} is not implemented yet.`);
};
