import type { Tensor } from "../../tensor/tensor.js";

export function relu(input: Tensor): Tensor {
  return input.relu();
}
export function relu6(input: Tensor): Tensor {
  return input.relu6();
}
export function softplus(input: Tensor): Tensor {
  return input.softplus();
}
export function add(a: Tensor, b: Tensor): Tensor {
  return a.add(b);
}

export function addN(inputs: readonly Tensor[]): Tensor {
  if (inputs.length === 0)
    throw new RangeError("addN requires at least one tensor.");
  const first = inputs[0]!;
  for (const input of inputs) {
    input.assertUsable();
    if (input.device !== first.device)
      throw new TypeError("addN requires tensors from the same TypeNN device.");
    if (
      input.shape.length !== first.shape.length ||
      input.shape.some((dimension, index) => dimension !== first.shape[index])
    )
      throw new RangeError(
        "addN requires every input tensor to have the same shape.",
      );
  }
  return inputs.slice(1).reduce((total, input) => total.add(input), first);
}

export function sub(a: Tensor, b: Tensor): Tensor {
  return a.sub(b);
}
export function mul(a: Tensor, b: Tensor): Tensor {
  return a.mul(b);
}
export function div(a: Tensor, b: Tensor): Tensor {
  return a.div(b);
}
export function maximum(a: Tensor, b: Tensor): Tensor {
  return a.maximum(b);
}
export function minimum(a: Tensor, b: Tensor): Tensor {
  return a.minimum(b);
}
export function pow(a: Tensor, b: Tensor): Tensor {
  return a.pow(b);
}
export function squaredDifference(a: Tensor, b: Tensor): Tensor {
  return a.sub(b).square();
}
export function prelu(input: Tensor, alpha: Tensor): Tensor {
  return input.prelu(alpha);
}
export function neg(input: Tensor): Tensor {
  return input.neg();
}
export function abs(input: Tensor): Tensor {
  return input.abs();
}
export function square(input: Tensor): Tensor {
  return input.square();
}
export function sqrt(input: Tensor): Tensor {
  return input.sqrt();
}
export function exp(input: Tensor): Tensor {
  return input.exp();
}
export function log(input: Tensor): Tensor {
  return input.log();
}
export function sigmoid(input: Tensor): Tensor {
  return input.sigmoid();
}
export function tanh(input: Tensor): Tensor {
  return input.tanh();
}
export function acos(input: Tensor): Tensor {
  return input.acos();
}
export function acosh(input: Tensor): Tensor {
  return input.acosh();
}
export function asin(input: Tensor): Tensor {
  return input.asin();
}
export function asinh(input: Tensor): Tensor {
  return input.asinh();
}
export function atan(input: Tensor): Tensor {
  return input.atan();
}
export function atanh(input: Tensor): Tensor {
  return input.atanh();
}
export function step(input: Tensor): Tensor {
  return input.step();
}
export function floor(input: Tensor): Tensor {
  return input.floor();
}
export function ceil(input: Tensor): Tensor {
  return input.ceil();
}
export function round(input: Tensor): Tensor {
  return input.round();
}
export function sign(input: Tensor): Tensor {
  return input.sign();
}
export function reciprocal(input: Tensor): Tensor {
  return input.reciprocal();
}
export function rsqrt(input: Tensor): Tensor {
  return input.rsqrt();
}
export function sin(input: Tensor): Tensor {
  return input.sin();
}
export function cos(input: Tensor): Tensor {
  return input.cos();
}
export function tan(input: Tensor): Tensor {
  return input.tan();
}
export function expm1(input: Tensor): Tensor {
  return input.expm1();
}
export function log1p(input: Tensor): Tensor {
  return input.log1p();
}
export function sinh(input: Tensor): Tensor {
  return input.sinh();
}
export function cosh(input: Tensor): Tensor {
  return input.cosh();
}
export function gelu(input: Tensor): Tensor {
  return input.gelu();
}
export function leakyRelu(input: Tensor, alpha = 0.2): Tensor {
  return input.leakyRelu(alpha);
}
export function elu(input: Tensor, alpha = 1): Tensor {
  return input.elu(alpha);
}
export function clipByValue(
  input: Tensor,
  minimum: number,
  maximum: number,
): Tensor {
  return input.clipByValue(minimum, maximum);
}
