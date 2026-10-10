import {
  Tensor,
  type Conv2DOptions,
  type Pool2DOptions,
} from "../tensor/tensor.js";

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
      input.shape.some((d, i) => d !== first.shape[i])
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

export function matMul(
  a: Tensor,
  b: Tensor,
  options: {
    readonly transposeA?: boolean;
    readonly transposeB?: boolean;
  } = {},
): Tensor {
  return a.matmul(b, options);
}

export function conv2d(
  input: Tensor,
  filter: Tensor,
  options: Conv2DOptions = {},
): Tensor {
  return input.conv2d(filter, options);
}

export function conv2dGradients(
  input: Tensor,
  filter: Tensor,
  outputGradient: Tensor,
  options: Conv2DOptions = {},
): readonly [inputGradient: Tensor, filterGradient: Tensor] {
  return input.conv2dGradients(filter, outputGradient, options);
}

export function maxPool(input: Tensor, options: Pool2DOptions): Tensor {
  return input.maxPool(options);
}
export function avgPool(input: Tensor, options: Pool2DOptions): Tensor {
  return input.avgPool(options);
}
export function maxPoolGrad(
  input: Tensor,
  outputGradient: Tensor,
  options: Pool2DOptions,
): Tensor {
  return input.maxPoolGrad(outputGradient, options);
}
export function avgPoolGrad(
  input: Tensor,
  outputGradient: Tensor,
  options: Pool2DOptions,
): Tensor {
  return input.avgPoolGrad(outputGradient, options);
}

export function sum(input: Tensor, axis?: number): Tensor {
  return input.sum(axis);
}

export function mean(input: Tensor, axis?: number): Tensor {
  return input.mean(axis);
}

export function prod(input: Tensor, axis?: number): Tensor {
  return input.prod(axis);
}

export function max(input: Tensor, axis?: number): Tensor {
  return input.max(axis);
}
export function min(input: Tensor, axis?: number): Tensor {
  return input.min(axis);
}

export function argMax(input: Tensor, axis = 0): Tensor {
  return input.argMax(axis);
}

export function argMin(input: Tensor, axis = 0): Tensor {
  return input.argMin(axis);
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

export function softmax(input: Tensor, axis = -1): Tensor {
  return input.softmax(axis);
}

export function logSoftmax(input: Tensor, axis = -1): Tensor {
  return input.logSoftmax(axis);
}

export function softmaxCrossEntropyWithLogits(
  logits: Tensor,
  labels: Tensor,
  options: { readonly reduction?: "none" | "sum" | "mean" } = {},
): Tensor {
  return logits.softmaxCrossEntropyWithLogits(labels, options);
}
