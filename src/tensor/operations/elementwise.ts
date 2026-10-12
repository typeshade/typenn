import { constantLike, scalarLike } from "../../autograd/accumulate.js";
import {
  BinaryOperation,
  add,
  addExpand,
  addStrided,
  binaryGrad,
  div,
  divExpand,
  divStrided,
  maximum,
  maximumExpand,
  maximumStrided,
  minimum,
  minimumExpand,
  minimumStrided,
  mul,
  mulExpand,
  mulStrided,
  power,
  powerExpand,
  powerStrided,
  prelu,
  preluExpand,
  preluStrided,
  sub,
  subExpand,
  subStrided,
} from "../../kernels/binary.shade.ts";
import {
  absMap,
  acosMap,
  acoshMap,
  asinMap,
  asinhMap,
  atanMap,
  atanhMap,
  ceilMap,
  clipMap,
  cosMap,
  coshMap,
  eluMap,
  expMap,
  expm1Map,
  floorMap,
  geluMap,
  leakyReluMap,
  log1pMap,
  logMap,
  negMap,
  reciprocalMap,
  relu as reluKernel,
  relu6Map,
  reluGrad as reluGradKernel,
  roundMap,
  rsqrtMap,
  sigmoidMap,
  signMap,
  sinMap,
  sinhMap,
  softplusMap,
  sqrtMap,
  squareMap,
  stepMap,
  tanMap,
  tanhMap,
} from "../../kernels/map.shade.ts";
import type { Kernel, KernelValue } from "../node.js";
import { broadcastLayout, sameShape, type BroadcastLayout } from "../shape.js";
import type { Tensor } from "../tensor.js";
import { assertCompatible, operation } from "./common.js";

export function relu(input: Tensor): Tensor {
  input.assertUsable();
  return operation(input.shape, "relu", reluKernel, [input], [], (upstream) => [
    reluGrad(input, upstream),
  ]);
}

export function reluGrad(input: Tensor, dy: Tensor): Tensor {
  assertCompatible(input, dy);
  if (!sameShape(input.shape, dy.shape))
    throw new RangeError(
      "reluGrad requires input and output gradient tensors to have equal shapes.",
    );
  return operation(input.shape, "reluGrad", reluGradKernel, [input, dy]);
}

/** A unary operation: its kernel function, and its gradient given input, output and upstream. */
type UnaryRule = readonly [
  kernel: Kernel,
  gradient: (input: Tensor, output: Tensor, upstream: Tensor) => Tensor,
];

const zero = (input: Tensor): Tensor => constantLike(input, 0);

const UNARY = {
  acos: [
    acosMap,
    (input, _output, upstream) =>
      upstream.div(scalarLike(input, 1).sub(input.square()).sqrt().neg()),
  ],
  acosh: [
    acoshMap,
    (input, _output, upstream) =>
      upstream.div(input.square().sub(scalarLike(input, 1)).sqrt()),
  ],
  asin: [
    asinMap,
    (input, _output, upstream) =>
      upstream.div(scalarLike(input, 1).sub(input.square()).sqrt()),
  ],
  asinh: [
    asinhMap,
    (input, _output, upstream) =>
      upstream.div(input.square().add(scalarLike(input, 1)).sqrt()),
  ],
  atan: [
    atanMap,
    (input, _output, upstream) =>
      upstream.div(input.square().add(scalarLike(input, 1))),
  ],
  atanh: [
    atanhMap,
    (input, _output, upstream) =>
      upstream.div(scalarLike(input, 1).sub(input.square())),
  ],
  step: [stepMap, zero],
  neg: [negMap, (_input, _output, upstream) => upstream.neg()],
  abs: [absMap, (input, _output, upstream) => upstream.mul(input.sign())],
  square: [
    squareMap,
    (input, _output, upstream) => upstream.mul(input).mul(scalarLike(input, 2)),
  ],
  sqrt: [
    sqrtMap,
    (_input, output, upstream) =>
      upstream.div(output.mul(scalarLike(output, 2))),
  ],
  exp: [expMap, (_input, output, upstream) => upstream.mul(output)],
  log: [logMap, (input, _output, upstream) => upstream.div(input)],
  reciprocal: [
    reciprocalMap,
    (_input, output, upstream) => upstream.mul(output.square()).neg(),
  ],
  rsqrt: [
    rsqrtMap,
    (_input, output, upstream) =>
      upstream.mul(output.square()).mul(output).mul(scalarLike(output, -0.5)),
  ],
  sin: [sinMap, (input, _output, upstream) => upstream.mul(input.cos())],
  cos: [cosMap, (input, _output, upstream) => upstream.mul(input.sin()).neg()],
  tan: [
    tanMap,
    (_input, output, upstream) =>
      upstream.mul(scalarLike(output, 1).add(output.square())),
  ],
  expm1: [expm1Map, (input, _output, upstream) => upstream.mul(input.exp())],
  log1p: [
    log1pMap,
    (input, _output, upstream) => upstream.div(scalarLike(input, 1).add(input)),
  ],
  sinh: [sinhMap, (input, _output, upstream) => upstream.mul(input.cosh())],
  cosh: [coshMap, (input, _output, upstream) => upstream.mul(input.sinh())],
  sigmoid: [
    sigmoidMap,
    (_input, output, upstream) =>
      upstream.mul(output).mul(scalarLike(output, 1).sub(output)),
  ],
  tanh: [
    tanhMap,
    (_input, output, upstream) =>
      upstream.mul(scalarLike(output, 1).sub(output.square())),
  ],
  floor: [floorMap, zero],
  ceil: [ceilMap, zero],
  round: [roundMap, zero],
  sign: [signMap, zero],
} satisfies Record<string, UnaryRule>;

export type UnaryOperation = keyof typeof UNARY;

export function unary(input: Tensor, op: UnaryOperation): Tensor {
  input.assertUsable();
  const [kernel, gradient] = UNARY[op] as UnaryRule;
  return operation(input.shape, op, kernel, [input], [], (upstream, output) => [
    gradient(input, output, upstream),
  ]);
}

/** Activations with no scalar: `gelu`, `relu6` and `softplus`. */
const SPECIAL = {
  gelu: geluMap,
  relu6: relu6Map,
  softplus: softplusMap,
} satisfies Record<string, Kernel>;

export type SpecialOperation = keyof typeof SPECIAL;

/** The gradient rule of an activation TypeNN has a kernel function for and no rule yet. */
function noGradient(op: string) {
  return (): never => {
    throw new Error(`Autograd for activation ${op} is not implemented yet.`);
  };
}

export function special(input: Tensor, op: SpecialOperation): Tensor {
  input.assertUsable();
  return operation(input.shape, op, SPECIAL[op], [input], [], noGradient(op));
}

export function leakyRelu(input: Tensor, alpha = 0.2): Tensor {
  if (!Number.isFinite(alpha) || alpha < 0)
    throw new RangeError("leakyRelu alpha must be finite and non-negative.");
  input.assertUsable();
  return operation(
    input.shape,
    "leakyRelu",
    leakyReluMap,
    [input],
    [alpha],
    noGradient("leakyRelu"),
  );
}

export function elu(input: Tensor, alpha = 1): Tensor {
  if (!Number.isFinite(alpha) || alpha < 0)
    throw new RangeError("elu alpha must be finite and non-negative.");
  input.assertUsable();
  return operation(
    input.shape,
    "elu",
    eluMap,
    [input],
    [alpha],
    noGradient("elu"),
  );
}

export function clipByValue(
  input: Tensor,
  minimum: number,
  maximum: number,
): Tensor {
  if (
    !Number.isFinite(minimum) ||
    !Number.isFinite(maximum) ||
    minimum > maximum
  )
    throw new RangeError(
      "clipByValue requires finite bounds with minimum <= maximum.",
    );
  input.assertUsable();
  return operation(
    input.shape,
    "clipByValue",
    clipMap,
    [input],
    [minimum, maximum],
    noGradient("clipByValue"),
  );
}

/** A binary operation: its code for `binaryGrad`, and a kernel function for each way the operands
 *  meet the output (`src/kernels/binary.shade.ts`). */
interface BinaryRule {
  readonly code: BinaryOperation;
  readonly same: Kernel;
  readonly expand: Kernel;
  readonly strided: Kernel;
}

const BINARY = {
  add: {
    code: BinaryOperation.Add,
    same: add,
    expand: addExpand,
    strided: addStrided,
  },
  sub: {
    code: BinaryOperation.Subtract,
    same: sub,
    expand: subExpand,
    strided: subStrided,
  },
  mul: {
    code: BinaryOperation.Multiply,
    same: mul,
    expand: mulExpand,
    strided: mulStrided,
  },
  div: {
    code: BinaryOperation.Divide,
    same: div,
    expand: divExpand,
    strided: divStrided,
  },
  maximum: {
    code: BinaryOperation.Maximum,
    same: maximum,
    expand: maximumExpand,
    strided: maximumStrided,
  },
  minimum: {
    code: BinaryOperation.Minimum,
    same: minimum,
    expand: minimumExpand,
    strided: minimumStrided,
  },
  pow: {
    code: BinaryOperation.Power,
    same: power,
    expand: powerExpand,
    strided: powerStrided,
  },
  prelu: {
    code: BinaryOperation.PRelu,
    same: prelu,
    expand: preluExpand,
    strided: preluStrided,
  },
} satisfies Record<string, BinaryRule>;

export type BinaryName = keyof typeof BINARY;

export function binary(left: Tensor, right: Tensor, op: BinaryName): Tensor {
  assertCompatible(left, right);
  const layout = broadcastLayout(left.shape, right.shape);
  const rule: BinaryRule = BINARY[op];
  const rank = layout.shape.length;
  // The operand that has the shape of the output is read at the loop's index, which lets the call
  // share a dispatch with the one that wrote it.
  let kernel: Kernel = rule.same;
  let values: KernelValue[] = [];
  if (!sameShape(left.shape, right.shape)) {
    if (sameShape(left.shape, layout.shape)) {
      kernel = rule.expand;
      values = [rank, layout.dimensions, layout.bStrides];
    } else {
      kernel = rule.strided;
      values = [rank, layout.dimensions, layout.aStrides, layout.bStrides];
    }
  }
  return operation(
    layout.shape,
    op,
    kernel,
    [left, right],
    values,
    (upstream) => [
      binaryGradient(rule, left, right, upstream, layout, true),
      binaryGradient(rule, left, right, upstream, layout, false),
    ],
  );
}

/** The gradient of `a op b` with respect to `a` or to `b`, summed over the axes it was broadcast on. */
function binaryGradient(
  rule: BinaryRule,
  a: Tensor,
  b: Tensor,
  dy: Tensor,
  layout: BroadcastLayout,
  wrtA: boolean,
): Tensor {
  const target = wrtA ? a : b;
  const targetSize = target.shape.reduce((size, axis) => size * axis, 1);
  const outputSize = layout.shape.reduce((size, axis) => size * axis, 1);
  return operation(
    target.shape,
    "binaryGrad",
    binaryGrad,
    [a, b, dy],
    [
      layout.shape.length,
      layout.dimensions,
      layout.aStrides,
      layout.bStrides,
      outputSize / targetSize,
      rule.code,
      wrtA ? 1 : 0,
    ],
  );
}
