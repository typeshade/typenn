import type { BinaryOperation } from "../nodes/binary.js";
import type { SpecialOperation } from "../nodes/special.js";
import type { UnaryOperation } from "../nodes/unary.js";
import { broadcastLayout, sameShape } from "../shape.js";
import type { Tensor } from "../tensor.js";
import { assertCompatible } from "./common.js";

export function relu(input: Tensor): Tensor {
  input.assertUsable();
  return input.device.createOperation(input.shape, { kind: "relu", input });
}

export function reluGrad(input: Tensor, dy: Tensor): Tensor {
  assertCompatible(input, dy);
  if (!sameShape(input.shape, dy.shape))
    throw new RangeError(
      "reluGrad requires input and output gradient tensors to have equal shapes.",
    );
  return input.device.createOperation(input.shape, {
    kind: "reluGrad",
    input,
    dy,
  });
}

export function unary(input: Tensor, op: UnaryOperation): Tensor {
  input.assertUsable();
  return input.device.createOperation(input.shape, {
    kind: "unary",
    op,
    input,
  });
}

export function special(
  input: Tensor,
  op: SpecialOperation,
  alpha = 0,
  minimum = 0,
  maximum = 0,
): Tensor {
  input.assertUsable();
  return input.device.createOperation(input.shape, {
    kind: "special",
    op,
    input,
    alpha,
    minimum,
    maximum,
  });
}

export function leakyRelu(input: Tensor, alpha = 0.2): Tensor {
  if (!Number.isFinite(alpha) || alpha < 0)
    throw new RangeError("leakyRelu alpha must be finite and non-negative.");
  return special(input, "leakyRelu", alpha);
}

export function elu(input: Tensor, alpha = 1): Tensor {
  if (!Number.isFinite(alpha) || alpha < 0)
    throw new RangeError("elu alpha must be finite and non-negative.");
  return special(input, "elu", alpha);
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
  return special(input, "clipByValue", 0, minimum, maximum);
}

export function binary(
  left: Tensor,
  right: Tensor,
  op: BinaryOperation,
): Tensor {
  assertCompatible(left, right);
  const layout = broadcastLayout(left.shape, right.shape);
  return left.device.createOperation(layout.shape, {
    kind: "binary",
    op,
    a: left,
    b: right,
    dimensions: layout.dimensions,
    aStrides: layout.aStrides,
    bStrides: layout.bStrides,
  });
}
