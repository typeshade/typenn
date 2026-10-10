import { sameShape } from "../shape.js";
import type { Tensor } from "../tensor.js";
import type { Conv2DOptions } from "../spatial.js";
import { spatialWindow } from "../spatial.js";
import { assertCompatible } from "./common.js";

export interface MatmulOptions {
  readonly transposeA?: boolean;
  readonly transposeB?: boolean;
}

export function matmul(
  a: Tensor,
  b: Tensor,
  options: MatmulOptions = {},
): Tensor {
  assertCompatible(a, b);
  if (a.shape.length !== 2 || b.shape.length !== 2)
    throw new RangeError("matmul currently supports rank-2 tensors only.");

  const transposeA = options.transposeA ?? false;
  const transposeB = options.transposeB ?? false;
  const rows = transposeA ? a.shape[1]! : a.shape[0]!;
  const inner = transposeA ? a.shape[0]! : a.shape[1]!;
  const otherInner = transposeB ? b.shape[1]! : b.shape[0]!;
  const columns = transposeB ? b.shape[0]! : b.shape[1]!;
  if (inner !== otherInner)
    throw new RangeError(
      `matmul inner dimensions must match; got ${inner} and ${otherInner}.`,
    );

  return a.device.createOperation([rows, columns], {
    kind: "matmul",
    a,
    b,
    inner,
    transposeA,
    transposeB,
  });
}

interface Conv2dMetadata {
  readonly input: Tensor;
  readonly filter: Tensor;
  readonly inputChannels: number;
  readonly kernelHeight: number;
  readonly kernelWidth: number;
  readonly outputChannels: number;
  readonly outputHeight: number;
  readonly outputWidth: number;
  readonly strideY: number;
  readonly strideX: number;
  readonly padTop: number;
  readonly padLeft: number;
}

function conv2dMetadata(
  input: Tensor,
  filter: Tensor,
  options: Conv2DOptions,
): Conv2dMetadata & { readonly batch: number } {
  if (input.shape.length !== 4 || filter.shape.length !== 4)
    throw new RangeError(
      "conv2d expects NHWC input [batch, height, width, channels] and HWIO filter.",
    );
  const [batch, height, width, inputChannels] = input.shape as readonly [
    number,
    number,
    number,
    number,
  ];
  const [kernelHeight, kernelWidth, filterChannels, outputChannels] =
    filter.shape as readonly [number, number, number, number];
  if (inputChannels !== filterChannels)
    throw new RangeError(
      `conv2d input channels (${inputChannels}) must match filter channels (${filterChannels}).`,
    );
  const window = spatialWindow(
    height,
    width,
    kernelHeight,
    kernelWidth,
    options,
    "conv2d",
  );
  return {
    input,
    filter,
    batch,
    inputChannels,
    kernelHeight,
    kernelWidth,
    outputChannels,
    outputHeight: window.outputHeight,
    outputWidth: window.outputWidth,
    strideY: window.strideY,
    strideX: window.strideX,
    padTop: window.padTop,
    padLeft: window.padLeft,
  };
}

export function conv2d(
  input: Tensor,
  filter: Tensor,
  options: Conv2DOptions = {},
): Tensor {
  assertCompatible(input, filter);
  const metadata = conv2dMetadata(input, filter, options);
  const { batch, ...nodeMetadata } = metadata;
  return input.device.createOperation(
    [
      batch,
      metadata.outputHeight,
      metadata.outputWidth,
      metadata.outputChannels,
    ],
    { kind: "conv2d", ...nodeMetadata },
  );
}

export function conv2dGradients(
  input: Tensor,
  filter: Tensor,
  dy: Tensor,
  options: Conv2DOptions = {},
): readonly [Tensor, Tensor] {
  assertCompatible(input, filter);
  assertCompatible(input, dy);
  const metadata = conv2dMetadata(input, filter, options);
  const expectedShape = [
    metadata.batch,
    metadata.outputHeight,
    metadata.outputWidth,
    metadata.outputChannels,
  ];
  if (!sameShape(expectedShape, dy.shape))
    throw new RangeError(
      `conv2d output gradient shape must be [${expectedShape.join(", ")}].`,
    );

  const { batch: _batch, ...nodeMetadata } = metadata;
  return [
    input.device.createOperation(input.shape, {
      kind: "conv2dGrad",
      op: "conv2dInputGrad",
      ...nodeMetadata,
      dy,
    }),
    input.device.createOperation(filter.shape, {
      kind: "conv2dGrad",
      op: "conv2dFilterGrad",
      ...nodeMetadata,
      dy,
    }),
  ];
}
