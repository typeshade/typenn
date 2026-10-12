import {
  conv2d as conv2dKernel,
  conv2dFilterGrad,
  conv2dInputGrad,
} from "../../kernels/conv.shade.ts";
import { matmul as matmulKernel } from "../../kernels/matmul.shade.ts";
import { sameShape } from "../shape.js";
import type { Tensor } from "../tensor.js";
import { spatialWindow, type Conv2DOptions } from "../spatial.js";
import { assertCompatible, operation } from "./common.js";

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

  return operation(
    [rows, columns],
    "matmul",
    matmulKernel,
    [a, b],
    [rows, inner, columns, transposeA ? 1 : 0, transposeB ? 1 : 0],
    (upstream) => [
      transposeA
        ? b.matmul(upstream, { transposeA: transposeB, transposeB: true })
        : upstream.matmul(b, { transposeB: !transposeB }),
      transposeB
        ? upstream.matmul(a, { transposeA: true, transposeB: transposeA })
        : a.matmul(upstream, { transposeA: !transposeA }),
    ],
  );
}

interface Conv2dMetadata {
  readonly batch: number;
  readonly height: number;
  readonly width: number;
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
): Conv2dMetadata {
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
    batch,
    height,
    width,
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
  const m = conv2dMetadata(input, filter, options);
  return operation(
    [m.batch, m.outputHeight, m.outputWidth, m.outputChannels],
    "conv2d",
    conv2dKernel,
    [input, filter],
    [
      m.height,
      m.width,
      m.inputChannels,
      m.kernelHeight,
      m.kernelWidth,
      m.outputChannels,
      m.outputHeight,
      m.outputWidth,
      m.strideY,
      m.strideX,
      m.padTop,
      m.padLeft,
    ],
    (upstream) => conv2dGradients(input, filter, upstream, options),
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
  const m = conv2dMetadata(input, filter, options);
  const expectedShape = [
    m.batch,
    m.outputHeight,
    m.outputWidth,
    m.outputChannels,
  ];
  if (!sameShape(expectedShape, dy.shape))
    throw new RangeError(
      `conv2d output gradient shape must be [${expectedShape.join(", ")}].`,
    );
  const window = [m.strideY, m.strideX, m.padTop, m.padLeft];
  return [
    operation(
      input.shape,
      "conv2dGrad",
      conv2dInputGrad,
      [filter, dy],
      [
        m.height,
        m.width,
        m.inputChannels,
        m.kernelHeight,
        m.kernelWidth,
        m.outputChannels,
        m.outputHeight,
        m.outputWidth,
        ...window,
      ],
    ),
    operation(
      filter.shape,
      "conv2dGrad",
      conv2dFilterGrad,
      [input, dy],
      [
        m.batch,
        m.height,
        m.width,
        m.inputChannels,
        m.kernelWidth,
        m.outputChannels,
        m.outputHeight,
        m.outputWidth,
        ...window,
      ],
    ),
  ];
}
