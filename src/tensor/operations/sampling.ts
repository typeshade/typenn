import {
  gridSampleNHWC as gridSampleNHWCKernel,
  resizeBilinear as resizeBilinearKernel,
} from "../../kernels/grid-sample.shade.ts";
import { sizeOf, type Shape } from "../shape.js";
import type { Tensor } from "../tensor.js";
import { assertCompatible, operation } from "./common.js";

export interface SamplingOptions {
  /** Coordinate convention used by PyTorch's grid_sample and interpolate. Defaults to false. */
  readonly alignCorners?: boolean;
}

function alignCorners(options: SamplingOptions): number {
  if (
    options.alignCorners !== undefined &&
    typeof options.alignCorners !== "boolean"
  )
    throw new TypeError("alignCorners must be a boolean.");
  return options.alignCorners === true ? 1 : 0;
}

function noGradient(op: string) {
  return (): never => {
    throw new Error(`Autograd for ${op} is not implemented yet.`);
  };
}

function nhwcShape(
  shape: Shape,
  op: string,
): readonly [number, number, number, number] {
  if (shape.length !== 4)
    throw new RangeError(`${op} expects an NHWC rank-4 tensor.`);
  return shape as readonly [number, number, number, number];
}

/**
 * Bilinearly samples an NHWC image using a normalized `[B, outH, outW, 2]` x/y grid.
 * Finite-coordinate interpolation matches PyTorch; non-finite grid coordinates produce NaN in
 * TypeNN.
 * Out-of-bounds taps use zero padding, matching `torch.nn.functional.grid_sample`.
 * This inference operation has no gradient rule.
 */
export function gridSampleNHWC(
  input: Tensor,
  grid: Tensor,
  options: SamplingOptions = {},
): Tensor {
  assertCompatible(input, grid);
  const [batch, height, width, channels] = nhwcShape(
    input.shape,
    "gridSampleNHWC",
  );
  if (grid.shape.length !== 4 || grid.shape[0] !== batch || grid.shape[3] !== 2)
    throw new RangeError(
      `gridSampleNHWC grid must have shape [${batch}, outHeight, outWidth, 2].`,
    );
  const [, outHeight, outWidth] = grid.shape;
  const outputShape = [batch, outHeight!, outWidth!, channels];
  sizeOf(outputShape);
  return operation(
    outputShape,
    "gridSampleNHWC",
    gridSampleNHWCKernel,
    [input, grid],
    [height, width, channels, outHeight!, outWidth!, alignCorners(options)],
    noGradient("gridSampleNHWC"),
  );
}

/**
 * Bilinearly resizes an NHWC image. Edge samples clamp to the nearest edge value, matching
 * `torch.nn.functional.interpolate(mode="bilinear")`. This inference operation has no gradient.
 */
export function resizeBilinear(
  input: Tensor,
  outHeight: number,
  outWidth: number,
  options: SamplingOptions = {},
): Tensor {
  input.assertUsable();
  const [batch, height, width, channels] = nhwcShape(
    input.shape,
    "resizeBilinear",
  );
  if (!Number.isSafeInteger(outHeight) || outHeight < 1)
    throw new RangeError(
      "resizeBilinear output height must be a positive safe integer.",
    );
  if (!Number.isSafeInteger(outWidth) || outWidth < 1)
    throw new RangeError(
      "resizeBilinear output width must be a positive safe integer.",
    );
  const outputShape = [batch, outHeight, outWidth, channels];
  sizeOf(outputShape);
  return operation(
    outputShape,
    "resizeBilinear",
    resizeBilinearKernel,
    [input],
    [height, width, channels, outHeight, outWidth, alignCorners(options)],
    noGradient("resizeBilinear"),
  );
}
