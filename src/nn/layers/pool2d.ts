import type { Tensor } from "../../tensor/tensor.js";
import type { Pool2DOptions } from "../../tensor/spatial.js";
import { Module } from "../module.js";

/** The layer default: `strides` is `filterSize` when omitted, as in Keras and PyTorch. */
function withDefaultStrides(options: Pool2DOptions): Pool2DOptions {
  return { ...options, strides: options.strides ?? options.filterSize };
}

/**
 * Maximum over each window of NHWC input. It has no parameters. When `strides` is omitted it is
 * `filterSize`, so the windows do not overlap (the `maxPool` operation defaults to 1).
 */
export class MaxPool2d extends Module {
  readonly options: Pool2DOptions;

  constructor(options: Pool2DOptions) {
    super();
    this.options = withDefaultStrides(options);
  }

  forward(input: Tensor): Tensor {
    this.assertInputDevice(input.device);
    return input.maxPool(this.options);
  }
}

/**
 * Average over each window of NHWC input. It has no parameters. When `strides` is omitted it is
 * `filterSize`, so the windows do not overlap (the `avgPool` operation defaults to 1).
 */
export class AvgPool2d extends Module {
  readonly options: Pool2DOptions;

  constructor(options: Pool2DOptions) {
    super();
    this.options = withDefaultStrides(options);
  }

  forward(input: Tensor): Tensor {
    this.assertInputDevice(input.device);
    return input.avgPool(this.options);
  }
}
