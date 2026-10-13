import type { Tensor } from "../../tensor/tensor.js";
import type { Pool2DOptions } from "../../tensor/spatial.js";
import { Module } from "../module.js";

/** Maximum over each window of NHWC input. It has no parameters. */
export class MaxPool2d extends Module {
  constructor(readonly options: Pool2DOptions) {
    super();
  }

  forward(input: Tensor): Tensor {
    this.assertInputDevice(input.device);
    return input.maxPool(this.options);
  }
}

/** Average over each window of NHWC input. It has no parameters. */
export class AvgPool2d extends Module {
  constructor(readonly options: Pool2DOptions) {
    super();
  }

  forward(input: Tensor): Tensor {
    this.assertInputDevice(input.device);
    return input.avgPool(this.options);
  }
}
