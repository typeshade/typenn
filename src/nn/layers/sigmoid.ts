import type { Tensor } from "../../tensor/tensor.js";
import { Module } from "../module.js";

/** The logistic function, applied elementwise. */
export class Sigmoid extends Module {
  forward(input: Tensor): Tensor {
    this.assertInputDevice(input.device);
    return input.sigmoid();
  }
}
