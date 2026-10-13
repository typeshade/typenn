import type { Tensor } from "../../tensor/tensor.js";
import { Module } from "../module.js";

/** The hyperbolic tangent, applied elementwise. */
export class Tanh extends Module {
  forward(input: Tensor): Tensor {
    this.assertInputDevice(input.device);
    return input.tanh();
  }
}
