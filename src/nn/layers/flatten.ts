import { Tensor } from "../../tensor/tensor.js";
import { Module } from "../module.js";

/** Flatten each sample while preserving the leading batch dimension. */
export class Flatten extends Module {
  forward(input: Tensor): Tensor {
    this.assertInputDevice(input.device);
    if (input.shape.length < 2)
      throw new RangeError("Flatten expects a batch dimension and features.");
    const batch = input.shape[0]!;
    const features = input.shape.slice(1).reduce((size, dim) => size * dim, 1);
    return input.reshape([batch, features]);
  }
}
