import { Tensor } from "../../tensor/tensor.js";
import { Module } from "../module.js";

export class ReLU extends Module {
  forward(input: Tensor): Tensor {
    this.assertInputDevice(input.device);
    return input.relu();
  }
}
