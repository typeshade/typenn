import { Tensor } from "../tensor/tensor.js";
import { Module } from "./module.js";

export class Sequential extends Module {
  readonly layers: readonly Module[];

  constructor(...layers: Module[]) {
    super();
    if (layers.length === 0)
      throw new RangeError("Sequential requires at least one layer.");
    this.layers = layers.map((layer, index) =>
      this.register(String(index), layer),
    );
  }

  forward(input: Tensor): Tensor {
    this.assertInputDevice(input.device);
    return this.layers.reduce((value, layer) => layer.forward(value), input);
  }
}
