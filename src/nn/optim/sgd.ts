import type { Tensor } from "../../tensor/tensor.js";
import { Optimizer } from "./optimizer.js";

export interface SGDOptions {
  readonly lr: number;
}

export class SGD extends Optimizer {
  readonly kind = "sgd" as const;
  readonly learningRate: number;

  constructor(parameters: readonly Tensor[], options: SGDOptions) {
    super(parameters);
    if (!Number.isFinite(options.lr) || options.lr <= 0)
      throw new RangeError("SGD lr must be a positive finite number.");
    this.learningRate = options.lr;
  }

  async step(): Promise<void> {
    const gradients = await this.gradients();
    for (const [index, parameter] of this.parameters.entries()) {
      const updated = parameter.parameterValues().slice();
      const gradient = gradients[index]!;
      for (let element = 0; element < updated.length; element++)
        updated[element] -= this.learningRate * gradient[element]!;
      await parameter.assign(updated);
    }
  }
}
