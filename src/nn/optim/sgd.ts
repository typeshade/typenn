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
    const updates = [];
    for (const parameter of this.parameters) {
      updates.push({
        parameter,
        values: parameter.parameterValues().slice(),
        gradient: await this.gradient(parameter),
      });
    }
    for (const { parameter, values, gradient } of updates) {
      const updated = values;
      for (let index = 0; index < updated.length; index++)
        updated[index] -= this.learningRate * gradient[index]!;
      await parameter.assign(updated);
    }
  }
}
