import type { Tensor } from "../../tensor/tensor.js";
import { Optimizer } from "./optimizer.js";
import { sgdUpdateOf } from "./update.js";

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

  /** Update every parameter on the device, in one plan. No gradient is read back. */
  async step(): Promise<void> {
    const gradients = this.gradientTensors();
    const updates = this.parameters.map((parameter, index) =>
      sgdUpdateOf(parameter, gradients[index]!, this.learningRate),
    );
    const device = this.parameters[0]!.device;
    await device.computeAll(updates);
    try {
      this.parameters.forEach((parameter, index) =>
        device.adopt(parameter, updates[index]!),
      );
    } finally {
      // An adoption that failed (a disposed parameter) leaves no computed array behind.
      for (const update of updates) device.release(update);
    }
  }
}
