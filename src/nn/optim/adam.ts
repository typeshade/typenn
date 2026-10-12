import type { Tensor } from "../../tensor/tensor.js";
import { Optimizer } from "./optimizer.js";

export interface AdamOptions {
  readonly lr: number;
  readonly beta1?: number;
  readonly beta2?: number;
  readonly epsilon?: number;
}

interface Moments {
  readonly first: Float32Array;
  readonly second: Float32Array;
}

export class Adam extends Optimizer {
  readonly kind = "adam" as const;
  readonly learningRate: number;
  readonly beta1: number;
  readonly beta2: number;
  readonly epsilon: number;
  private readonly moments = new WeakMap<Tensor, Moments>();
  private iteration = 0;

  constructor(parameters: readonly Tensor[], options: AdamOptions) {
    super(parameters);
    this.learningRate = options.lr;
    this.beta1 = options.beta1 ?? 0.9;
    this.beta2 = options.beta2 ?? 0.999;
    this.epsilon = options.epsilon ?? 1e-8;
    if (!Number.isFinite(this.learningRate) || this.learningRate <= 0)
      throw new RangeError("Adam lr must be a positive finite number.");
    if (this.beta1 < 0 || this.beta1 >= 1 || !Number.isFinite(this.beta1))
      throw new RangeError("Adam beta1 must be in the range [0, 1).");
    if (this.beta2 < 0 || this.beta2 >= 1 || !Number.isFinite(this.beta2))
      throw new RangeError("Adam beta2 must be in the range [0, 1).");
    if (!Number.isFinite(this.epsilon) || this.epsilon <= 0)
      throw new RangeError("Adam epsilon must be a positive finite number.");
  }

  async step(): Promise<void> {
    const gradients = await this.gradients();
    this.iteration++;
    const firstCorrection = 1 - this.beta1 ** this.iteration;
    const secondCorrection = 1 - this.beta2 ** this.iteration;
    for (const [index, parameter] of this.parameters.entries()) {
      const updated = parameter.parameterValues().slice();
      const gradient = gradients[index]!;
      let moment = this.moments.get(parameter);
      if (!moment) {
        moment = {
          first: new Float32Array(updated.length),
          second: new Float32Array(updated.length),
        };
        this.moments.set(parameter, moment);
      }
      for (let element = 0; element < updated.length; element++) {
        const value = gradient[element]!;
        moment.first[element] =
          this.beta1 * moment.first[element]! + (1 - this.beta1) * value;
        moment.second[element] =
          this.beta2 * moment.second[element]! +
          (1 - this.beta2) * value * value;
        const first = moment.first[element]! / firstCorrection;
        const second = moment.second[element]! / secondCorrection;
        updated[element] -=
          (this.learningRate * first) / (Math.sqrt(second) + this.epsilon);
      }
      await parameter.assign(updated);
    }
  }
}
