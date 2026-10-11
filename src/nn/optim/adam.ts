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
    if (this.parameters.some((parameter) => !parameter.grad))
      throw new Error("Call backward() before optimizer.step().");
    this.iteration++;
    const firstCorrection = 1 - this.beta1 ** this.iteration;
    const secondCorrection = 1 - this.beta2 ** this.iteration;
    const updates = [];
    for (const parameter of this.parameters) {
      updates.push({
        parameter,
        values: parameter.parameterValues().slice(),
        gradient: await this.gradient(parameter),
      });
    }
    for (const { parameter, values, gradient } of updates) {
      let moment = this.moments.get(parameter);
      if (!moment) {
        moment = {
          first: new Float32Array(values.length),
          second: new Float32Array(values.length),
        };
        this.moments.set(parameter, moment);
      }
      const updated = values;
      for (let index = 0; index < updated.length; index++) {
        const value = gradient[index]!;
        moment.first[index] =
          this.beta1 * moment.first[index]! + (1 - this.beta1) * value;
        moment.second[index] =
          this.beta2 * moment.second[index]! + (1 - this.beta2) * value * value;
        const first = moment.first[index]! / firstCorrection;
        const second = moment.second[index]! / secondCorrection;
        updated[index] -=
          (this.learningRate * first) / (Math.sqrt(second) + this.epsilon);
      }
      await parameter.assign(updated);
    }
  }
}
