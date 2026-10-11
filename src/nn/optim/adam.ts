import type { Resident } from "typeshade/runtime";
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
  readonly firstResident?: Resident<Float32Array>;
  readonly secondResident?: Resident<Float32Array>;
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
    const device = this.parameters[0]!.device;
    this.iteration++;
    const firstCorrection = 1 - this.beta1 ** this.iteration;
    const secondCorrection = 1 - this.beta2 ** this.iteration;
    if (device.backend.name !== "cpu") {
      const updates = this.parameters.map((parameter) => {
        if (!parameter.grad)
          throw new Error("Call backward() before optimizer.step().");
        let moment = this.moments.get(parameter);
        if (!moment) {
          const size = parameter.shape.reduce(
            (count, dimension) => count * dimension,
            1,
          );
          moment = {
            first: new Float32Array(size),
            second: new Float32Array(size),
            firstResident: device.createResidentKernelBuffer(
              new Float32Array(size),
            ),
            secondResident: device.createResidentKernelBuffer(
              new Float32Array(size),
            ),
          };
          this.moments.set(parameter, moment);
        }
        const firstMoment = moment.firstResident;
        const secondMoment = moment.secondResident;
        if (!firstMoment || !secondMoment)
          throw new Error("Adam GPU moment buffers were not initialized.");
        return {
          parameter,
          gradient: parameter.grad,
          firstMoment,
          secondMoment,
        };
      });
      await device.runOptimizerUpdates("adam", updates, {
        learningRate: this.learningRate,
        beta1: this.beta1,
        beta2: this.beta2,
        epsilon: this.epsilon,
        firstCorrection,
        secondCorrection,
      });
      return;
    }
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
