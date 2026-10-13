import type { Tensor } from "../../tensor/tensor.js";
import { sizeOf } from "../../tensor/shape.js";
import { Optimizer } from "./optimizer.js";
import { adamFirstOf, adamSecondOf, adamUpdateOf } from "./update.js";

export interface AdamOptions {
  readonly lr: number;
  readonly beta1?: number;
  readonly beta2?: number;
  readonly epsilon?: number;
}

/** The moments of one parameter. They are device tensors of the parameter's shape. */
interface Moments {
  first: Tensor;
  second: Tensor;
}

export class Adam extends Optimizer {
  readonly kind = "adam" as const;
  readonly learningRate: number;
  readonly beta1: number;
  readonly beta2: number;
  readonly epsilon: number;
  private readonly moments = new Map<Tensor, Moments>();
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

  /**
   * Update every parameter and both moments on the device, in one plan. The new moments and
   * parameters take the arrays the plan computed; no gradient is read back and no parameter is
   * uploaded. The first step starts the moments at zero.
   */
  async step(): Promise<void> {
    const device = this.parameters[0]!.device;
    const gradients = this.gradientTensors();
    const iteration = this.iteration + 1;
    const firstCorrection = 1 - this.beta1 ** iteration;
    const secondCorrection = 1 - this.beta2 ** iteration;
    const moments = this.parameters.map((parameter) =>
      this.momentsOf(parameter),
    );
    const firsts = this.parameters.map((_, index) =>
      adamFirstOf(moments[index]!.first, gradients[index]!, this.beta1),
    );
    const seconds = this.parameters.map((_, index) =>
      adamSecondOf(moments[index]!.second, gradients[index]!, this.beta2),
    );
    const updates = this.parameters.map((parameter, index) =>
      adamUpdateOf(parameter, firsts[index]!, seconds[index]!, [
        this.learningRate,
        firstCorrection,
        secondCorrection,
        this.epsilon,
      ]),
    );
    await device.computeAll([...updates, ...firsts, ...seconds]);
    // The plan is complete: the results replace the arrays they were computed from.
    try {
      this.parameters.forEach((parameter, index) => {
        device.adopt(moments[index]!.first, firsts[index]!);
        device.adopt(moments[index]!.second, seconds[index]!);
        device.adopt(parameter, updates[index]!);
      });
    } finally {
      // An adoption that failed (a disposed parameter) leaves no computed array behind.
      for (const tensor of [...updates, ...firsts, ...seconds])
        device.release(tensor);
    }
    this.iteration = iteration;
  }

  override async dispose(): Promise<void> {
    // The moments hold no host value that a reader needs: free their arrays without a read-back.
    for (const { first, second } of this.moments.values()) {
      const device = first.device;
      if (device.isDisposed()) break;
      device.release(first);
      device.release(second);
    }
    this.moments.clear();
    this.iteration = 0;
  }

  /** The moments of a parameter, which start at zero. */
  private momentsOf(parameter: Tensor): Moments {
    let moments = this.moments.get(parameter);
    if (!moments) {
      const zeros = () =>
        parameter.device.tensor(new Float32Array(sizeOf(parameter.shape)), {
          shape: parameter.shape,
        });
      moments = { first: zeros(), second: zeros() };
      this.moments.set(parameter, moments);
    }
    return moments;
  }
}
