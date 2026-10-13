import type { Tensor } from "../../tensor/tensor.js";

/** The options of `step()`. */
export interface StepOptions {
  /**
   * Tensors to read back in the same plan as the update. Each is computed in that plan from the
   * parameter values before the update, and read once after the plan. A loss that the step does
   * not otherwise compute is computed there too, so reading it adds no plan.
   */
  readonly alsoRead?: readonly Tensor[];
}

export abstract class Optimizer {
  protected constructor(readonly parameters: readonly Tensor[]) {
    if (parameters.length === 0)
      throw new RangeError("An Optimizer requires at least one parameter.");
    const device = parameters[0]!.device;
    if (new Set(parameters).size !== parameters.length)
      throw new RangeError("An Optimizer cannot contain duplicate parameters.");
    for (const parameter of parameters) {
      if (!parameter.requiresGrad)
        throw new TypeError("Optimizers require trainable parameter tensors.");
      if (parameter.device !== device)
        throw new TypeError("An Optimizer's parameters must share one device.");
    }
  }

  async zeroGrad(): Promise<void> {
    const retained = this.parameters;
    await Promise.all(
      this.parameters.map(async (parameter) => {
        await parameter.grad?.disposeGraph(retained);
        parameter.grad = undefined;
      }),
    );
  }

  /**
   * Update every parameter on the device, in one plan, and return the values of `alsoRead` in the
   * order given. Nothing is read back except those values and the partials of a reduction. The
   * step does not wait for the device to run the plan.
   */
  abstract step(options?: StepOptions): Promise<Float32Array[]>;

  /** The `alsoRead` tensors of `options`, checked for this optimizer's device. */
  protected readTensors(
    options: StepOptions,
    device: object,
  ): readonly Tensor[] {
    const alsoRead = options.alsoRead ?? [];
    for (const tensor of alsoRead) {
      tensor.assertUsable();
      if (tensor.device !== device)
        throw new TypeError(
          "alsoRead tensors must share the optimizer's device.",
        );
    }
    return alsoRead;
  }

  /**
   * Free the arrays the optimizer holds on the device besides the parameters (the moments of
   * Adam). The parameters and the device are not disposed. Call it when the optimizer is no
   * longer used and the device lives on. A later `step()` starts the state again from zero.
   */
  async dispose(): Promise<void> {}

  /**
   * The gradient tensor of each parameter, not yet read. `step()` computes them on the device in
   * the plan of its update. Throws when `backward()` has not run.
   */
  protected gradientTensors(): Tensor[] {
    return this.parameters.map((parameter) => {
      parameter.assertUsable();
      if (!parameter.grad)
        throw new Error("Call backward() before optimizer.step().");
      return parameter.grad;
    });
  }

  /**
   * The gradient of each parameter, read back in one plan. Not used by `step()`, which updates
   * the parameters on the device; kept for the kernel-graph view of the lab.
   */
  protected async gradients(): Promise<Float32Array[]> {
    return this.parameters[0]!.device.materializeAll(this.gradientTensors());
  }
}
