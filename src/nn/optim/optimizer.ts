import type { Tensor } from "../../tensor/tensor.js";

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

  abstract step(): Promise<void>;

  protected async gradient(parameter: Tensor): Promise<Float32Array> {
    if (!parameter.grad)
      throw new Error("Call backward() before optimizer.step().");
    return parameter.grad.data();
  }
}
