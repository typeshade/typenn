import { Tensor } from "../../tensor/tensor.js";
import type { Device } from "../../runtime/device.js";
import { Module } from "../module.js";
import { random } from "../random.js";

export interface LinearOptions {
  readonly seed?: number;
  /** Initializer range for each weight; biases start at zero. */
  readonly weightScale?: number;
}

export class Linear extends Module {
  /**
   * The host copy of the weights, row-major `[inFeatures, outFeatures]`. An optimizer updates the
   * parameter on the device, so this array holds the trained values only after `Module.sync()`,
   * after `Trainer.fit()`, or after a read of the parameter (`parameterValues()`, `data()`).
   */
  readonly weight: Float32Array;
  /** The host copy of the biases. It is refreshed like `weight`. */
  readonly bias: Float32Array;
  private readonly deviceTensors = new WeakMap<
    Device,
    { weight: Tensor; bias: Tensor }
  >();

  constructor(
    readonly inFeatures: number,
    readonly outFeatures: number,
    options: LinearOptions = {},
  ) {
    super();
    if (!Number.isSafeInteger(inFeatures) || inFeatures < 1)
      throw new RangeError(
        "Linear inFeatures must be a positive safe integer.",
      );
    if (!Number.isSafeInteger(outFeatures) || outFeatures < 1)
      throw new RangeError(
        "Linear outFeatures must be a positive safe integer.",
      );
    const scale = options.weightScale ?? 0.01;
    if (!Number.isFinite(scale) || scale < 0)
      throw new RangeError("weightScale must be non-negative.");
    const next = random(options.seed ?? 123);
    this.weight = Float32Array.from(
      { length: inFeatures * outFeatures },
      () => (next() - 0.5) * (scale * 2),
    );
    this.bias = new Float32Array(outFeatures);
  }

  forward(input: Tensor): Tensor {
    this.assertInputDevice(input.device);
    if (input.shape.length !== 2 || input.shape[1] !== this.inFeatures)
      throw new RangeError(
        `Linear expected [batch, ${this.inFeatures}] input; got [${input.shape.join(", ")}].`,
      );
    const { weight, bias } = this.parameterTensors(input.device);
    return input.matmul(weight).add(bias);
  }

  protected override ownParameters(device: Device): readonly Tensor[] {
    const { weight, bias } = this.parameterTensors(device);
    return [weight, bias];
  }

  /** @internal Expose graph leaves to optimizers without copying their values. */
  parameterTensors(device: Device): {
    readonly weight: Tensor;
    readonly bias: Tensor;
  } {
    let params = this.deviceTensors.get(device);
    if (!params) {
      params = {
        weight: device.createParameter(this.weight, {
          shape: [this.inFeatures, this.outFeatures],
        }),
        bias: device.createParameter(this.bias, { shape: [this.outFeatures] }),
      };
      this.deviceTensors.set(device, params);
    }
    return params;
  }

  /** @internal Drop cached device tensors after a Trainer changes host-side parameters. */
  async refreshDeviceParameters(device: Device): Promise<void> {
    const params = this.deviceTensors.get(device);
    if (!params) return;
    await Promise.all([params.weight.dispose(), params.bias.dispose()]);
    this.deviceTensors.delete(device);
  }
}
