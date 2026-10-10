import { Tensor } from "../tensor/tensor.js";
import type { Device } from "../runtime/device.js";
import { Module } from "./module.js";

function random(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

export interface LinearOptions {
  readonly seed?: number;
  /** Initializer range for each weight; biases start at zero. */
  readonly weightScale?: number;
}

export class Linear extends Module {
  readonly weight: Float32Array;
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
    if (input.shape.length !== 2 || input.shape[1] !== this.inFeatures)
      throw new RangeError(
        `Linear expected [batch, ${this.inFeatures}] input; got [${input.shape.join(", ")}].`,
      );
    let params = this.deviceTensors.get(input.device);
    if (!params) {
      params = {
        weight: input.device.tensor(this.weight, {
          shape: [this.inFeatures, this.outFeatures],
        }),
        bias: input.device.tensor(this.bias, { shape: [this.outFeatures] }),
      };
      this.deviceTensors.set(input.device, params);
    }
    return input.matmul(params.weight).add(params.bias);
  }

  /** @internal Drop cached device tensors after a Trainer changes host-side parameters. */
  async refreshDeviceParameters(device: Device): Promise<void> {
    const params = this.deviceTensors.get(device);
    if (!params) return;
    await Promise.all([params.weight.dispose(), params.bias.dispose()]);
    this.deviceTensors.delete(device);
  }
}

export class ReLU extends Module {
  forward(input: Tensor): Tensor {
    return input.relu();
  }
}
