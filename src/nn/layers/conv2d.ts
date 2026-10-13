import { Tensor } from "../../tensor/tensor.js";
import type { Device } from "../../runtime/device.js";
import type { Conv2DOptions } from "../../tensor/spatial.js";
import { Module } from "../module.js";
import { random } from "./linear.js";

export interface Conv2dOptions {
  readonly kernelSize: number | readonly [number, number];
  readonly strides?: number | readonly [number, number];
  /** `"valid"` (the default) or `"same"`. */
  readonly padding?: "same" | "valid";
  /** Whether the layer adds a bias. The default is true. */
  readonly bias?: boolean;
  readonly seed?: number;
}

/**
 * A 2D convolution over NHWC input. The filter has the shape [kh, kw, inChannels, outChannels]
 * and the bias has the shape [outChannels]. The weights start with He-uniform values, and the
 * biases start at zero.
 */
export class Conv2d extends Module {
  readonly kernelSize: readonly [number, number];
  readonly strides: readonly [number, number];
  readonly padding: "same" | "valid";
  readonly weight: Float32Array;
  readonly bias?: Float32Array;
  private readonly deviceTensors = new WeakMap<
    Device,
    { weight: Tensor; bias?: Tensor }
  >();

  constructor(
    readonly inChannels: number,
    readonly outChannels: number,
    options: Conv2dOptions,
  ) {
    super();
    if (!Number.isSafeInteger(inChannels) || inChannels < 1)
      throw new RangeError(
        "Conv2d inChannels must be a positive safe integer.",
      );
    if (!Number.isSafeInteger(outChannels) || outChannels < 1)
      throw new RangeError(
        "Conv2d outChannels must be a positive safe integer.",
      );
    this.kernelSize = pair(options.kernelSize, "kernelSize");
    this.strides = pair(options.strides ?? 1, "strides");
    this.padding = options.padding ?? "valid";
    if (this.padding !== "valid" && this.padding !== "same")
      throw new TypeError(
        `Unknown Conv2d padding mode: ${String(this.padding)}.`,
      );

    const [kernelHeight, kernelWidth] = this.kernelSize;
    const fanIn = kernelHeight * kernelWidth * inChannels;
    const limit = Math.sqrt(6 / fanIn);
    const next = random(options.seed ?? 123);
    this.weight = Float32Array.from(
      { length: fanIn * outChannels },
      () => (next() * 2 - 1) * limit,
    );
    if (options.bias ?? true) this.bias = new Float32Array(outChannels);
  }

  forward(input: Tensor): Tensor {
    this.assertInputDevice(input.device);
    if (input.shape.length !== 4 || input.shape[3] !== this.inChannels)
      throw new RangeError(
        `Conv2d expected [batch, height, width, ${this.inChannels}] input; got [${input.shape.join(", ")}].`,
      );
    const { weight, bias } = this.parameterTensors(input.device);
    const output = input.conv2d(weight, this.conv2dOptions());
    return bias ? output.add(bias) : output;
  }

  protected override ownParameters(device: Device): readonly Tensor[] {
    const { weight, bias } = this.parameterTensors(device);
    return bias ? [weight, bias] : [weight];
  }

  /** @internal Expose graph leaves to optimizers without copying their values. */
  parameterTensors(device: Device): {
    readonly weight: Tensor;
    readonly bias?: Tensor;
  } {
    let params = this.deviceTensors.get(device);
    if (!params) {
      params = {
        weight: device.createParameter(this.weight, {
          shape: [...this.kernelSize, this.inChannels, this.outChannels],
        }),
        bias: this.bias
          ? device.createParameter(this.bias, { shape: [this.outChannels] })
          : undefined,
      };
      this.deviceTensors.set(device, params);
    }
    return params;
  }

  private conv2dOptions(): Conv2DOptions {
    return { strides: this.strides, padding: this.padding };
  }
}

function pair(
  value: number | readonly [number, number],
  name: string,
): readonly [number, number] {
  const [first, second] = typeof value === "number" ? [value, value] : value;
  if (
    !Number.isSafeInteger(first) ||
    first! < 1 ||
    !Number.isSafeInteger(second) ||
    second! < 1
  )
    throw new RangeError(`Conv2d ${name} must be positive safe integers.`);
  return [first!, second!];
}
