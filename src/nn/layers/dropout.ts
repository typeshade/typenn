import { sizeOf } from "../../tensor/shape.js";
import { Tensor } from "../../tensor/tensor.js";
import { Module } from "../module.js";

export interface DropoutOptions {
  readonly seed?: number;
}

function random(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

export class Dropout extends Module {
  private readonly next: () => number;

  constructor(
    readonly rate: number,
    options: DropoutOptions = {},
  ) {
    super();
    if (!Number.isFinite(rate) || rate < 0 || rate >= 1)
      throw new RangeError("Dropout rate must be in the range [0, 1).");
    this.next = random(options.seed ?? 123);
  }

  forward(input: Tensor): Tensor {
    this.assertInputDevice(input.device);
    if (!this.isTraining() || this.rate === 0) return input;
    const keepProbability = 1 - this.rate;
    const mask = new Float32Array(sizeOf(input.shape));
    for (let i = 0; i < mask.length; i++)
      mask[i] = this.next() < keepProbability ? 1 / keepProbability : 0;
    const maskTensor = input.device.tensor(mask, { shape: input.shape });
    return input.mul(maskTensor);
  }
}
