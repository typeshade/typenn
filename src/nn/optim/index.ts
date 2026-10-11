import type { Tensor } from "../../tensor/tensor.js";
import { Adam, type AdamOptions } from "./adam.js";
import { SGD, type SGDOptions } from "./sgd.js";

export { Adam, SGD, type AdamOptions, type SGDOptions };

export interface LegacySGDOptions {
  readonly learningRate: number;
}

export interface LegacySGDOptimizer {
  readonly kind: "sgd";
  readonly learningRate: number;
}

function createSGD(parameters: readonly Tensor[], options: SGDOptions): SGD;
function createSGD(options: LegacySGDOptions): LegacySGDOptimizer;
function createSGD(
  parametersOrOptions: readonly Tensor[] | LegacySGDOptions,
  options?: SGDOptions,
): SGD | LegacySGDOptimizer {
  if (!("learningRate" in parametersOrOptions)) {
    if (!options) throw new TypeError("SGD requires optimizer options.");
    return new SGD(parametersOrOptions, options);
  }
  if (
    !Number.isFinite(parametersOrOptions.learningRate) ||
    parametersOrOptions.learningRate <= 0
  )
    throw new RangeError("SGD learningRate must be a positive finite number.");
  return { kind: "sgd", learningRate: parametersOrOptions.learningRate };
}

export const optim = {
  SGD: createSGD,
  Adam(parameters: readonly Tensor[], options: AdamOptions): Adam {
    return new Adam(parameters, options);
  },
};
