import type { Device } from "../runtime/device.js";
import { Linear } from "./layers.js";
import { CrossEntropyLoss } from "./loss/cross-entropy.js";
import { SGD } from "./optim/sgd.js";
export interface SGDOptimizer {
  readonly kind: "sgd";
  readonly learningRate: number;
}

export interface SGDOptions {
  readonly learningRate: number;
}

export interface ClassificationDataset {
  /** Row-major f32 samples: `size * inputFeatures` values. */
  readonly inputs: Float32Array;
  /** Integer class id for each sample. */
  readonly labels: Uint32Array;
}

export interface TrainerOptions {
  readonly device: Device;
  readonly optimizer: SGDOptimizer;
}

export interface FitOptions {
  readonly epochs: number;
  readonly batchSize: number;
  readonly onEpoch?: (metrics: EpochMetrics) => void;
}

export interface EpochMetrics {
  readonly epoch: number;
  readonly loss: number;
  readonly accuracy: number;
}

/**
 * Initial trainer: single Linear classifier with integer labels and SGD. A step is the Tensor
 * graph `inputs.matmul(weight).add(bias)`, the mean cross entropy of its logits, `backward()` and
 * an `SGD` step, so it runs on whatever kernel functions and tier the device has.
 */
export class Trainer {
  private disposed = false;
  private fitting = false;

  constructor(
    readonly model: Linear,
    readonly device: Device,
    readonly optimizer: SGDOptimizer,
  ) {
    device.assertUsable();
  }

  async fit(
    dataset: ClassificationDataset,
    options: FitOptions,
  ): Promise<EpochMetrics[]> {
    this.assertUsable();
    if (this.fitting) throw new Error("Trainer.fit() is already running.");
    this.validate(dataset, options);
    this.fitting = true;

    const { inFeatures } = this.model;
    const { weight, bias } = this.model.parameterTensors(this.device);
    const sgd = new SGD([weight, bias], { lr: this.optimizer.learningRate });
    const training = new CrossEntropyLoss({ reduction: "mean" });
    const evaluation = new CrossEntropyLoss({ reduction: "none" });
    const capacity = Math.min(options.batchSize, dataset.labels.length);
    const history: EpochMetrics[] = [];

    /** The logits of the samples from `offset` on, and the labels of those samples. */
    const forward = (offset: number, count: number) => {
      const inputs = this.device.tensor(
        dataset.inputs.subarray(
          offset * inFeatures,
          (offset + count) * inFeatures,
        ),
        { shape: [count, inFeatures] },
      );
      return {
        logits: inputs.matmul(weight).add(bias),
        labels: dataset.labels.subarray(offset, offset + count),
      };
    };

    try {
      for (let epoch = 1; epoch <= options.epochs; epoch++) {
        for (
          let offset = 0;
          offset < dataset.labels.length;
          offset += capacity
        ) {
          const count = Math.min(capacity, dataset.labels.length - offset);
          const { logits, labels } = forward(offset, count);
          const loss = training.forward(logits, labels);
          await loss.backward();
          await sgd.step();
          await sgd.zeroGrad();
          await loss.disposeGraph([weight, bias]);
        }

        let loss = 0;
        let correct = 0;
        for (
          let offset = 0;
          offset < dataset.labels.length;
          offset += capacity
        ) {
          const count = Math.min(capacity, dataset.labels.length - offset);
          const { logits, labels } = forward(offset, count);
          const losses = evaluation.forward(logits, labels);
          const predictions = logits.argMax(1);
          const [rowLosses, rowPredictions] = [
            await losses.data(),
            await predictions.data(),
          ];
          for (let row = 0; row < count; row++) {
            loss += rowLosses[row]!;
            if (rowPredictions[row] === labels[row]) correct++;
          }
          await predictions.disposeGraph([weight, bias]);
          await losses.disposeGraph([weight, bias]);
        }
        const metrics = {
          epoch,
          loss: loss / dataset.labels.length,
          accuracy: correct / dataset.labels.length,
        };
        history.push(metrics);
        options.onEpoch?.(metrics);
      }
      return history;
    } finally {
      this.fitting = false;
    }
  }

  async dispose(): Promise<void> {
    if (this.fitting)
      throw new Error("Cannot dispose a Trainer while fit() is running.");
    this.disposed = true;
  }

  private validate(dataset: ClassificationDataset, options: FitOptions): void {
    const size = dataset.labels.length;
    if (size < 1 || dataset.inputs.length !== size * this.model.inFeatures)
      throw new RangeError(
        "Dataset input and label lengths do not match the Linear input size.",
      );
    for (const label of dataset.labels) {
      if (label >= this.model.outFeatures)
        throw new RangeError(
          "Dataset label exceeds the Linear output class count.",
        );
    }
    if (!Number.isSafeInteger(options.epochs) || options.epochs < 1)
      throw new RangeError("epochs must be a positive safe integer.");
    if (!Number.isSafeInteger(options.batchSize) || options.batchSize < 1)
      throw new RangeError("batchSize must be a positive safe integer.");
    for (const value of dataset.inputs) {
      if (!Number.isFinite(value))
        throw new RangeError("Dataset inputs must be finite.");
    }
  }

  private assertUsable(): void {
    if (this.disposed) throw new Error("Trainer has been disposed.");
    this.device.assertUsable();
  }
}

export function createTrainer(model: Linear, options: TrainerOptions): Trainer {
  if (!(model instanceof Linear))
    throw new TypeError(
      "The initial TypeNN Trainer supports a single Linear classifier.",
    );
  if (options.optimizer.kind !== "sgd")
    throw new TypeError("The initial TypeNN Trainer supports SGD only.");
  if (
    !Number.isFinite(options.optimizer.learningRate) ||
    options.optimizer.learningRate <= 0
  )
    throw new RangeError("SGD learningRate must be a positive finite number.");
  return new Trainer(model, options.device, options.optimizer);
}
