import type { Device } from "../runtime/device.js";
import { type LinearTrainingEntry } from "../runtime/device.js";
import { Linear } from "./layers.js";

export interface SGDOptions {
  readonly learningRate: number;
}

export interface SGDOptimizer {
  readonly kind: "sgd";
  readonly learningRate: number;
}

export const optim = {
  SGD(options: SGDOptions): SGDOptimizer {
    if (!Number.isFinite(options.learningRate) || options.learningRate <= 0)
      throw new RangeError(
        "SGD learningRate must be a positive finite number.",
      );
    return { kind: "sgd", learningRate: options.learningRate };
  },
};

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

interface KernelShape {
  count: number;
  inputFeatures: number;
  outputFeatures: number;
  learningRate: number;
}

type Buffer =
  | Float32Array
  | Uint32Array
  | {
      read(): Promise<Float32Array | Uint32Array>;
      write(value: Float32Array | Uint32Array): void;
      destroy(): void;
    };

/**
 * Initial trainer: single Linear classifier with integer labels and SGD.
 * Forward, objective, explicit backward, and update all execute through TypeShade.
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

    const capacity = Math.min(options.batchSize, dataset.labels.length);
    const inputBatch = new Float32Array(capacity * this.model.inFeatures);
    const labelBatch = new Uint32Array(capacity);
    const shape: KernelShape = {
      count: capacity,
      inputFeatures: this.model.inFeatures,
      outputFeatures: this.model.outFeatures,
      learningRate: this.optimizer.learningRate,
    };
    const buffers = {
      inputs: this.device.createKernelBuffer(inputBatch),
      labels: this.device.createKernelBuffer(labelBatch),
      weights: this.device.createKernelBuffer(this.model.weight),
      bias: this.device.createKernelBuffer(this.model.bias),
      logits: this.device.createKernelBuffer(
        new Float32Array(capacity * this.model.outFeatures),
      ),
      delta: this.device.createKernelBuffer(
        new Float32Array(capacity * this.model.outFeatures),
      ),
      losses: this.device.createKernelBuffer(new Float32Array(capacity)),
      stats: this.device.createKernelBuffer(new Float32Array(2)),
      gradWeights: this.device.createKernelBuffer(
        new Float32Array(this.model.weight.length),
      ),
      gradBias: this.device.createKernelBuffer(
        new Float32Array(this.model.bias.length),
      ),
    };
    const history: EpochMetrics[] = [];

    const prepareBatch = (offset: number, count: number): KernelShape => {
      inputBatch.fill(0);
      labelBatch.fill(0);
      inputBatch.set(
        dataset.inputs.subarray(
          offset * this.model.inFeatures,
          (offset + count) * this.model.inFeatures,
        ),
      );
      labelBatch.set(dataset.labels.subarray(offset, offset + count));
      this.device.writeKernelBuffer(buffers.inputs, inputBatch);
      this.device.writeKernelBuffer(buffers.labels, labelBatch);
      return { ...shape, count };
    };

    const run = async (
      entries: readonly LinearTrainingEntry[],
      currentShape: KernelShape,
    ): Promise<void> => {
      const command = (
        entry: LinearTrainingEntry,
        bindings: Record<string, unknown>,
        workgroups: number,
      ) => ({
        entry,
        bindings: { shape: currentShape, ...bindings },
        workgroups,
      });
      const forward = command(
        "linearForward",
        {
          inputs: buffers.inputs,
          weights: buffers.weights,
          bias: buffers.bias,
          logits: buffers.logits,
        },
        Math.ceil((currentShape.count * currentShape.outputFeatures) / 64),
      );
      const objective = command(
        "linearObjective",
        {
          labels: buffers.labels,
          logits: buffers.logits,
          delta: buffers.delta,
          losses: buffers.losses,
        },
        Math.ceil(currentShape.count / 64),
      );
      const reduce = command(
        "linearReduce",
        {
          labels: buffers.labels,
          logits: buffers.logits,
          losses: buffers.losses,
          stats: buffers.stats,
        },
        1,
      );
      const backward = command(
        "linearBackward",
        {
          inputs: buffers.inputs,
          delta: buffers.delta,
          gradWeights: buffers.gradWeights,
          gradBias: buffers.gradBias,
        },
        Math.ceil(
          Math.max(this.model.weight.length, this.model.outFeatures) / 64,
        ),
      );
      const update = command(
        "linearUpdate",
        {
          weights: buffers.weights,
          bias: buffers.bias,
          gradWeights: buffers.gradWeights,
          gradBias: buffers.gradBias,
        },
        Math.ceil(
          Math.max(this.model.weight.length, this.model.outFeatures) / 64,
        ),
      );
      const commands = entries.map(
        (entry) =>
          ({
            linearForward: forward,
            linearObjective: objective,
            linearReduce: reduce,
            linearBackward: backward,
            linearUpdate: update,
          })[entry],
      );
      await this.device.runLinearTrainingFrame(commands);
    };

    try {
      for (let epoch = 1; epoch <= options.epochs; epoch++) {
        for (
          let offset = 0;
          offset < dataset.labels.length;
          offset += capacity
        ) {
          const count = Math.min(capacity, dataset.labels.length - offset);
          const currentShape = prepareBatch(offset, count);
          await run(
            [
              "linearForward",
              "linearObjective",
              "linearBackward",
              "linearUpdate",
            ],
            currentShape,
          );
        }

        let loss = 0;
        let correct = 0;
        for (
          let offset = 0;
          offset < dataset.labels.length;
          offset += capacity
        ) {
          const count = Math.min(capacity, dataset.labels.length - offset);
          const currentShape = prepareBatch(offset, count);
          await run(
            ["linearForward", "linearObjective", "linearReduce"],
            currentShape,
          );
          const stats = await this.device.readKernelBuffer(buffers.stats);
          loss += stats[0]! * count;
          correct += stats[1]!;
        }
        const metrics = {
          epoch,
          loss: loss / dataset.labels.length,
          accuracy: correct / dataset.labels.length,
        };
        history.push(metrics);
        options.onEpoch?.(metrics);
      }

      this.model.weight.set(
        (await this.device.readKernelBuffer(buffers.weights)) as Float32Array,
      );
      this.model.bias.set(
        (await this.device.readKernelBuffer(buffers.bias)) as Float32Array,
      );
      await this.model.refreshDeviceParameters(this.device);
      return history;
    } finally {
      for (const buffer of Object.values(buffers))
        this.device.releaseKernelBuffer(buffer);
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
