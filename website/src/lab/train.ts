// Training, evaluation and prediction for the Lab. Framework-free: the TypeNN namespace is passed
// in, so the same code runs in the island, in the automation hook and in a node check.

import type { LoadedDataset, LoadedSplit } from "./datasets";
import { bagOfWordsBatch, imageBatch, tabularBatch } from "./datasets";
import version from "../vendor/typenn/version.json";

export type NN = typeof import("../vendor/typenn/index.js");
export type Device = import("../vendor/typenn/index.js").Device;
export type DeviceDataset = import("../vendor/typenn/index.js").DeviceDataset;
export type Residency = import("../vendor/typenn/index.js").Residency;
type DatasetInputs = import("../vendor/typenn/index.js").DatasetInputs;
type DatasetSplit = import("../vendor/typenn/index.js").DatasetSplit;
type DeviceDatasetOptions = import("../vendor/typenn/index.js").DeviceDatasetOptions;
type Module = import("../vendor/typenn/index.js").Module;

export type Preset = "mlp" | "cnn" | "bag" | "regression";
export type OptimizerName = "adam" | "sgd";
export type TierPreference = "auto" | "webgpu" | "webgl2" | "cpu";

export interface RunConfig {
  readonly dataset: string;
  readonly model: Preset;
  readonly hidden: number;
  readonly optimizer: OptimizerName;
  readonly learningRate: number;
  readonly batchSize: number;
  readonly epochs: number;
  readonly seed: number;
  readonly tier: TierPreference;
  /** Smoke runs: stop after this many training steps in all. The last epoch is still evaluated. */
  readonly maxSteps?: number;
  /** Smoke runs: evaluate on this many batches of 256 test samples, from the first. */
  readonly maxEvalBatches?: number;
  /** "quick" marks the short run of the Try section. Absent means "full". */
  readonly kind?: RunKind;
}

export type RunKind = "quick" | "full";

export interface Metrics {
  /** Share of correct predictions, from 0 to 1 (classification). */
  readonly accuracy?: number;
  /** Root mean squared error in USD (regression). */
  readonly rmse?: number;
  /** Mean absolute error in USD (regression). */
  readonly mae?: number;
}

export interface EpochResult {
  readonly epoch: number;
  /** Mean training loss of the epoch, over its steps. */
  readonly trainLoss: number;
  readonly test: Metrics;
  /** Test samples the metrics were computed on: all of them unless maxEvalBatches capped it. */
  readonly testSamples: number;
  readonly seconds: number;
}

export interface RunResult {
  readonly config: RunConfig;
  readonly tier: string;
  readonly fallbackReason?: string;
  readonly epochs: readonly EpochResult[];
  readonly steps: number;
  readonly stepsPerEpoch: number;
  readonly samplesSeen: number;
  readonly seconds: number;
  /** The metrics of the last finished evaluation. */
  readonly test: Metrics;
  /** False when the run was stopped before its last epoch. */
  readonly completed: boolean;
  /** "quick" for the short run of the Try section, else "full". */
  readonly kind: RunKind;
  /** True when the run ended at config.maxSteps. A user stop leaves it false. */
  readonly capped: boolean;
  /** Where the batches of the run were built: on the device, or on the host with the reason. */
  readonly residency: Residency;
  /** The device bytes the dataset held during the run. */
  readonly datasetBytes: number;
  /** Host time of the one dataset upload, in seconds. */
  readonly uploadSeconds: number;
  readonly userAgent: string;
  readonly typennCommit: string;
  readonly finishedAt: string;
}

export interface StepInfo {
  readonly epoch: number;
  /** 1-based step within the epoch. */
  readonly step: number;
  readonly stepsPerEpoch: number;
  readonly loss: number;
  readonly samplesPerSecond: number;
  readonly elapsedSeconds: number;
}

export interface Hooks {
  onStep?: (info: StepInfo) => void;
  onEpoch?: (info: {
    readonly epoch: number;
    readonly testMetric: number;
    readonly seconds: number;
    readonly metrics: Metrics;
  }) => void;
  shouldStop?: () => boolean;
  /** Called once the model is on the device, before the first step. The caller keeps it for prediction. */
  onModel?: (model: Module) => void;
  /** Called with the device dataset of the run. The caller then owns it and disposes it. */
  onDataset?: (dataset: DeviceDataset) => void;
}

/** The shape of the input and the output of a model, from the dataset. */
export interface ModelSpec {
  readonly preset: Preset;
  /** Per-sample input shape: [h, w, c] for images, [vocabulary] for text, [features] for tables. */
  readonly inputShape: readonly number[];
  readonly outputs: number;
  readonly hidden: number;
  readonly seed: number;
  /** Feature normalisation of a tabular dataset, from its manifest (train-split statistics). */
  readonly features?: {
    readonly mean: readonly number[];
    readonly std: readonly number[];
  };
}

export interface TargetScale {
  readonly mean: number;
  readonly std: number;
}

/** Map a dataset to the model it can train. */
export function specFor(data: LoadedDataset, config: RunConfig): ModelSpec {
  const { manifest } = data;
  const outputs =
    manifest.task === "regression" ? 1 : (manifest.classes?.length ?? 0);
  const inputShape =
    manifest.task === "image-classification"
      ? manifest.splits.train.x.shape
      : manifest.task === "text-classification"
        ? [manifest.vocabulary ?? 10000]
        : [manifest.features?.length ?? 8];
  const features = manifest.normalization
    ? { mean: manifest.normalization.mean, std: manifest.normalization.std }
    : undefined;
  return {
    preset: config.model,
    inputShape,
    outputs,
    hidden: config.hidden,
    seed: config.seed,
    features,
  };
}

/** Per-sample input size, the product of the input shape. */
export function inputSize(spec: ModelSpec): number {
  return spec.inputShape.reduce((product, n) => product * n, 1);
}

/** Build the preset model (not yet on a device). Layers take their seeds from `spec.seed`. */
export function buildModel(spec: ModelSpec, nn: NN): Module {
  const seed = spec.seed;
  const linear = (input: number, output: number, k: number) =>
    new nn.Linear(input, output, {
      seed: seed + k,
      weightScale: 1 / Math.sqrt(input),
    });
  const { preset, hidden, outputs } = spec;
  const inputs = inputSize(spec);

  switch (preset) {
    case "mlp":
      return new nn.Sequential(
        new nn.Flatten(),
        linear(inputs, hidden, 1),
        new nn.ReLU(),
        new nn.Dropout(0.2, { seed: seed + 2 }),
        linear(hidden, outputs, 3),
      );
    case "cnn": {
      const [height, width, channels] = spec.inputShape as [
        number,
        number,
        number,
      ];
      const flat = 32 * Math.floor(height / 4) * Math.floor(width / 4);
      return new nn.Sequential(
        new nn.Conv2d(channels, 16, {
          kernelSize: 3,
          padding: "same",
          seed: seed + 1,
        }),
        new nn.ReLU(),
        new nn.MaxPool2d({ filterSize: 2 }),
        new nn.Conv2d(16, 32, {
          kernelSize: 3,
          padding: "same",
          seed: seed + 2,
        }),
        new nn.ReLU(),
        new nn.MaxPool2d({ filterSize: 2 }),
        new nn.Flatten(),
        linear(flat, outputs, 3),
      );
    }
    case "bag":
      return new nn.Sequential(
        linear(inputs, hidden, 1),
        new nn.ReLU(),
        new nn.Dropout(0.3, { seed: seed + 2 }),
        linear(hidden, outputs, 3),
      );
    case "regression":
      return new nn.Sequential(
        linear(inputs, hidden, 1),
        new nn.ReLU(),
        linear(hidden, 1, 2),
      );
  }
}

/** Fixed preset per task, and the presets that a dataset can take. */
export function presetsFor(task: LoadedDataset["manifest"]["task"]): Preset[] {
  if (task === "image-classification") return ["mlp", "cnn"];
  if (task === "text-classification") return ["bag"];
  return ["regression"];
}

/** Seeded RNG (mulberry32). Same seed, same sequence on every browser. */
export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 0x100000000;
  };
}

/** Mean and standard deviation of the training targets. A zero std is treated as 1. */
export function targetScale(split: LoadedSplit): TargetScale {
  let sum = 0;
  for (let i = 0; i < split.count; i++) sum += split.y[i]!;
  const mean = sum / split.count;
  let square = 0;
  for (let i = 0; i < split.count; i++) square += (split.y[i]! - mean) ** 2;
  const std = Math.sqrt(square / split.count);
  return { mean, std: std === 0 ? 1 : std };
}

const YIELD = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

/**
 * The device-dataset options of a run: the sources of the split, uploaded once to the device. The
 * regression target is normalised on the device with the train-split statistics.
 */
export function datasetOptions(
  data: LoadedDataset,
  spec: ModelSpec,
  scale: TargetScale | undefined,
): DeviceDatasetOptions {
  const size = inputSize(spec);
  const inputs: DatasetInputs =
    spec.preset === "bag"
      ? { kind: "text", vocab: spec.inputShape[0]! }
      : spec.preset === "regression"
        ? {
            kind: "tabular",
            features: size,
            mean: spec.features!.mean,
            std: spec.features!.std,
            shape: spec.inputShape,
          }
        : { kind: "images", features: size, shape: spec.inputShape };
  return {
    inputs,
    train: toSplit(data.train),
    test: toSplit(data.test),
    target: scale,
  };
}

function toSplit(split: LoadedSplit): DatasetSplit {
  return {
    count: split.count,
    x: split.x,
    y: split.y,
    offsets: split.offsets,
  };
}

/**
 * Train the preset model on the full train split for `config.epochs` epochs, then evaluate on the
 * full test split after each epoch. The batches are built on the device from a dataset uploaded
 * once per run. `device` is opened by the caller and is not disposed here. The dataset goes to
 * `hooks.onDataset` when that hook is set, and the caller then disposes it; otherwise it is
 * disposed when the run ends.
 */
export async function train(
  config: RunConfig,
  data: LoadedDataset,
  nn: NN,
  device: Device,
  hooks: Hooks = {},
): Promise<RunResult> {
  const spec = specFor(data, config);
  const model = buildModel(spec, nn).to(device);
  hooks.onModel?.(model);
  const parameters = model.parameters();
  const regression = config.model === "regression";
  if (regression && !spec.features)
    throw new Error("A regression dataset needs its feature normalisation.");
  const scale = regression ? targetScale(data.train) : undefined;
  const optimizer =
    config.optimizer === "adam"
      ? nn.optim.Adam(parameters, { lr: config.learningRate })
      : nn.optim.SGD(parameters, { lr: config.learningRate });
  const criterion = regression ? new nn.MSELoss() : new nn.CrossEntropyLoss();
  const N = data.train.count;
  const B = config.batchSize;
  const stepsPerEpoch = Math.ceil(N / B);
  const rng = mulberry32(config.seed);
  const uploadStart = performance.now();
  const dataset = device.dataset(datasetOptions(data, spec, scale));
  const uploadSeconds = (performance.now() - uploadStart) / 1000;
  const keep = hooks.onDataset !== undefined;
  hooks.onDataset?.(dataset);
  const started = performance.now();
  const epochs: EpochResult[] = [];
  let steps = 0;
  let samples = 0;
  let test: Metrics = {};
  let completed = true;
  let capped = false;

  try {
    stop: for (let epoch = 1; epoch <= config.epochs; epoch++) {
      model.train();
      const epochStart = performance.now();
      // One upload of the epoch order: a seed drawn from the run's generator.
      dataset.shuffle(Math.floor(rng() * 0x100000000));
      let lossSum = 0;
      let epochSamples = 0;
      let s = 0;
      for (const batch of dataset.batches(B)) {
        s++;
        if (hooks.shouldStop?.()) {
          completed = false;
          break stop;
        }
        if (config.maxSteps !== undefined && steps >= config.maxSteps) {
          capped = true;
          break;
        }
        const n = batch.size;
        await optimizer.zeroGrad();
        const logits = model.forward(batch.input);
        const loss = regression
          ? (criterion as InstanceType<NN["MSELoss"]>).forward(
              logits,
              batch.labels,
            )
          : (criterion as InstanceType<NN["CrossEntropyLoss"]>).forward(
              logits,
              batch.labels,
            );
        await loss.backward();
        // One plan updates the parameters and reads the loss; the loss is the value before the update.
        const [[value]] = await optimizer.step({ alsoRead: [loss] });
        await loss.disposeGraph(parameters);
        const lossValue = value ?? Number.NaN;
        lossSum += lossValue * n;
        epochSamples += n;
        steps++;
        samples += n;
        if (config.maxSteps !== undefined && steps >= config.maxSteps)
          capped = true;
        const elapsed = (performance.now() - started) / 1000;
        hooks.onStep?.({
          epoch,
          step: s,
          stepsPerEpoch,
          loss: lossValue,
          samplesPerSecond: samples / Math.max(elapsed, 1e-9),
          elapsedSeconds: elapsed,
        });
        await YIELD();
        if (capped) break;
      }
      const evaluated = await evaluate(
        model,
        device,
        data,
        dataset,
        spec,
        scale,
        256,
        config.maxEvalBatches,
      );
      const evalMetrics = evaluated.metrics;
      test = evalMetrics;
      const seconds = (performance.now() - epochStart) / 1000;
      epochs.push({
        epoch,
        trainLoss: epochSamples > 0 ? lossSum / epochSamples : Number.NaN,
        test: evalMetrics,
        testSamples: evaluated.samples,
        seconds,
      });
      hooks.onEpoch?.({
        epoch,
        testMetric: regression ? evalMetrics.rmse! : evalMetrics.accuracy!,
        seconds,
        metrics: evalMetrics,
      });
      if (capped) {
        completed = false;
        break stop;
      }
    }
  } finally {
    if (!keep) dataset.dispose();
  }

  return {
    config,
    tier: device.backend.name,
    fallbackReason: device.backend.fallbackReason,
    epochs,
    steps,
    stepsPerEpoch,
    samplesSeen: samples,
    seconds: (performance.now() - started) / 1000,
    test,
    completed,
    kind: config.kind ?? "full",
    capped,
    residency: dataset.residency,
    datasetBytes: dataset.bytes,
    uploadSeconds,
    userAgent:
      typeof navigator === "undefined" ? "unknown" : navigator.userAgent,
    typennCommit: version.commit,
    finishedAt: new Date().toISOString(),
  };
}

/** The model input of the indices of a split, in the layout of `spec.inputShape`. */
export function fillInputs(
  split: LoadedSplit,
  indices: ArrayLike<number>,
  spec: ModelSpec,
  out: Float32Array,
): Float32Array {
  if (spec.preset === "bag")
    return bagOfWordsBatch(split, indices, spec.inputShape[0]!, out);
  if (spec.preset === "regression") {
    if (!spec.features)
      throw new Error("A regression dataset needs its feature normalisation.");
    return tabularBatch(
      split,
      indices,
      spec.features.mean,
      spec.features.std,
      out,
    );
  }
  return imageBatch(split, indices, out);
}

/** Evaluate on the full test split, in batches of the device dataset, with the model in eval mode and no backward. */
export async function evaluate(
  model: Module,
  device: Device,
  data: LoadedDataset,
  dataset: DeviceDataset,
  spec: ModelSpec,
  scale: TargetScale | undefined,
  batchSize = 256,
  maxBatches?: number,
): Promise<{ metrics: Metrics; samples: number }> {
  void device;
  model.eval();
  const parameters = model.parameters();
  const split = data.test;
  const N = split.count;
  const limit =
    maxBatches === undefined ? N : Math.min(N, maxBatches * batchSize);
  let correct = 0;
  let squared = 0;
  let absolute = 0;
  let start = 0;
  try {
    for (const batch of dataset.evaluation(batchSize)) {
      if (start >= limit) break;
      const n = Math.min(batch.size, limit - start);
      const logits = model.forward(batch.input);
      const values = await logits.data();
      if (scale) {
        for (let k = 0; k < n; k++) {
          const predicted = values[k]! * scale.std + scale.mean;
          const error = predicted - split.y[start + k]!;
          squared += error * error;
          absolute += Math.abs(error);
        }
      } else {
        const outputs = spec.outputs;
        for (let k = 0; k < n; k++) {
          let best = 0;
          for (let c = 1; c < outputs; c++)
            if (values[k * outputs + c]! > values[k * outputs + best]!)
              best = c;
          if (best === split.y[start + k]) correct++;
        }
      }
      await logits.disposeGraph(parameters);
      start += batch.size;
      await YIELD();
    }
  } finally {
    model.train();
  }
  if (scale)
    return {
      metrics: { rmse: Math.sqrt(squared / limit), mae: absolute / limit },
      samples: limit,
    };
  return { metrics: { accuracy: correct / limit }, samples: limit };
}

/**
 * Run the model on one input in eval mode. For `probabilities`, the output is softmax over the
 * classes. For `value`, it is the raw output (the regression target in standardised units).
 */
export async function predict(
  model: Module,
  device: Device,
  input: { readonly values: Float32Array; readonly shape: readonly number[] },
  kind: "probabilities" | "value",
): Promise<Float32Array> {
  model.eval();
  const parameters = model.parameters();
  try {
    const tensor = device.tensor(input.values, { shape: input.shape });
    let output = model.forward(tensor);
    if (kind === "probabilities") output = output.softmax(1);
    const values = await output.data();
    await output.disposeGraph(parameters);
    return values;
  } finally {
    model.train();
  }
}
