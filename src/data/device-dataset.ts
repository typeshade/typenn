import { resident, type Resident } from "typeshade/runtime";
import type { Device } from "../runtime/device.js";
import {
  bagOfWords,
  gatherFirst,
  gatherNext,
} from "../kernels/dataset.shade.ts";
import type { OpNode } from "../tensor/node.js";
import { Tensor } from "../tensor/tensor.js";
import {
  DEFAULT_SHARD_BYTES,
  elementsPerWord,
  rowsPerShard,
} from "./shards.js";
import { hostBagOfWords, hostFeatures, hostLabels } from "./host.js";

/** A source array of one split: unsigned integers (u8, u16, u32) or f32. */
export type SourceArray = Uint8Array | Uint16Array | Uint32Array | Float32Array;

/** One split of a dataset, as the host gives it. The arrays are read, not copied, until upload. */
export interface DatasetSplit {
  /** The number of samples. */
  readonly count: number;
  /**
   * The inputs: `count * features` values for images (u8) and tabular data (f32), or the token
   * ids of all samples for text (u16 or u32), with `offsets` bounding each sample.
   */
  readonly x: SourceArray;
  /**
   * One value per sample: integer class ids (u8, u16 or u32, each below 2^24), or f32 regression
   * targets.
   */
  readonly y: SourceArray;
  /** Text only: `count + 1` token offsets, starting at 0. Sample `s` is `x[offsets[s]..offsets[s+1])`. */
  readonly offsets?: Uint32Array;
}

/** How the inputs of a dataset are read. */
export type DatasetInputs =
  | {
      /** u8 pixels, scaled to [0, 1] as `x / 255`. */
      readonly kind: "images";
      /** Values per sample. */
      readonly features: number;
      /** The batch shape of the inputs, whose product is `features`. Default `[features]`. */
      readonly shape?: readonly number[];
    }
  | {
      /** f32 features, normalised as `(x - mean) / std`. A zero std is read as 1. */
      readonly kind: "tabular";
      readonly features: number;
      readonly mean: ArrayLike<number>;
      readonly std: ArrayLike<number>;
      readonly shape?: readonly number[];
    }
  | {
      /** Text: multi-hot bag of words over the first `vocab` token ids. */
      readonly kind: "text";
      readonly vocab: number;
    };

/** The options of `Device.dataset`. */
export interface DeviceDatasetOptions {
  readonly inputs: DatasetInputs;
  readonly train: DatasetSplit;
  readonly test: DatasetSplit;
  /**
   * Regression targets only: each target is `(y - mean) / std`, with a zero std read as 1. A
   * regression batch label has shape `[size, 1]`.
   */
  readonly target?: { readonly mean: number; readonly std: number };
  /**
   * The largest shard of a source, in bytes. The default is the device's storage binding limit
   * when TypeShade exposes it, and `DEFAULT_SHARD_BYTES` otherwise. A test sets it small.
   */
  readonly shardBytes?: number;
}

/** One batch: the inputs and the labels, as Tensors of the lazy graph. */
export interface DeviceBatch {
  /** The model inputs, shape `[size, ...inputShape]` (or `[size, vocab]` for text). */
  readonly input: Tensor;
  /** Class ids as f32 values, shape `[size]`; or regression targets, shape `[size, 1]`. */
  readonly labels: Tensor;
  /** The number of samples in the batch. */
  readonly size: number;
}

/**
 * Where the dataset keeps its data. `device` means the batches are built on the device from the
 * uploaded sources. `host` means each batch is built on the host and uploaded, with `reason` the
 * cause. The `shards` counts are the number of device shards of each split's input field.
 */
export interface Residency {
  readonly tier: "device" | "host";
  readonly reason?: string;
  readonly shards: { readonly train: number; readonly test: number };
}

/** A packed or f32 field of a split: its shards, and the normalisation. */
interface Field {
  readonly features: number;
  readonly width: 8 | 16 | 32;
  readonly perWord: number;
  readonly isFloat: 0 | 1;
  readonly shards: readonly {
    readonly first: number;
    readonly rows: number;
    readonly words: Resident<Uint32Array>;
  }[];
  readonly mean: Resident<Float32Array>;
  readonly std: Resident<Float32Array>;
}

/** The device state of one split. */
interface DeviceSplitState {
  readonly input:
    | Field
    | {
        readonly tokens: Resident<Uint32Array>;
        readonly perWord: number;
        readonly width: 16 | 32;
        readonly offsets: Resident<Uint32Array>;
      };
  readonly labels: Field;
  /** The row order of the split: the permutation the batches are taken from. */
  readonly perm: Resident<Uint32Array>;
}

/** The split as the dataset holds it: the host copy, the order, and the device state if any. */
interface SplitState {
  readonly split: DatasetSplit;
  /** The order of the samples. The test split keeps the identity order. */
  readonly order: Uint32Array;
  readonly device?: DeviceSplitState;
}

/** The seeded generator of `shuffle`: the mulberry32 generator of the lab, with the same sequence. */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function bitsOf(values: SourceArray): 8 | 16 | 32 {
  if (values instanceof Uint8Array) return 8;
  if (values instanceof Uint16Array) return 16;
  return 32;
}

/**
 * The packed u32 words of `count` values of `width` bits, from element `first * features` on. A
 * float source keeps its f32 bits.
 */
function packWords(
  values: SourceArray,
  first: number,
  rows: number,
  features: number,
  width: 8 | 16 | 32,
): Uint32Array {
  const count = rows * features;
  const perWord = elementsPerWord(width);
  if (values instanceof Float32Array) {
    const words = new Uint32Array(count);
    words.set(
      new Uint32Array(
        values.buffer,
        values.byteOffset + first * features * 4,
        count,
      ),
    );
    return words;
  }
  const words = new Uint32Array(Math.ceil(count / perWord));
  const start = first * features;
  for (let e = 0; e < count; e++) {
    const value = values[start + e]!;
    const word = Math.floor(e / perWord);
    const shift = (e % perWord) * width;
    words[word] = (words[word]! | ((value << shift) >>> 0)) >>> 0;
  }
  return words;
}

/** Validate the options; throw a RangeError or TypeError that names the problem. */
function validate(options: DeviceDatasetOptions): void {
  const { inputs, train, test } = options;
  for (const [name, split] of [
    ["train", train],
    ["test", test],
  ] as const) {
    if (!Number.isSafeInteger(split.count) || split.count < 1)
      throw new RangeError(`The ${name} split needs at least one sample.`);
    if (split.y.length !== split.count)
      throw new RangeError(
        `The ${name} split has ${split.y.length} labels for ${split.count} samples.`,
      );
    if (inputs.kind === "text") {
      if (
        split.offsets === undefined ||
        split.offsets.length !== split.count + 1 ||
        split.offsets[0] !== 0
      )
        throw new RangeError(
          `The ${name} text split needs count + 1 offsets, starting at 0.`,
        );
      for (let s = 0; s < split.count; s++)
        if (
          split.offsets[s + 1]! < split.offsets[s]! ||
          split.offsets[s + 1]! > split.x.length
        )
          throw new RangeError(
            `The ${name} text split has offsets outside its tokens.`,
          );
    } else {
      const features = inputs.features;
      if (!Number.isSafeInteger(features) || features < 1)
        throw new RangeError(
          "The features of a dataset input must be a positive safe integer.",
        );
      if (split.x.length !== split.count * features)
        throw new RangeError(
          `The ${name} split has ${split.x.length} inputs; ${split.count} samples of ${features} need ${split.count * features}.`,
        );
    }
    if (inputs.kind === "images" && !(split.x instanceof Uint8Array))
      throw new TypeError("Image inputs are u8 pixels (a Uint8Array).");
    if (inputs.kind === "tabular" && !(split.x instanceof Float32Array))
      throw new TypeError("Tabular inputs are f32 features (a Float32Array).");
    if (
      inputs.kind === "text" &&
      (split.x instanceof Float32Array || split.x instanceof Uint8Array)
    )
      throw new TypeError(
        "Text inputs are token ids (a Uint16Array or Uint32Array).",
      );
    if (!(split.y instanceof Float32Array)) {
      for (const label of split.y) {
        if (!Number.isSafeInteger(label) || label < 0 || label >= 2 ** 24)
          throw new RangeError(
            `Class id ${label} is not an integer in [0, 2^24) (exact as f32).`,
          );
      }
    }
  }
  if (
    inputs.kind === "text" &&
    (!Number.isSafeInteger(inputs.vocab) || inputs.vocab < 1)
  )
    throw new RangeError("A text dataset needs a positive integer vocab.");
  if (inputs.kind === "tabular") {
    if (
      inputs.mean.length !== inputs.features ||
      inputs.std.length !== inputs.features
    )
      throw new RangeError("Tabular mean and std need one value per feature.");
  }
  if (
    options.shardBytes !== undefined &&
    (!Number.isSafeInteger(options.shardBytes) || options.shardBytes < 4)
  )
    throw new RangeError("shardBytes is a whole number of bytes, at least 4.");
}

/**
 * A dataset held on a device. The sources are uploaded once, in shards of the device's binding
 * limit, and every batch is gathered on the device by a kernel. A training step uploads no sample
 * bytes. The only upload is the order of an epoch, `N` u32 values (4N bytes), made by `shuffle()`
 * or `order()`.
 *
 * Batches are Tensors of the lazy graph: `input` and `labels` run when the model reads them. Call
 * `dispose()` to free the device memory. On a device that holds no storage arrays, or for a source
 * that cannot fit a shard, the dataset builds the batches on the host (see `residency`).
 */
export class DeviceDataset {
  readonly inputs: DatasetInputs;
  readonly residency: Residency;
  /** The shape of one batch input, without the batch axis. */
  readonly inputShape: readonly number[];
  readonly trainCount: number;
  readonly testCount: number;
  private readonly splits: {
    readonly train: SplitState;
    readonly test: SplitState;
  };
  private readonly handles: {
    readonly resident: Resident<unknown>;
    readonly bytes: number;
  }[] = [];
  private readonly labelShape: (count: number) => number[];
  private readonly target?: { readonly mean: number; readonly std: number };
  private disposed = false;

  /** @internal Use `Device.dataset`. */
  constructor(
    private readonly device: Device,
    options: DeviceDatasetOptions,
  ) {
    validate(options);
    const { inputs } = options;
    this.inputs = inputs;
    this.trainCount = options.train.count;
    this.testCount = options.test.count;
    this.target = options.target;
    const regression = options.train.y instanceof Float32Array;
    this.labelShape = (count) => (regression ? [count, 1] : [count]);
    this.inputShape =
      inputs.kind === "text"
        ? [inputs.vocab]
        : [...(inputs.shape ?? [inputs.features])];
    if (
      inputs.kind !== "text" &&
      this.inputShape.reduce((a, b) => a * b, 1) !== inputs.features
    )
      throw new RangeError(
        "The input shape must have as many values as the features.",
      );

    const shardBytes =
      options.shardBytes ?? device.storageBindingLimit() ?? DEFAULT_SHARD_BYTES;
    const reason =
      device.backend.name === "webgl2"
        ? "The WebGL2 tier has no storage arrays for the gather kernels, so batches are built on the host."
        : this.fitReason(options, shardBytes);
    if (reason === undefined) {
      try {
        const train = this.uploadSplit(
          options.train,
          options,
          shardBytes,
          "train",
        );
        const test = this.uploadSplit(
          options.test,
          options,
          shardBytes,
          "test",
        );
        this.splits = { train, test };
        this.residency = {
          tier: "device",
          shards: { train: shardCount(train), test: shardCount(test) },
        };
        return;
      } catch (error) {
        for (const { resident: handle } of this.handles.splice(0))
          handle.destroy();
        throw error;
      }
    }
    this.splits = {
      train: { split: options.train, order: identity(options.train.count) },
      test: { split: options.test, order: identity(options.test.count) },
    };
    this.residency = { tier: "host", reason, shards: { train: 0, test: 0 } };
  }

  /** The device bytes this dataset holds: the uploaded sources, orders and normalisation. */
  get bytes(): number {
    return this.handles.reduce((total, handle) => total + handle.bytes, 0);
  }

  /**
   * Set a new random order of the train split from `seed`, a Fisher-Yates shuffle. The host
   * uploads one permutation of `trainCount` u32 values for it.
   */
  shuffle(seed: number): void {
    this.assertUsable();
    const order = identity(this.trainCount);
    const random = mulberry32(seed);
    for (let i = this.trainCount - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [order[i], order[j]] = [order[j]!, order[i]!];
    }
    this.order(order);
  }

  /**
   * Set the order of the train split: `indices` must hold each sample index once. The host
   * uploads one permutation of `trainCount` u32 values for it.
   */
  order(indices: ArrayLike<number>): void {
    this.assertUsable();
    if (indices.length !== this.trainCount)
      throw new RangeError(
        `The order has ${indices.length} indices; the train split has ${this.trainCount} samples.`,
      );
    const seen = new Uint8Array(this.trainCount);
    const order = new Uint32Array(this.trainCount);
    for (let k = 0; k < this.trainCount; k++) {
      const index = indices[k]!;
      if (
        !Number.isSafeInteger(index) ||
        index < 0 ||
        index >= this.trainCount ||
        seen[index]
      )
        throw new RangeError(
          "The order must be a permutation of the train sample indices.",
        );
      seen[index] = 1;
      order[k] = index;
    }
    this.splits.train.order.set(order);
    this.splits.train.device?.perm.write(order);
  }

  /**
   * The batch of the train split at positions `offset` to `offset + count` of the current order.
   * The batch holds fewer samples at the end of an epoch.
   */
  batch(offset: number, count: number): DeviceBatch {
    return this.makeBatch(this.splits.train, offset, count);
  }

  /** The batches of one epoch over the train split in the current order, all but the last of `batchSize`. */
  *batches(batchSize: number): Generator<DeviceBatch> {
    for (let offset = 0; offset < this.trainCount; offset += batchSize)
      yield this.batch(offset, Math.min(batchSize, this.trainCount - offset));
  }

  /** The batches of the test split, in order. */
  *evaluation(batchSize: number): Generator<DeviceBatch> {
    this.assertUsable();
    if (!Number.isSafeInteger(batchSize) || batchSize < 1)
      throw new RangeError("batchSize must be a positive safe integer.");
    for (let offset = 0; offset < this.testCount; offset += batchSize)
      yield this.makeBatch(
        this.splits.test,
        offset,
        Math.min(batchSize, this.testCount - offset),
      );
  }

  /** Free every device array of the dataset. A later use throws. Tensors built earlier are not usable. */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const { resident: handle } of this.handles.splice(0)) handle.destroy();
  }

  private makeBatch(
    state: SplitState,
    offset: number,
    count: number,
  ): DeviceBatch {
    this.assertUsable();
    if (
      !Number.isSafeInteger(offset) ||
      !Number.isSafeInteger(count) ||
      offset < 0 ||
      count < 1 ||
      offset + count > state.split.count
    )
      throw new RangeError(
        `The batch [${offset}, ${offset + count}) is outside the split.`,
      );
    const device = state.device;
    if (device === undefined) return this.hostBatch(state, offset, count);
    const shape = (rows: number, rest: readonly number[]) => [rows, ...rest];
    const input =
      "tokens" in device.input
        ? this.bagChain(device.input, device.perm, offset, count)
        : this.gatherChain(
            device.input,
            device.perm,
            offset,
            shape(count, this.inputShape),
          );
    const labels = this.gatherChain(
      device.labels,
      device.perm,
      offset,
      this.labelShape(count),
    );
    return { input, labels, size: count };
  }

  /** The gather of a field: one kernel call per shard, each changing only the rows of its shard. */
  private gatherChain(
    field: Field,
    perm: Resident<Uint32Array>,
    start: number,
    shape: readonly number[],
  ): Tensor {
    let previous: Tensor | undefined;
    for (const shard of field.shards) {
      const node = {
        kind: "op",
        op: "gather",
        kernel: previous === undefined ? gatherFirst : gatherNext,
        inputs: previous === undefined ? [] : [previous],
        arrays: [perm, shard.words, field.mean, field.std],
        values: [
          start,
          shard.first,
          shard.rows,
          field.features,
          field.perWord,
          field.width,
          field.isFloat,
        ],
      } as const satisfies OpNode;
      previous = this.device.createOperation(shape, node as OpNode);
    }
    return previous!;
  }

  private bagChain(
    text: {
      readonly tokens: Resident<Uint32Array>;
      readonly perWord: number;
      readonly width: 16 | 32;
      readonly offsets: Resident<Uint32Array>;
    },
    perm: Resident<Uint32Array>,
    start: number,
    count: number,
  ): Tensor {
    const vocab = (this.inputs as { vocab: number }).vocab;
    return this.device.createOperation([count, vocab], {
      kind: "op",
      op: "bagOfWords",
      kernel: bagOfWords,
      inputs: [],
      arrays: [perm, text.tokens, text.offsets],
      values: [start, vocab, text.perWord, text.width],
    });
  }

  private hostBatch(
    state: SplitState,
    offset: number,
    count: number,
  ): DeviceBatch {
    const indices = state.order.subarray(offset, offset + count);
    const { split } = state;
    const input =
      this.inputs.kind === "text"
        ? hostBagOfWords(split.x, split.offsets!, indices, this.inputs.vocab)
        : hostFeatures(
            split.x,
            indices,
            this.inputs.features,
            this.normalisation().mean,
            this.normalisation().std,
          );
    const labels = hostLabels(split.y, indices, this.target);
    return {
      input: this.device.tensor(input, { shape: [count, ...this.inputShape] }),
      labels: this.device.tensor(labels, { shape: this.labelShape(count) }),
      size: count,
    };
  }

  private normalisation(): { mean: ArrayLike<number>; std: ArrayLike<number> } {
    const inputs = this.inputs;
    if (inputs.kind === "images") {
      return {
        mean: new Float32Array(inputs.features),
        std: new Float32Array(inputs.features).fill(255),
      };
    }
    if (inputs.kind === "tabular") {
      return {
        mean: inputs.mean,
        std: Array.from(inputs.std, (s) => (s === 0 ? 1 : s)),
      };
    }
    return { mean: [], std: [] };
  }

  /** Why the sources cannot all be device resident, or undefined when they can. */
  private fitReason(
    options: DeviceDatasetOptions,
    shardBytes: number,
  ): string | undefined {
    if (options.inputs.kind === "text") {
      for (const split of [options.train, options.test]) {
        const perWord = elementsPerWord(
          split.x instanceof Uint16Array ? 16 : 32,
        );
        if (
          Math.ceil(split.x.length / perWord) * 4 > shardBytes ||
          split.offsets!.byteLength > shardBytes
        )
          return "A text source is larger than one shard. Text is not sharded, so batches are built on the host.";
      }
    } else {
      const features = options.inputs.features;
      const inputWidth = options.inputs.kind === "images" ? 8 : 32;
      if (rowsPerShard(features, elementsPerWord(inputWidth), shardBytes) < 1)
        return "One sample is larger than a shard, so batches are built on the host.";
    }
    return undefined;
  }

  private uploadSplit(
    split: DatasetSplit,
    options: DeviceDatasetOptions,
    shardBytes: number,
    name: "train" | "test",
  ): SplitState {
    const { inputs } = options;
    const order = identity(split.count);
    const perm = this.keep(order);
    let input: DeviceSplitState["input"];
    if (inputs.kind === "text") {
      const width = split.x instanceof Uint16Array ? 16 : 32;
      const perWord = elementsPerWord(width);
      const tokens = new Uint32Array(Math.ceil(split.x.length / perWord));
      for (let t = 0; t < split.x.length; t++) {
        const word = Math.floor(t / perWord);
        tokens[word] =
          (tokens[word]! | ((split.x[t]! << ((t % perWord) * width)) >>> 0)) >>>
          0;
      }
      input = {
        tokens: this.keep(tokens),
        perWord,
        width,
        offsets: this.keep(split.offsets!),
      };
    } else {
      const width = inputs.kind === "images" ? 8 : 32;
      const normal = this.normalisation();
      input = this.field(
        split.x,
        split.count,
        inputs.features,
        width,
        normal.mean,
        normal.std,
        shardBytes,
        name,
      );
    }
    const labelWidth = bitsOf(split.y);
    const regression = split.y instanceof Float32Array;
    const mean = regression ? [this.target?.mean ?? 0] : [0];
    const std = regression ? [this.target?.std || 1] : [1];
    const labels = this.field(
      split.y,
      split.count,
      1,
      regression ? 32 : labelWidth,
      mean,
      std,
      shardBytes,
      name,
    );
    return {
      split,
      order,
      device: { input, labels, perm },
    };
  }

  /** The shards of a field, uploaded: the words of each shard and the normalisation. */
  private field(
    values: SourceArray,
    count: number,
    features: number,
    width: 8 | 16 | 32,
    mean: ArrayLike<number>,
    std: ArrayLike<number>,
    shardBytes: number,
    _name: string,
  ): Field {
    const perWord = elementsPerWord(width);
    const rows = rowsPerShard(features, perWord, shardBytes);
    if (rows < 1) throw new RangeError("A sample does not fit a shard.");
    const shards: {
      first: number;
      rows: number;
      words: Resident<Uint32Array>;
    }[] = [];
    for (let first = 0; first < count; first += rows) {
      const shardRows = Math.min(rows, count - first);
      shards.push({
        first,
        rows: shardRows,
        words: this.keep(packWords(values, first, shardRows, features, width)),
      });
    }
    const meanArray = Float32Array.from(mean);
    const stdArray = Float32Array.from(std, (s) => (s === 0 ? 1 : s));
    return {
      features,
      width,
      perWord,
      isFloat: values instanceof Float32Array ? 1 : 0,
      shards,
      mean: this.keep(meanArray),
      std: this.keep(stdArray),
    };
  }

  private keep<T extends Uint32Array | Float32Array>(array: T): Resident<T> {
    const handle = resident(array);
    this.handles.push({
      resident: handle as Resident<unknown>,
      bytes: array.byteLength,
    });
    return handle;
  }

  private assertUsable(): void {
    if (this.disposed) throw new Error("DeviceDataset has been disposed.");
    this.device.assertUsable();
  }
}

/** The number of shards of the input field of a split, for `residency`. */
function shardCount(state: SplitState): number {
  const input = state.device?.input;
  return input !== undefined && "shards" in input ? input.shards.length : 1;
}

function identity(count: number): Uint32Array {
  const order = new Uint32Array(count);
  for (let i = 0; i < count; i++) order[i] = i;
  return order;
}
