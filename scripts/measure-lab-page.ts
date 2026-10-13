// Runs in the page of scripts/measure-lab.mjs: one training run of the Lab's train loop on a
// synthetic dataset with the shape of a Lab dataset, on WebGPU. It counts the bytes the page
// uploads to the GPU queue (queue.writeBuffer, and mappedAtCreation buffers) and times each step.
// The values of the data do not change a step's cost, so random bytes stand in for the dataset.
// Not part of the library: it is bundled only for the measurement.

export interface BenchOptions {
  readonly dataset: "mnist" | "cifar10";
  /** False skips the one-time upload probe (a diagnosis). */
  readonly probe?: boolean;
  readonly steps: number;
  readonly warmup: number;
}

const PRESETS = {
  mnist: {
    shape: [28, 28, 1],
    train: 60000,
    test: 10000,
    model: "mlp",
    hidden: 128,
    batchSize: 64,
    learningRate: 0.001,
  },
  cifar10: {
    shape: [32, 32, 3],
    train: 50000,
    test: 10000,
    model: "cnn",
    hidden: 128,
    batchSize: 64,
    learningRate: 0.001,
  },
} as const;

/** Bytes the page has uploaded to the GPU queue so far. */
let uploaded = 0;

function countUploads(): void {
  const write = GPUQueue.prototype.writeBuffer;
  GPUQueue.prototype.writeBuffer = function (
    this: GPUQueue,
    ...args: [GPUBuffer, number, unknown, number?, number?]
  ) {
    const [, , data, dataOffset, size] = args;
    const view = ArrayBuffer.isView(data) ? data : null;
    const perElement = view ? view.BYTES_PER_ELEMENT : 1;
    const total = view
      ? view.byteLength / perElement
      : (data as ArrayBuffer).byteLength;
    const elements = size ?? total - (dataOffset ?? 0);
    uploaded += elements * perElement;
    return (write as (...values: unknown[]) => void).apply(this, args);
  } as typeof write;
  const create = GPUDevice.prototype.createBuffer;
  GPUDevice.prototype.createBuffer = function (
    this: GPUDevice,
    descriptor: GPUBufferDescriptor,
  ) {
    if (descriptor.mappedAtCreation) uploaded += descriptor.size;
    return create.call(this, descriptor);
  };
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 0x100000000;
  };
}

function syntheticDataset(
  preset: (typeof PRESETS)[keyof typeof PRESETS],
  id: string,
) {
  const size = preset.shape.reduce((a, b) => a * b, 1);
  const random = mulberry32(1);
  const split = (count: number) => {
    const x = new Uint8Array(count * size);
    for (let i = 0; i < x.length; i++) x[i] = Math.floor(random() * 256);
    const y = new Uint8Array(count);
    for (let i = 0; i < count; i++) y[i] = Math.floor(random() * 10);
    return { count, x, y };
  };
  const field = (dtype: string, shape: number[]) => ({
    dtype,
    shape,
    parts: [],
  });
  const splitRef = (count: number) => ({
    count,
    x: field("uint8", preset.shape),
    y: field("uint8", []),
  });
  return {
    info: {
      id,
      title: id,
      task: "image-classification",
      totalBytes: 0,
      manifest: "",
      origin: "mirror",
    },
    manifest: {
      id,
      title: id,
      task: "image-classification",
      license: "",
      citation: "",
      classes: Array.from({ length: 10 }, (_, i) => String(i)),
      splits: { train: splitRef(preset.train), test: splitRef(preset.test) },
    },
    train: split(preset.train),
    test: split(preset.test),
  };
}

/** Runs in the page: returns the measurements of one run. */
export async function bench(
  lab: any,
  nn: any,
  options: BenchOptions,
): Promise<Record<string, unknown>> {
  countUploads();
  const preset = PRESETS[options.dataset];
  const data = syntheticDataset(preset, options.dataset);
  const config = {
    dataset: options.dataset,
    model: preset.model,
    hidden: preset.hidden,
    optimizer: "adam",
    learningRate: preset.learningRate,
    batchSize: preset.batchSize,
    epochs: 1,
    seed: 1,
    tier: "webgpu",
    maxSteps: options.warmup + options.steps,
    maxEvalBatches: 1,
  };
  const device = await nn.device({ device: "webgpu" });
  if (device.backend.name !== "webgpu")
    throw new Error(`The tier is ${device.backend.name}, not webgpu.`);
  const adapter = await navigator.gpu.requestAdapter();
  const adapterInfo = adapter?.info ?? null;

  // The one-time upload of the dataset. Only the device-resident build has one. The sources upload
  // lazily, on the first read of a batch, so the probe reads one batch of each split.
  let oneTime: Record<string, unknown> | null = null;
  if (typeof lab.datasetOptions === "function" && options.probe !== false) {
    const spec = lab.specFor(data, config);
    const start = uploaded;
    const t0 = performance.now();
    const probeDataset = device.dataset(
      lab.datasetOptions(data, spec, undefined),
    );
    // The sources upload when a batch first reads them: one train batch and one test batch force
    // every shard of both splits.
    await probeDataset.batch(0, 1).input.data();
    for (const batch of probeDataset.evaluation(1)) {
      await batch.input.data();
      break;
    }
    oneTime = {
      seconds: (performance.now() - t0) / 1000,
      bytes: uploaded - start,
      residency: probeDataset.residency,
      heldBytes: probeDataset.bytes,
    };
    probeDataset.dispose();
  }

  const times: number[] = [];
  const upAt: number[] = [];
  let atDataset = uploaded;
  let ownDataset: { dispose(): void } | undefined;
  let residency: unknown = undefined;
  let heldBytes: number | undefined;
  const started = uploaded;
  const result = await lab.train(config, data, nn, device, {
    onDataset: (d: any) => {
      ownDataset = d;
      atDataset = uploaded;
      residency = d.residency;
      heldBytes = d.bytes;
    },
    onStep: (info: { elapsedSeconds: number }) => {
      times.push(info.elapsedSeconds);
      upAt.push(uploaded);
    },
  });
  ownDataset?.dispose();

  const deltas: number[] = [];
  for (let k = options.warmup + 1; k <= options.warmup + options.steps; k++)
    deltas.push((times[k - 1]! - times[k - 2]!) * 1000);
  const sorted = [...deltas].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)]!;
  const mean = deltas.reduce((a, b) => a + b, 0) / deltas.length;
  const windowBytes =
    upAt[options.warmup + options.steps - 1]! - upAt[options.warmup - 1]!;
  return {
    dataset: options.dataset,
    steps: deltas.length,
    warmup: options.warmup,
    stepsRun: result.steps,
    medianMs: median,
    meanMs: mean,
    minMs: sorted[0],
    maxMs: sorted[sorted.length - 1],
    bytesPerStep: windowBytes / options.steps,
    epochOrderBytes: upAt[0]! - atDataset,
    firstStepBytes: upAt[0]! - started,
    uploadDeltas: upAt
      .slice(0, 6)
      .map((value, i) => value - (i === 0 ? started : upAt[i - 1]!)),
    datasetUploadWindow: atDataset - started,
    oneTime,
    residency: residency ?? {
      tier: "host",
      reason: "no device dataset in this build",
    },
    heldBytes: heldBytes ?? 0,
    adapter: adapterInfo
      ? {
          vendor: adapterInfo.vendor,
          architecture: adapterInfo.architecture,
          description: adapterInfo.description,
        }
      : null,
    typennCommit: result.typennCommit,
    tier: device.backend.name,
    userAgent: navigator.userAgent,
  };
}
