import test, { after } from "node:test";
import assert from "node:assert/strict";
import { fakeGpu } from "./_fake-gpu.mjs";
import { closeNn, loadNn } from "./_nn.mjs";

// Device-resident datasets. The batches of a dataset are built on the device from the uploaded
// sources; these tests hold each batch to a host reference written from the formulas of the
// batch helpers (images /255, normalisation with a zero std read as 1, multi-hot bags), bit for
// bit on the CPU tier. A tiny `shardBytes` forces several shards, so a batch spans a boundary.

const nn = await loadNn();
after(closeNn);

/** A deterministic integer sequence in [0, limit). */
function ints(count, limit, seed = 1) {
  let state = seed >>> 0;
  return Array.from({ length: count }, () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state % limit;
  });
}

/** The first `count` values of the order, a permutation of `0..n-1` built from `seed`. */
function permutation(n, seed) {
  const order = Array.from({ length: n }, (_, i) => i);
  const random = ints(n, 2 ** 30, seed);
  for (let i = n - 1; i > 0; i--) {
    const j = random[i] % (i + 1);
    [order[i], order[j]] = [order[j], order[i]];
  }
  return order;
}

/** Reference images: `x / 255` for each pixel of the indexed samples. */
const refImages = (x, F, indices) =>
  Float32Array.from(
    indices.flatMap((i) =>
      Array.from(x.subarray(i * F, i * F + F), (v) => v / 255),
    ),
  );

/** Reference tabular: `(x - mean) / std`, a zero std read as 1. */
const refTabular = (x, F, indices, mean, std) =>
  Float32Array.from(
    indices.flatMap((i) =>
      Array.from(
        { length: F },
        (_, f) => (x[i * F + f] - mean[f]) / (std[f] === 0 ? 1 : std[f]),
      ),
    ),
  );

/** Reference multi-hot bags over the first `vocab` ids. */
function refBags(x, offsets, indices, vocab) {
  const out = new Float32Array(indices.length * vocab);
  indices.forEach((i, k) => {
    for (let t = offsets[i]; t < offsets[i + 1]; t++)
      if (x[t] < vocab) out[k * vocab + x[t]] = 1;
  });
  return out;
}

async function values(tensor) {
  return Array.from(await tensor.data());
}

/** Build every batch of an epoch and check it against the reference of the same indices. */
async function checkEpoch(
  dataset,
  batchSize,
  expectInput,
  expectLabels,
  order,
) {
  dataset.order(order);
  const N = order.length;
  let seen = 0;
  for (const batch of dataset.batches(batchSize)) {
    const indices = order.slice(seen, seen + batch.size);
    assert.deepEqual(batch.input.shape.slice(0, 1), [indices.length]);
    assert.deepEqual(
      await values(batch.input),
      Array.from(expectInput(indices)),
    );
    assert.deepEqual(
      await values(batch.labels),
      Array.from(expectLabels(indices)),
    );
    seen += batch.size;
  }
  assert.equal(seen, N);
}

test("images: device batches equal x/255, across shards, with a short last batch", async () => {
  const device = await nn.device({ device: "cpu" });
  const N = 23,
    F = 6;
  const x = Uint8Array.from(ints(N * F, 256, 3));
  const y = Uint8Array.from(ints(N, 10, 4));
  // 16 bytes is 4 words, so 16 pixels per shard: 4 samples per shard, several shards.
  const dataset = device.dataset({
    inputs: { kind: "images", features: F },
    train: { count: N, x, y },
    test: { count: N, x, y },
    shardBytes: 16,
  });
  assert.equal(dataset.residency.tier, "device");
  assert.ok(dataset.residency.shards.train > 1);
  const order = permutation(N, 9);
  await checkEpoch(
    dataset,
    5,
    (i) => refImages(x, F, i),
    (i) => Float32Array.from(i, (s) => y[s]),
    order,
  );
  // Eval order: the test split in its own order, and the last short batch (23 = 4*5 + 3).
  const evaluated = [];
  for (const batch of dataset.evaluation(5)) {
    assert.equal(batch.labels.shape[0], batch.size);
    evaluated.push(...(await values(batch.labels)));
  }
  assert.deepEqual(
    evaluated,
    Array.from(y, (v) => v),
  );
  dataset.dispose();
  await device.dispose();
});

test("images: a batch spanning a shard boundary matches the host reference", async () => {
  const device = await nn.device({ device: "cpu" });
  const N = 9,
    F = 3;
  const x = Uint8Array.from(ints(N * F, 256, 5));
  const y = Uint8Array.from(ints(N, 4, 6));
  // 8 bytes is 2 words, so 8 pixels: 2 samples per shard. Samples 0..8 use five shards.
  const dataset = device.dataset({
    inputs: { kind: "images", features: F, shape: [3, 1] },
    train: { count: N, x, y },
    test: { count: N, x, y },
    shardBytes: 8,
  });
  assert.equal(dataset.residency.shards.train, 5);
  dataset.order([8, 0, 4, 1, 7, 2, 6, 3, 5]);
  const batch = dataset.batch(0, 6);
  assert.deepEqual(batch.input.shape, [6, 3, 1]);
  assert.deepEqual(
    await values(batch.input),
    Array.from(refImages(x, F, [8, 0, 4, 1, 7, 2])),
  );
  dataset.dispose();
  await device.dispose();
});

test("tabular: device batches equal (x - mean) / std with a zero std read as 1", async () => {
  const device = await nn.device({ device: "cpu" });
  const N = 11,
    F = 4;
  // Values and normalisation are exact in f32, so the device arithmetic equals the host's.
  const x = Float32Array.from(ints(N * F, 64, 7), (v) => v / 4);
  const y = Uint32Array.from(ints(N, 3, 8));
  const mean = [1.5, -2, 0.25, 4];
  const std = [2, 0, 0.5, 8];
  const dataset = device.dataset({
    inputs: { kind: "tabular", features: F, mean, std },
    train: { count: N, x, y },
    test: { count: N, x, y },
    shardBytes: 24,
  });
  await checkEpoch(
    dataset,
    4,
    (i) => refTabular(x, F, i, mean, std),
    (i) => Float32Array.from(i, (s) => y[s]),
    permutation(N, 11),
  );
  dataset.dispose();
  await device.dispose();
});

test("text: device bags equal the multi-hot reference, ids at or above vocab ignored", async () => {
  const device = await nn.device({ device: "cpu" });
  const docs = [[1, 5, 9], [2], [], [0, 3, 3, 7, 40], [9, 9], [6, 1]];
  const N = docs.length,
    vocab = 8;
  const offsets = new Uint32Array(N + 1);
  docs.forEach((doc, s) => (offsets[s + 1] = offsets[s] + doc.length));
  const x = Uint16Array.from(docs.flat());
  const y = Uint8Array.from([0, 1, 0, 1, 0, 1]);
  const dataset = device.dataset({
    inputs: { kind: "text", vocab },
    train: { count: N, x, y, offsets },
    test: { count: N, x, y, offsets },
  });
  assert.equal(dataset.residency.tier, "device");
  const order = permutation(N, 13);
  await checkEpoch(
    dataset,
    4,
    (i) => refBags(x, offsets, i, vocab),
    (i) => Float32Array.from(i, (s) => y[s]),
    order,
  );
  dataset.dispose();
  await device.dispose();
});

test("labels: u8, u16 and u32 class ids, and f32 regression targets with a scale", async () => {
  const device = await nn.device({ device: "cpu" });
  const N = 7,
    F = 2;
  const x = Uint8Array.from(ints(N * F, 256, 14));
  for (const y of [
    Uint8Array.from(ints(N, 200, 15)),
    Uint16Array.from(ints(N, 60000, 16)),
    Uint32Array.from(ints(N, 2 ** 24, 17)),
  ]) {
    const dataset = device.dataset({
      inputs: { kind: "images", features: F },
      train: { count: N, x, y },
      test: { count: N, x, y },
      shardBytes: 8,
    });
    const order = permutation(N, 19);
    await checkEpoch(
      dataset,
      3,
      (i) => refImages(x, F, i),
      (i) => Float32Array.from(i, (s) => y[s]),
      order,
    );
    dataset.dispose();
  }
  const targets = Float32Array.from([0.5, -1.25, 3, 2, 0, 1, -4]);
  const xs = Float32Array.from(ints(N * 3, 16, 18), (v) => v / 2);
  const dataset = device.dataset({
    inputs: { kind: "tabular", features: 3, mean: [0, 0.5, 1], std: [1, 2, 0] },
    train: { count: N, x: xs, y: targets },
    test: { count: N, x: xs, y: targets },
    target: { mean: 1, std: 4 },
  });
  const order = permutation(N, 20);
  dataset.order(order);
  const batch = dataset.batch(0, 3);
  assert.deepEqual(batch.labels.shape, [3, 1]);
  assert.deepEqual(
    await values(batch.labels),
    order.slice(0, 3).map((s) => (targets[s] - 1) / 4),
  );
  dataset.dispose();
  await device.dispose();
});

test("order: a permutation is validated, and the epoch uploads one u32 permutation", async () => {
  const device = await nn.device({ device: "cpu" });
  const N = 4;
  const x = Uint8Array.from(ints(N * 2, 256, 21));
  const y = Uint8Array.from([0, 1, 2, 3]);
  const dataset = device.dataset({
    inputs: { kind: "images", features: 2 },
    train: { count: N, x, y },
    test: { count: N, x, y },
  });
  assert.throws(() => dataset.order([0, 0, 1, 2]), RangeError);
  assert.throws(() => dataset.order([0, 1, 2]), RangeError);
  assert.throws(() => dataset.batch(3, 2), RangeError);
  dataset.order([3, 2, 1, 0]);
  dataset.dispose();
  await device.dispose();
});

test("loss: device class-id labels give the array form's loss and gradients", async () => {
  const device = await nn.device({ device: "cpu" });
  const rows = 4,
    classes = 5;
  const logits = Float32Array.from(
    ints(rows * classes, 17, 22),
    (v) => v / 4 - 2,
  );
  const labels = [4, 0, 2, 2];
  const deviceLabels = device.tensor(Float32Array.from(labels), {
    shape: [rows],
  });
  for (const reduction of ["none", "sum", "mean"]) {
    const loss = new nn.CrossEntropyLoss({ reduction });
    const input = device.tensor(logits, { shape: [rows, classes] });
    const fromArray = loss.forward(input, labels);
    const fromDevice = loss.forward(input, deviceLabels);
    assert.deepEqual(await values(fromDevice), await values(fromArray));
    const [gradArray, gradDevice] = [
      nn.gradients(
        fromArray,
        [input],
        reduction === "none"
          ? device.tensor(new Float32Array(rows).fill(1), { shape: [rows] })
          : undefined,
      )[0],
      nn.gradients(
        fromDevice,
        [input],
        reduction === "none"
          ? device.tensor(new Float32Array(rows).fill(1), { shape: [rows] })
          : undefined,
      )[0],
    ];
    assert.deepEqual(await values(gradDevice), await values(gradArray));
  }
  await device.dispose();
});

test("loss: MSE takes a device target Tensor and gives the array form's value", async () => {
  const device = await nn.device({ device: "cpu" });
  const prediction = device.tensor([1, 2, 3, 4], { shape: [2, 2] });
  const target = [0.5, 2, 1, 4.5];
  const fromArray = new nn.MSELoss().forward(
    prediction,
    Float32Array.from(target),
  );
  const fromTensor = new nn.MSELoss().forward(
    prediction,
    device.tensor(target, { shape: [2, 2] }),
  );
  assert.deepEqual(await values(fromTensor), await values(fromArray));
  await device.dispose();
});

test("dispose: bytes are freed, a later use throws, and the dataset is not usable", async () => {
  const device = await nn.device({ device: "cpu" });
  const N = 6,
    F = 2;
  const x = Uint8Array.from(ints(N * F, 256, 23));
  const y = Uint8Array.from(ints(N, 3, 24));
  const dataset = device.dataset({
    inputs: { kind: "images", features: F },
    train: { count: N, x, y },
    test: { count: N, x, y },
  });
  // Two splits: the shard words, the normalisation, and each order.
  assert.ok(dataset.bytes > 0);
  dataset.dispose();
  assert.equal(dataset.bytes, 0);
  dataset.dispose();
  assert.throws(() => dataset.batch(0, 2), /disposed/);
  assert.throws(() => dataset.shuffle(1), /disposed/);
  await device.dispose();
});

test("residency: a sample larger than a shard is reported and batches are built on the host", async () => {
  const device = await nn.device({ device: "cpu" });
  const N = 5,
    F = 4;
  const x = Uint8Array.from(ints(N * F, 256, 25));
  const y = Uint8Array.from(ints(N, 3, 26));
  // 4 bytes is one word: a 4-pixel sample of u8 takes one word, so a shard holds one sample. Use
  // shardBytes 2 (below one word) to make no sample fit.
  // A 5-pixel sample takes two words, more than a 4-byte shard: the dataset reports the host tier.
  const F5 = 5;
  const x5 = Uint8Array.from(ints(N * F5, 256, 29));
  const dataset = device.dataset({
    inputs: { kind: "images", features: F5 },
    train: { count: N, x: x5, y },
    test: { count: N, x: x5, y },
    shardBytes: 4,
  });
  assert.equal(dataset.residency.tier, "host");
  assert.match(dataset.residency.reason, /larger than a shard/);
  assert.equal(dataset.bytes, 0);
  dataset.order([4, 3, 2, 1, 0]);
  const batch = dataset.batch(0, 2);
  assert.deepEqual(
    await values(batch.input),
    Array.from(refImages(x5, F5, [4, 3])),
  );
  assert.deepEqual(await values(batch.labels), [y[4], y[3]]);
  dataset.dispose();
  await device.dispose();
});

// Measures the upload bytes of one step, with the default single shard, at two sizes. A step that
// uploads only scalars does not grow with N. The permutation is re-sent on the plans of a step, so
// the bytes grow with N (about 4 bytes per sample). This stays a TODO until the runtime keeps the
// permutation resident (see TODO.md).
const stepUploads = async (N) => {
  const fake = fakeGpu();
  const device = await nn.device({ device: "webgpu", gpuDevice: fake.device });
  const F = 8;
  const x = Uint8Array.from(ints(N * F, 256, 27));
  const y = Uint8Array.from(ints(N, 4, 28));
  const dataset = device.dataset({
    inputs: { kind: "images", features: F },
    train: { count: N, x, y },
    test: { count: N, x, y },
  });
  try {
    await dataset.batch(0, 4).input.data();
    const warm = fake.stats;
    const step = dataset.batch(4, 4);
    await step.input.data();
    await step.labels.data();
    return fake.stats.uploadBytes - warm.uploadBytes;
  } finally {
    dataset.dispose();
    await device.dispose();
  }
};
test(
  "fake WebGPU: a step's upload bytes do not grow with the dataset size",
  { todo: "open: the permutation is re-sent on each plan of a step" },
  async () => {
    const small = await stepUploads(12);
    const large = await stepUploads(1200);
    console.log(
      `fake WebGPU: a step uploads ${small} bytes at N=12 and ${large} bytes at N=1200`,
    );
    assert.ok(
      large < small + 1024,
      `a step uploads ${large} bytes at N=1200, ${small} at N=12`,
    );
  },
);
