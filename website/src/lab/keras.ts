// Browser loader for the five datasets that the page downloads from the Keras storage: MNIST,
// Fashion-MNIST, IMDB, Reuters and California housing. The site mirrors only the CIFAR editions
// (datasets.ts). The source bytes, the SHA-256 table and the preprocessing are those of
// scripts/fetch_datasets.py, which follows the Keras loaders. The pure helpers below need no DOM:
// scripts/test-keras-direct.mjs runs them under Node on the cached source files.
//
// Why the JSON API URL: the XML URL (storage.googleapis.com/tensorflow/...) sends no CORS headers,
// so a browser refuses it. The JSON API URL of the same object sends them (checked in Chromium on
// 2026-10-13) and serves the same bytes. verifySource checks every file against the pinned table.

import type {
  DatasetInfo,
  FieldRef,
  LoadedDataset,
  LoadedSplit,
  LoadOptions,
  Manifest,
  Numeric,
  SplitRef,
  Task,
} from "./datasets";

export type KerasId =
  "mnist" | "fashion_mnist" | "imdb" | "reuters" | "california_housing";

export const KERAS_IDS: readonly KerasId[] = [
  "mnist",
  "fashion_mnist",
  "imdb",
  "reuters",
  "california_housing",
];

const XML_BASE = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/";
const JSON_API = "https://storage.googleapis.com/storage/v1/b/tensorflow/o/";
const CACHE_NAME = "typenn-keras-v1";

const SEED = 113;
const VOCABULARY = 10000;
const INDEX_FROM = 3;
const START_CHAR = 1;
const OOV_CHAR = 2;
const TEST_SPLIT = 0.2;
const IMAGE_PIXELS = 28 * 28;
const CALIFORNIA_FEATURES = 8;

/** The JSON API URL of a file in the Keras storage. This is what the browser fetches. */
export function kerasUrl(name: string): string {
  return `${JSON_API}${encodeURIComponent(`tf-keras-datasets/${name}`)}?alt=media`;
}

/** The canonical URL of a file, as the Keras loaders name it. Shown in the manifest's sources. */
export function kerasCanonicalUrl(name: string): string {
  return XML_BASE + name;
}

/** SHA-256 and byte length of each source file. The values match scripts/fetch_datasets.py. */
export const KERAS_SOURCES: Readonly<
  Record<string, { sha256: string; bytes: number }>
> = {
  "mnist.npz": {
    sha256: "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1",
    bytes: 11490434,
  },
  "train-labels-idx1-ubyte.gz": {
    sha256: "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
    bytes: 29515,
  },
  "train-images-idx3-ubyte.gz": {
    sha256: "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    bytes: 26421880,
  },
  "t10k-labels-idx1-ubyte.gz": {
    sha256: "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
    bytes: 5148,
  },
  "t10k-images-idx3-ubyte.gz": {
    sha256: "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    bytes: 4422102,
  },
  "imdb.npz": {
    sha256: "69664113be75683a8fe16e3ed0ab59fda8886cb3cd7ada244f7d9544e4676b9f",
    bytes: 17464789,
  },
  "imdb_word_index.json": {
    sha256: "027313fe9a320280f3bf0fac5baf58c4a683cf39a7ed9c3d3d3912bde4284cf6",
    bytes: 1641221,
  },
  "reuters.npz": {
    sha256: "d6586e694ee56d7a4e65172e12b3e987c03096cb01eab99753921ef915959916",
    bytes: 2110848,
  },
  "reuters_word_index.json": {
    sha256: "9b56000edc4b34a32e89fd5a23b2605594682c74efcf7d8f4e2a5514b771e633",
    bytes: 550378,
  },
  "california_housing.npz": {
    sha256: "1a2e3a52e0398de6463aebe6f4a8da34fb21fbb6b934cf88c3425e766f2a1a6f",
    bytes: 743530,
  },
};

const FASHION_CLASSES = [
  "T-shirt/top",
  "Trouser",
  "Pullover",
  "Dress",
  "Coat",
  "Sandal",
  "Shirt",
  "Sneaker",
  "Bag",
  "Ankle boot",
];
const TEXT_LICENSE_IMDB =
  "No explicit license that we know of. Released by Maas et al. (2011) with the ACL paper (ai.stanford.edu/~amaas/data/sentiment); cite it.";
const TEXT_LICENSE_REUTERS =
  "Reuters-21578: Reuters Ltd. and Carnegie Group made the collection available for free distribution for research purposes (per its readme). Not a standard license.";

interface Spec {
  readonly id: KerasId;
  readonly title: string;
  readonly task: Task;
  readonly files: readonly string[];
  readonly counts: { readonly train: number; readonly test: number };
  readonly license: string;
  readonly citation: string;
  readonly classes: readonly string[] | null;
  readonly features?: readonly string[];
  readonly target?: string;
  readonly text: boolean;
}

const SPECS: Readonly<Record<KerasId, Spec>> = {
  mnist: {
    id: "mnist",
    title: "MNIST handwritten digits",
    task: "image-classification",
    files: ["mnist.npz"],
    counts: { train: 60000, test: 10000 },
    license:
      "Creative Commons Attribution-Share Alike 3.0 (CC BY-SA 3.0), per the Keras documentation.",
    citation:
      "LeCun, Cortes and Burges, The MNIST database of handwritten digits.",
    classes: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    text: false,
  },
  fashion_mnist: {
    id: "fashion_mnist",
    title: "Fashion-MNIST clothing",
    task: "image-classification",
    files: [
      "train-labels-idx1-ubyte.gz",
      "train-images-idx3-ubyte.gz",
      "t10k-labels-idx1-ubyte.gz",
      "t10k-images-idx3-ubyte.gz",
    ],
    counts: { train: 60000, test: 10000 },
    license:
      "MIT License, Zalando SE (github.com/zalandoresearch/fashion-mnist).",
    citation:
      "Xiao, Rasul and Vollgraf, Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms, 2017.",
    classes: FASHION_CLASSES,
    text: false,
  },
  imdb: {
    id: "imdb",
    title: "IMDB movie reviews (sentiment)",
    task: "text-classification",
    files: ["imdb.npz", "imdb_word_index.json"],
    counts: { train: 25000, test: 25000 },
    license: TEXT_LICENSE_IMDB,
    citation:
      "Maas, Daly, Pham, Huang, Ng and Potts, Learning Word Vectors for Sentiment Analysis, ACL 2011.",
    classes: ["negative", "positive"],
    text: true,
  },
  reuters: {
    id: "reuters",
    title: "Reuters newswire topics",
    task: "text-classification",
    files: ["reuters.npz", "reuters_word_index.json"],
    counts: { train: 8982, test: 2246 },
    license: TEXT_LICENSE_REUTERS,
    citation:
      "Reuters-21578, Text Categorization Test Collection, distribution 1.0 (David D. Lewis).",
    classes: Array.from({ length: 46 }, (_, i) => String(i)),
    text: true,
  },
  california_housing: {
    id: "california_housing",
    title: "California housing prices (regression)",
    task: "regression",
    files: ["california_housing.npz"],
    counts: { train: 16512, test: 4128 },
    license:
      "StatLib dataset, Pace and Barry (1997), as distributed by Keras. We found no explicit license; cite the paper.",
    citation:
      "Pace and Barry, Sparse Spatial Autoregressions, Statistics and Probability Letters, 1997 (StatLib).",
    classes: null,
    features: [
      "longitude",
      "latitude",
      "housing_median_age",
      "total_rooms",
      "total_bedrooms",
      "population",
      "households",
      "median_income",
    ],
    target:
      "median_house_value, in US dollars (checked: values such as 252300.0)",
    text: false,
  },
};

const TEXT_ENCODING = {
  start_char: START_CHAR,
  oov_char: OOV_CHAR,
  index_from: INDEX_FROM,
  num_words: VOCABULARY,
};

/** The dataset entries of the Keras group, in the form the lab lists. Totals are the source bytes. */
export function kerasDatasetInfos(): DatasetInfo[] {
  return KERAS_IDS.map((id) => {
    const spec = SPECS[id];
    const totalBytes = spec.files.reduce(
      (sum, name) => sum + KERAS_SOURCES[name]!.bytes,
      0,
    );
    return {
      id,
      title: spec.title,
      task: spec.task,
      totalBytes,
      manifest: "",
      origin: "keras",
    };
  });
}

export function isKerasDataset(id: string): id is KerasId {
  return (KERAS_IDS as readonly string[]).includes(id);
}

/** A manifest for the cards, before any data is read: counts and text, no parts. */
export function kerasCardManifest(id: KerasId): Manifest {
  const spec = SPECS[id];
  const empty: SplitRef = {
    count: 0,
    x: { dtype: "uint8", shape: [], parts: [] },
    y: { dtype: "uint8", shape: [], parts: [] },
  };
  const split = (count: number): SplitRef => ({
    ...empty,
    count,
    ...(spec.text
      ? { offsets: { dtype: "uint32", shape: [], parts: [] } }
      : {}),
  });
  return {
    id,
    title: spec.title,
    task: spec.task,
    license: spec.license,
    citation: spec.citation,
    classes: spec.classes ? [...spec.classes] : null,
    features: spec.features ? [...spec.features] : undefined,
    target: spec.target,
    vocabulary: spec.text ? VOCABULARY : undefined,
    tokenEncoding: spec.text ? TEXT_ENCODING : undefined,
    sources: spec.files.map((name) => sourceEntry(name)),
    splits: {
      train: split(spec.counts.train),
      test: split(spec.counts.test),
    },
  };
}

function sourceEntry(name: string) {
  return {
    file: name,
    url: kerasCanonicalUrl(name),
    sha256: KERAS_SOURCES[name]!.sha256,
    origin: "Keras dataset storage; SHA-256 pinned 2026-10-13",
  };
}

// ---------------------------------------------------------------- fetch and cache

/** Compare the SHA-256 of a source file with the pinned table. Throws on mismatch. */
export async function verifySource(
  name: string,
  bytes: Uint8Array,
): Promise<void> {
  const expected = KERAS_SOURCES[name];
  if (!expected) throw new Error(`${name}: not a Keras source`);
  const actual = await sha256Hex(bytes);
  if (actual !== expected.sha256)
    throw new Error(`${name}: SHA-256 mismatch (got ${actual})`);
}

export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes as BufferSource);
  return Array.from(new Uint8Array(digest), (b) =>
    b.toString(16).padStart(2, "0"),
  ).join("");
}

async function openCache(): Promise<Cache | undefined> {
  try {
    if (typeof caches === "undefined") return undefined;
    return await caches.open(CACHE_NAME);
  } catch {
    return undefined;
  }
}

/** True when every source of the dataset is in the browser's cache. False without Cache API. */
export async function kerasCached(id: KerasId): Promise<boolean> {
  const cache = await openCache();
  if (!cache) return false;
  for (const name of SPECS[id].files) {
    const hit = await cache.match(kerasUrl(name)).catch(() => undefined);
    if (!hit) return false;
  }
  return true;
}

async function readBody(
  response: Response,
  onChunk: (bytes: number) => void,
): Promise<Uint8Array> {
  if (!response.body) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const parts: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    total += value.byteLength;
    onChunk(value.byteLength);
  }
  const out = new Uint8Array(total);
  let at = 0;
  for (const part of parts) {
    out.set(part, at);
    at += part.byteLength;
  }
  return out;
}

/** One source file: from the Cache API when it is there and verifies, else from the network. */
async function fetchSource(
  name: string,
  onChunk: (bytes: number) => void,
  signal?: AbortSignal,
): Promise<Uint8Array> {
  const url = kerasUrl(name);
  const cache = await openCache();
  const hit = cache ? await cache.match(url).catch(() => undefined) : undefined;
  if (hit) {
    try {
      const bytes = new Uint8Array(await hit.arrayBuffer());
      await verifySource(name, bytes);
      onChunk(bytes.byteLength);
      return bytes;
    } catch {
      await cache?.delete(url).catch(() => undefined);
    }
  }
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
  const bytes = await readBody(response, onChunk);
  await verifySource(name, bytes);
  if (cache) {
    cache
      .put(
        url,
        new Response(bytes.slice(0), {
          headers: { "Content-Type": "application/octet-stream" },
        }),
      )
      .catch(() => undefined);
  }
  return bytes;
}

// ---------------------------------------------------------------- load

export interface KerasDecoded {
  readonly train: LoadedSplit;
  readonly test: LoadedSplit;
  readonly manifest: Manifest;
  readonly vocabulary?: string[];
}

/** Download (or read from the cache), verify and decode one of the five datasets. */
export async function loadKerasDataset(
  id: KerasId,
  options: LoadOptions = {},
): Promise<LoadedDataset> {
  const { onProgress, signal } = options;
  const spec = SPECS[id];
  const total = spec.files.reduce(
    (sum, name) => sum + KERAS_SOURCES[name]!.bytes,
    0,
  );
  let loaded = 0;
  onProgress?.(0, total);
  const bytes = await Promise.all(
    spec.files.map((name) =>
      fetchSource(
        name,
        (n) => {
          loaded += n;
          onProgress?.(loaded, total);
        },
        signal,
      ),
    ),
  );
  const files = new Map(spec.files.map((name, i) => [name, bytes[i]!]));
  const decoded = await decodeKeras(id, files);
  const info = kerasDatasetInfos().find((d) => d.id === id)!;
  return {
    info,
    manifest: decoded.manifest,
    train: decoded.train,
    test: decoded.test,
    vocabulary: decoded.vocabulary,
  };
}

// ---------------------------------------------------------------- decode

/** Decode the source files of one dataset into the arrays the lab trains on. Pure apart from gunzip. */
export async function decodeKeras(
  id: KerasId,
  files: ReadonlyMap<string, Uint8Array>,
): Promise<KerasDecoded> {
  const spec = SPECS[id];
  for (const name of spec.files) {
    if (!files.has(name)) throw new Error(`${id}: missing ${name}`);
  }
  switch (id) {
    case "mnist":
      return decodeMnist(files.get("mnist.npz")!);
    case "fashion_mnist":
      return decodeFashion(files);
    case "imdb":
      return decodeImdb(
        files.get("imdb.npz")!,
        files.get("imdb_word_index.json")!,
      );
    case "reuters":
      return decodeReuters(
        files.get("reuters.npz")!,
        files.get("reuters_word_index.json")!,
      );
    case "california_housing":
      return decodeCalifornia(files.get("california_housing.npz")!);
  }
}

async function decodeMnist(npzBytes: Uint8Array): Promise<KerasDecoded> {
  const npz = await readNpz(npzBytes);
  const train = imageSplit(npz, "x_train", "y_train");
  const test = imageSplit(npz, "x_test", "y_test");
  return finish("mnist", train, test);
}

async function decodeFashion(
  files: ReadonlyMap<string, Uint8Array>,
): Promise<KerasDecoded> {
  const idx = async (name: string, magic: number): Promise<Uint8Array> => {
    const raw = await gunzip(files.get(name)!);
    const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    if (view.getUint32(0) !== magic)
      throw new Error(`${name}: unexpected IDX magic`);
    return raw;
  };
  const split = async (img: string, lab: string): Promise<LoadedSplit> => {
    const images = await idx(img, 2051);
    const labels = await idx(lab, 2049);
    const view = new DataView(
      images.buffer,
      images.byteOffset,
      images.byteLength,
    );
    const count = view.getUint32(4);
    if (view.getUint32(8) !== 28 || view.getUint32(12) !== 28)
      throw new Error(`${img}: expected 28x28 images`);
    if (images.byteLength !== 16 + count * IMAGE_PIXELS)
      throw new Error(`${img}: byte length does not match the header`);
    const labelView = new DataView(
      labels.buffer,
      labels.byteOffset,
      labels.byteLength,
    );
    if (labelView.getUint32(4) !== count || labels.byteLength !== 8 + count)
      throw new Error(`${lab}: count differs from ${img}`);
    return {
      count,
      x: images.slice(16),
      y: labels.slice(8),
    };
  };
  const train = await split(
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
  );
  const test = await split(
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
  );
  return finish("fashion_mnist", train, test);
}

async function decodeImdb(
  npzBytes: Uint8Array,
  indexBytes: Uint8Array,
): Promise<KerasDecoded> {
  const npz = await readNpz(npzBytes);
  const xTrain = objectArray(npz, "x_train");
  const yTrain = numbers(npz, "y_train");
  const xTest = objectArray(npz, "x_test");
  const yTest = numbers(npz, "y_test");
  const rng = new MT19937(SEED);
  const iTrain = shuffledIndices(rng, xTrain.length);
  const iTest = shuffledIndices(rng, xTest.length);
  const labelsTrain = iTrain.map((i) => yTrain[i]!);
  const labelsTest = iTest.map((i) => yTest[i]!);
  for (const label of [...labelsTrain, ...labelsTest]) {
    if (label !== 0 && label !== 1)
      throw new Error("imdb labels must be 0 or 1");
  }
  const train = textSplit(
    iTrain.map((i) => xTrain[i]!),
    labelsTrain,
  );
  const test = textSplit(
    iTest.map((i) => xTest[i]!),
    labelsTest,
  );
  const index = JSON.parse(new TextDecoder().decode(indexBytes)) as Record<
    string,
    number
  >;
  return finish("imdb", train, test, wordList(index));
}

async function decodeReuters(
  npzBytes: Uint8Array,
  indexBytes: Uint8Array,
): Promise<KerasDecoded> {
  const npz = await readNpz(npzBytes);
  const xs = objectArray(npz, "x");
  const ys = numbers(npz, "y");
  const order = shuffledIndices(new MT19937(SEED), xs.length);
  const seqs = order.map((i) => xs[i]!);
  const labels = order.map((i) => ys[i]!);
  const cut = Math.trunc(seqs.length * (1 - TEST_SPLIT));
  const train = textSplit(seqs.slice(0, cut), labels.slice(0, cut));
  const test = textSplit(seqs.slice(cut), labels.slice(cut));
  const index = JSON.parse(new TextDecoder().decode(indexBytes)) as Record<
    string,
    number
  >;
  return finish("reuters", train, test, wordList(index));
}

async function decodeCalifornia(npzBytes: Uint8Array): Promise<KerasDecoded> {
  const npz = await readNpz(npzBytes);
  const x = numbers(npz, "x");
  const y = numbers(npz, "y");
  const rows = y.length;
  if (x.length !== rows * CALIFORNIA_FEATURES)
    throw new Error("california_housing: x is not (rows, 8)");
  const order = shuffledIndices(new MT19937(SEED), rows);
  const cut = Math.trunc(rows * (1 - TEST_SPLIT));
  const trainRows = order.slice(0, cut);
  // numpy's mean and std along axis 0: sequential sums over the rows, then division by the count.
  const mean = new Float64Array(CALIFORNIA_FEATURES);
  for (const r of trainRows)
    for (let c = 0; c < CALIFORNIA_FEATURES; c++)
      mean[c] += x[r * CALIFORNIA_FEATURES + c]!;
  for (let c = 0; c < CALIFORNIA_FEATURES; c++) mean[c] = mean[c]! / cut;
  const sq = new Float64Array(CALIFORNIA_FEATURES);
  for (const r of trainRows)
    for (let c = 0; c < CALIFORNIA_FEATURES; c++) {
      const d = x[r * CALIFORNIA_FEATURES + c]! - mean[c]!;
      sq[c] += d * d;
    }
  const std = Array.from(sq, (s) => Math.sqrt(s / cut));
  const train = tabularSplit(order.slice(0, cut), x, y);
  const test = tabularSplit(order.slice(cut), x, y);
  const decoded = finish("california_housing", train, test);
  return {
    ...decoded,
    manifest: {
      ...decoded.manifest,
      normalization: { basis: "train" as const, mean: Array.from(mean), std },
    },
  };
}

function finish(
  id: KerasId,
  train: LoadedSplit,
  test: LoadedSplit,
  vocabulary?: string[],
): KerasDecoded {
  const spec = SPECS[id];
  if (train.count !== spec.counts.train || test.count !== spec.counts.test)
    throw new Error(
      `${id}: split counts ${train.count}/${test.count} differ from the spec`,
    );
  const xShape = (split: LoadedSplit): number[] => {
    if (spec.text) return [split.x.length];
    if (id === "california_housing") return [CALIFORNIA_FEATURES];
    return [28, 28, 1];
  };
  const splitRef = (split: LoadedSplit): SplitRef => ({
    count: split.count,
    x: fieldRef(split.x, xShape(split)),
    y: fieldRef(split.y, []),
    ...(split.offsets
      ? { offsets: fieldRef(split.offsets, [split.offsets.length]) }
      : {}),
  });
  const manifest: Manifest = {
    ...kerasCardManifest(id),
    splits: { train: splitRef(train), test: splitRef(test) },
  };
  return { train, test, manifest, vocabulary };
}

function fieldRef(array: Numeric, shape: number[]): FieldRef {
  return { dtype: dtypeOf(array), shape, parts: [] };
}

function dtypeOf(array: Numeric): FieldRef["dtype"] {
  if (array instanceof Uint8Array) return "uint8";
  if (array instanceof Uint16Array) return "uint16";
  if (array instanceof Uint32Array) return "uint32";
  return "float32";
}

function imageSplit(
  npz: Map<string, Npy>,
  xKey: string,
  yKey: string,
): LoadedSplit {
  const x = npzArray(npz, xKey);
  if (x.kind !== "numeric" || x.descr !== "|u1")
    throw new Error(`${xKey}: expected uint8 pixels`);
  const count = x.shape[0] ?? 0;
  if (x.bytes.byteLength !== count * IMAGE_PIXELS)
    throw new Error(`${xKey}: expected 28x28 images`);
  return { count, x: x.bytes.slice(), y: labelBytes(npz, yKey, count) };
}

function textSplit(
  seqs: readonly (readonly number[])[],
  labels: readonly number[],
): LoadedSplit {
  if (seqs.length !== labels.length)
    throw new Error("text split: sequences and labels differ");
  let total = 0;
  for (const seq of seqs) total += seq.length + 1;
  if (total > 0xffffffff)
    throw new Error("token stream too long for uint32 offsets");
  const tokens = new Uint16Array(total);
  const offsets = new Uint32Array(seqs.length + 1);
  let at = 0;
  for (let i = 0; i < seqs.length; i++) {
    offsets[i] = at;
    tokens[at++] = START_CHAR;
    for (const word of seqs[i]!) {
      const shifted = word + INDEX_FROM;
      tokens[at++] = shifted < VOCABULARY ? shifted : OOV_CHAR;
    }
  }
  offsets[seqs.length] = at;
  return {
    count: seqs.length,
    x: tokens,
    y: Uint8Array.from(labels, (label) => checkedByte(label, "label")),
    offsets,
  };
}

function tabularSplit(
  rows: readonly number[],
  x: Float64Array,
  y: Float64Array,
): LoadedSplit {
  const features = CALIFORNIA_FEATURES;
  const xs = new Float32Array(rows.length * features);
  const ys = new Float32Array(rows.length);
  rows.forEach((r, k) => {
    for (let c = 0; c < features; c++)
      xs[k * features + c] = Math.fround(x[r * features + c]!);
    ys[k] = Math.fround(y[r]!);
  });
  return { count: rows.length, x: xs, y: ys };
}

function wordList(index: Record<string, number>): string[] {
  const words = ["<pad>", "<start>", "<oov>"];
  while (words.length < VOCABULARY) words.push("");
  for (const [word, rank] of Object.entries(index)) {
    const token = rank + INDEX_FROM;
    if (token >= 3 && token < VOCABULARY) words[token] = word;
  }
  return words;
}

function checkedByte(value: number, what: string): number {
  if (!Number.isInteger(value) || value < 0 || value > 255)
    throw new Error(`${what} ${value} is outside uint8`);
  return value;
}

function labelBytes(
  npz: Map<string, Npy>,
  key: string,
  count: number,
): Uint8Array {
  const values = numbers(npz, key);
  if (values.length !== count)
    throw new Error(`${key}: count differs from the images`);
  return Uint8Array.from(values, (v) => checkedByte(v, key));
}

function npzArray(npz: Map<string, Npy>, key: string): Npy {
  const array = npz.get(key);
  if (!array) throw new Error(`npz has no ${key}`);
  return array;
}

function numbers(npz: Map<string, Npy>, key: string): Float64Array {
  return toFloat64(npzArray(npz, key));
}

function objectArray(npz: Map<string, Npy>, key: string): number[][] {
  const array = npzArray(npz, key);
  if (array.kind !== "object")
    throw new Error(`${key}: expected an object array`);
  for (const seq of array.items) {
    if (!Array.isArray(seq) || seq.some((v) => !Number.isInteger(v) || v < 0))
      throw new Error(`${key}: an item is not a list of token ids`);
  }
  return array.items as number[][];
}

// ---------------------------------------------------------------- gzip and zip

async function inflate(
  bytes: Uint8Array,
  format: "deflate-raw" | "gzip",
): Promise<Uint8Array> {
  const stream = new Blob([bytes as BlobPart])
    .stream()
    .pipeThrough(new DecompressionStream(format));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export function gunzip(bytes: Uint8Array): Promise<Uint8Array> {
  return inflate(bytes, "gzip");
}

/** Read an .npz (a zip of .npy members, stored or deflated) into name -> parsed array. */
export async function readNpz(bytes: Uint8Array): Promise<Map<string, Npy>> {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let eocd = -1;
  for (
    let i = bytes.byteLength - 22;
    i >= Math.max(0, bytes.byteLength - 22 - 65535);
    i--
  ) {
    if (view.getUint32(i, true) === 0x06054b50) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) throw new Error("npz: no end of central directory");
  const count = view.getUint16(eocd + 10, true);
  const cdOffset = view.getUint32(eocd + 16, true);
  if (count === 0xffff || cdOffset === 0xffffffff)
    throw new Error("npz: zip64 is not supported");
  const out = new Map<string, Npy>();
  const decoder = new TextDecoder();
  let p = cdOffset;
  for (let k = 0; k < count; k++) {
    if (view.getUint32(p, true) !== 0x02014b50)
      throw new Error("npz: bad central directory");
    const method = view.getUint16(p + 10, true);
    const compressed = view.getUint32(p + 20, true);
    const nameLength = view.getUint16(p + 28, true);
    const extraLength = view.getUint16(p + 30, true);
    const commentLength = view.getUint16(p + 32, true);
    const localOffset = view.getUint32(p + 42, true);
    const name = decoder.decode(bytes.subarray(p + 46, p + 46 + nameLength));
    p += 46 + nameLength + extraLength + commentLength;
    if (!name.endsWith(".npy")) continue;
    if (view.getUint32(localOffset, true) !== 0x04034b50)
      throw new Error(`npz: bad local header for ${name}`);
    const localName = view.getUint16(localOffset + 26, true);
    const localExtra = view.getUint16(localOffset + 28, true);
    const start = localOffset + 30 + localName + localExtra;
    const data = bytes.subarray(start, start + compressed);
    let member: Uint8Array;
    if (method === 0) member = data;
    else if (method === 8) member = await inflate(data, "deflate-raw");
    else throw new Error(`npz: ${name} uses zip method ${method}`);
    out.set(name.slice(0, -4), readNpy(member));
  }
  return out;
}

// ---------------------------------------------------------------- npy

export interface NpyNumeric {
  readonly kind: "numeric";
  readonly descr: string;
  readonly shape: number[];
  readonly bytes: Uint8Array;
}

export interface NpyObject {
  readonly kind: "object";
  readonly shape: number[];
  readonly items: unknown[];
}

export type Npy = NpyNumeric | NpyObject;

const ITEM_READERS: Record<
  string,
  { size: number; read: (view: DataView, at: number) => number }
> = {
  "|u1": { size: 1, read: (v, at) => v.getUint8(at) },
  "|i1": { size: 1, read: (v, at) => v.getInt8(at) },
  "<u2": { size: 2, read: (v, at) => v.getUint16(at, true) },
  "<i2": { size: 2, read: (v, at) => v.getInt16(at, true) },
  "<u4": { size: 4, read: (v, at) => v.getUint32(at, true) },
  "<i4": { size: 4, read: (v, at) => v.getInt32(at, true) },
  "<i8": { size: 8, read: (v, at) => Number(v.getBigInt64(at, true)) },
  "<f4": { size: 4, read: (v, at) => v.getFloat32(at, true) },
  "<f8": { size: 8, read: (v, at) => v.getFloat64(at, true) },
};

/** Parse one .npy file (format 1.0, 2.0 or 3.0). Object arrays are read through the pickle reader. */
export function readNpy(bytes: Uint8Array): Npy {
  const magic = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];
  if (bytes.byteLength < 10 || magic.some((b, i) => bytes[i] !== b))
    throw new Error("npy: bad magic");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const major = bytes[6];
  let headerStart: number;
  let headerLength: number;
  if (major === 1) {
    headerStart = 10;
    headerLength = view.getUint16(8, true);
  } else if (major === 2 || major === 3) {
    headerStart = 12;
    headerLength = view.getUint32(8, true);
  } else {
    throw new Error(`npy: unsupported version ${major}`);
  }
  const header = new TextDecoder(major === 3 ? "utf-8" : "latin1").decode(
    bytes.subarray(headerStart, headerStart + headerLength),
  );
  const descr = /'descr':\s*'([^']*)'/.exec(header)?.[1];
  const fortran = /'fortran_order':\s*(True|False)/.exec(header)?.[1];
  const shapeText = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1];
  if (descr === undefined || fortran === undefined || shapeText === undefined)
    throw new Error(`npy: cannot read the header ${header}`);
  if (fortran === "True")
    throw new Error("npy: Fortran order is not supported");
  const shape = shapeText
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map((s) => Number(s));
  const count = shape.reduce((a, b) => a * b, 1);
  const body = bytes.subarray(headerStart + headerLength);
  if (descr === "|O") {
    const items = unpickleObjectArray(body);
    if (items.length !== count)
      throw new Error("npy: object count differs from the shape");
    return { kind: "object", shape, items };
  }
  const reader = ITEM_READERS[descr];
  if (!reader) throw new Error(`npy: dtype ${descr} is not supported`);
  if (body.byteLength < count * reader.size)
    throw new Error("npy: data is shorter than the shape");
  return {
    kind: "numeric",
    descr,
    shape,
    bytes: body.slice(0, count * reader.size),
  };
}

/** Values of a numeric array as float64. Exact for every integer dtype used here. */
export function toFloat64(array: Npy): Float64Array {
  if (array.kind !== "numeric") throw new Error("expected a numeric array");
  const reader = ITEM_READERS[array.descr];
  if (!reader) throw new Error(`dtype ${array.descr} is not supported`);
  const view = new DataView(
    array.bytes.buffer,
    array.bytes.byteOffset,
    array.bytes.byteLength,
  );
  const count = array.bytes.byteLength / reader.size;
  const out = new Float64Array(count);
  for (let i = 0; i < count; i++) out[i] = reader.read(view, i * reader.size);
  return out;
}

// ---------------------------------------------------------------- pickle (numpy object arrays)

class GlobalRef {
  readonly name: string;
  constructor(name: string) {
    this.name = name;
  }
}

class ArrayPlaceholder {
  items: unknown[] | null = null;
}

class DtypePlaceholder {
  readonly descr: string;
  constructor(descr: string) {
    this.descr = descr;
  }
}

const ALLOWED_RECONSTRUCT = new Set([
  "numpy.core.multiarray._reconstruct",
  "numpy._core.multiarray._reconstruct",
]);

/**
 * The pickle opcodes that numpy writes for an object array of Python lists of ints (protocol 2 and
 * 3, as in imdb.npz and reuters.npz). Any other opcode, and any global other than numpy's
 * reconstruction, is refused, as Keras's restricted unpickler refuses it.
 */
export function unpickleObjectArray(data: Uint8Array): unknown[] {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const decoder = new TextDecoder();
  const stack: unknown[] = [];
  const marks: number[] = [];
  const memo = new Map<number, unknown>();
  let p = 0;
  const line = (): string => {
    const end = data.indexOf(0x0a, p);
    if (end < 0) throw new Error("pickle: unterminated line");
    const text = decoder.decode(data.subarray(p, end));
    p = end + 1;
    return text;
  };
  const top = (): unknown => {
    if (stack.length === 0) throw new Error("pickle: stack underflow");
    return stack[stack.length - 1];
  };
  const popMark = (): number => {
    const at = marks.pop();
    if (at === undefined) throw new Error("pickle: no mark");
    return at;
  };
  for (;;) {
    const op = String.fromCharCode(data[p++]!);
    switch (op) {
      case "\x80": // PROTO
        p += 1;
        break;
      case "\x95": // FRAME
        p += 8;
        break;
      case ".": // STOP
        if (stack.length !== 1)
          throw new Error("pickle: unexpected stack at STOP");
        return finishArray(stack[0]);
      case "c": {
        // GLOBAL module\nname\n
        const module = line();
        const name = line();
        stack.push(new GlobalRef(`${module}.${name}`));
        break;
      }
      case "q": // BINPUT
        memo.set(data[p++]!, top());
        break;
      case "r": // LONG_BINPUT
        memo.set(view.getUint32(p, true), top());
        p += 4;
        break;
      case "\x94": // MEMOIZE
        memo.set(memo.size, top());
        break;
      case "h": // BINGET
        stack.push(memoGet(memo, data[p++]!));
        break;
      case "j": // LONG_BINGET
        stack.push(memoGet(memo, view.getUint32(p, true)));
        p += 4;
        break;
      case "N": // NONE
        stack.push(null);
        break;
      case "\x88": // NEWTRUE
        stack.push(true);
        break;
      case "\x89": // NEWFALSE
        stack.push(false);
        break;
      case "K": // BININT1
        stack.push(data[p++]!);
        break;
      case "M": // BININT2
        stack.push(view.getUint16(p, true));
        p += 2;
        break;
      case "J": // BININT
        stack.push(view.getInt32(p, true));
        p += 4;
        break;
      case "U": {
        // SHORT_BINSTRING
        const n = data[p++]!;
        stack.push(decoder.decode(data.subarray(p, p + n)));
        p += n;
        break;
      }
      case "\x8c": {
        // SHORT_BINUNICODE
        const n = data[p++]!;
        stack.push(decoder.decode(data.subarray(p, p + n)));
        p += n;
        break;
      }
      case "]": // EMPTY_LIST
        stack.push([]);
        break;
      case ")": // EMPTY_TUPLE
        stack.push([]);
        break;
      case "(": // MARK
        marks.push(stack.length);
        break;
      case "t": {
        // TUPLE
        const at = popMark();
        stack.push(stack.splice(at));
        break;
      }
      case "\x85": // TUPLE1
        stack.push([stack.pop()]);
        break;
      case "\x86": {
        // TUPLE2
        const b = stack.pop();
        const a = stack.pop();
        stack.push([a, b]);
        break;
      }
      case "\x87": {
        // TUPLE3
        const c = stack.pop();
        const b = stack.pop();
        const a = stack.pop();
        stack.push([a, b, c]);
        break;
      }
      case "R": {
        // REDUCE
        const args = stack.pop();
        const callable = stack.pop();
        if (!(callable instanceof GlobalRef))
          throw new Error("pickle: REDUCE of a non-global");
        if (ALLOWED_RECONSTRUCT.has(callable.name)) {
          stack.push(new ArrayPlaceholder());
        } else if (callable.name === "numpy.dtype") {
          const descr = Array.isArray(args) ? args[0] : undefined;
          if (typeof descr !== "string")
            throw new Error("pickle: dtype without a name");
          stack.push(new DtypePlaceholder(descr));
        } else {
          throw new Error(`pickle: refusing ${callable.name}`);
        }
        break;
      }
      case "b": {
        // BUILD: the ndarray state (version, shape, dtype, fortran, objects)
        const state = stack.pop();
        const target = stack.pop();
        if (target instanceof DtypePlaceholder) {
          // numpy's dtype state (version, byteorder, fields, ..., itemsize, alignment, flags): not used.
          if (!Array.isArray(state))
            throw new Error("pickle: unexpected dtype state");
          stack.push(target);
          break;
        }
        if (!(target instanceof ArrayPlaceholder))
          throw new Error("pickle: BUILD of a non-array");
        if (!Array.isArray(state) || state.length !== 5)
          throw new Error("pickle: unexpected ndarray state");
        const dtype = state[2];
        if (
          !(dtype instanceof DtypePlaceholder) ||
          !dtype.descr.startsWith("O")
        )
          throw new Error("pickle: the array is not an object array");
        if (state[3] !== false)
          throw new Error("pickle: Fortran order is not supported");
        if (!Array.isArray(state[4])) throw new Error("pickle: no object list");
        target.items = state[4] as unknown[];
        stack.push(target);
        break;
      }
      case "a": {
        // APPEND
        const value = stack.pop();
        const list = top();
        if (!Array.isArray(list))
          throw new Error("pickle: APPEND to a non-list");
        list.push(value);
        break;
      }
      case "e": {
        // APPENDS
        const at = popMark();
        const list = stack[at - 1];
        if (!Array.isArray(list))
          throw new Error("pickle: APPENDS to a non-list");
        for (let i = at; i < stack.length; i++) list.push(stack[i]);
        stack.length = at;
        break;
      }
      default:
        throw new Error(
          `pickle: opcode ${JSON.stringify(op)} at ${p - 1} is not supported`,
        );
    }
  }
}

function memoGet(memo: Map<number, unknown>, index: number): unknown {
  if (!memo.has(index))
    throw new Error("pickle: BINGET of an unset memo entry");
  return memo.get(index);
}

function finishArray(value: unknown): unknown[] {
  if (!(value instanceof ArrayPlaceholder) || value.items === null)
    throw new Error("pickle: the top object is not an object array");
  return value.items;
}

// ---------------------------------------------------------------- numpy RandomState

/** MT19937 as numpy's legacy RandomState(seed) seeds and draws it (init_genrand, 32-bit draws). */
export class MT19937 {
  private readonly mt = new Uint32Array(624);
  private index = 624;

  constructor(seed: number) {
    this.mt[0] = seed >>> 0;
    for (let i = 1; i < 624; i++) {
      const prev = this.mt[i - 1]!;
      this.mt[i] = (Math.imul(1812433253, prev ^ (prev >>> 30)) + i) >>> 0;
    }
  }

  nextUint32(): number {
    if (this.index >= 624) this.twist();
    let y = this.mt[this.index++]!;
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  /** numpy's random_interval: a uniform integer in [0, max] by masked rejection. */
  randomInterval(max: number): number {
    if (max === 0) return 0;
    let mask = max;
    mask |= mask >>> 1;
    mask |= mask >>> 2;
    mask |= mask >>> 4;
    mask |= mask >>> 8;
    mask |= mask >>> 16;
    mask >>>= 0;
    let value: number;
    do {
      value = (this.nextUint32() & mask) >>> 0;
    } while (value > max);
    return value;
  }

  private twist(): void {
    const mt = this.mt;
    for (let i = 0; i < 624; i++) {
      const y = (mt[i]! & 0x80000000) | (mt[(i + 1) % 624]! & 0x7fffffff);
      let next = mt[(i + 397) % 624]! ^ (y >>> 1);
      if (y & 1) next ^= 0x9908b0df;
      mt[i] = next >>> 0;
    }
    this.index = 0;
  }
}

/** np.arange(n) shuffled in place by numpy's legacy RandomState.shuffle, from the same generator. */
export function shuffledIndices(rng: MT19937, length: number): number[] {
  const indices = Array.from({ length }, (_, i) => i);
  for (let i = length - 1; i >= 1; i--) {
    const j = rng.randomInterval(i);
    const swap = indices[i]!;
    indices[i] = indices[j]!;
    indices[j] = swap;
  }
  return indices;
}
