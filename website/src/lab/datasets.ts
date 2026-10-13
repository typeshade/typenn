// Browser loader for the dataset mirror that scripts/fetch_datasets.py writes to /data/.
// The pure helpers at the bottom take typed arrays and have no DOM dependency, so they run under
// node in scripts/test-datasets.mjs.

export type Task =
  "image-classification" | "text-classification" | "regression";

export interface DatasetInfo {
  id: string;
  title: string;
  task: Task;
  totalBytes: number;
  manifest: string;
}

export interface PartRef {
  file: string;
  bytes: number;
  sha256: string;
}

export interface FieldRef {
  dtype: "uint8" | "uint16" | "uint32" | "float32";
  shape: number[];
  parts: PartRef[];
}

export interface SplitRef {
  count: number;
  x: FieldRef;
  y: FieldRef;
  offsets?: FieldRef;
}

export interface Manifest {
  id: string;
  title: string;
  task: Task;
  license: string;
  citation: string;
  classes: string[] | null;
  features?: string[];
  vocabulary?: number;
  normalization?: { basis: "train"; mean: number[]; std: number[] };
  splits: { train: SplitRef; test: SplitRef };
}

export type Numeric = Uint8Array | Uint16Array | Uint32Array | Float32Array;

export interface LoadedSplit {
  count: number;
  x: Numeric;
  y: Numeric;
  offsets?: Uint32Array;
}

export interface LoadedDataset {
  info: DatasetInfo;
  manifest: Manifest;
  train: LoadedSplit;
  test: LoadedSplit;
}

export interface LoadOptions {
  onProgress?: (loadedBytes: number, totalBytes: number) => void;
  signal?: AbortSignal;
}

const CACHE_NAME = "typenn-datasets-v1";
const CONCURRENCY = 4;

export async function listDatasets(
  signal?: AbortSignal,
): Promise<DatasetInfo[]> {
  const response = await fetch("/data/index.json", { signal });
  if (!response.ok) throw new Error(`index.json: HTTP ${response.status}`);
  return (await response.json()) as DatasetInfo[];
}

export async function loadDataset(
  id: string,
  options: LoadOptions = {},
): Promise<LoadedDataset> {
  const { onProgress, signal } = options;
  const datasets = await listDatasets(signal);
  const info = datasets.find((d) => d.id === id);
  if (!info) throw new Error(`Unknown dataset: ${id}`);
  const base = `/data/${id}/`;
  const manifest = (await fetchJson(
    base + "manifest.json",
    signal,
  )) as Manifest;

  const parts: { ref: PartRef; path: string }[] = [];
  for (const split of ["train", "test"] as const) {
    const s = manifest.splits[split];
    for (const field of [s.x, s.y, s.offsets]) {
      if (!field) continue;
      for (const ref of field.parts) parts.push({ ref, path: base + ref.file });
    }
  }
  const totalBytes = parts.reduce((sum, p) => sum + p.ref.bytes, 0);
  let loadedBytes = 0;
  onProgress?.(0, totalBytes);

  const buffers = await runPool(parts, CONCURRENCY, async ({ ref, path }) => {
    const buffer = await fetchVerified(path, ref, signal);
    loadedBytes += buffer.byteLength;
    onProgress?.(loadedBytes, totalBytes);
    return buffer;
  });

  // runPool keeps the order of `parts`: per split, x parts, then y, then offsets.
  const cursor = { next: 0 };
  const readSplit = (name: "train" | "test"): LoadedSplit => {
    const s = manifest.splits[name];
    const x = typed(s.x.dtype, takeParts(s.x, buffers, cursor));
    const y = typed(s.y.dtype, takeParts(s.y, buffers, cursor));
    const offsets = s.offsets
      ? (typed("uint32", takeParts(s.offsets, buffers, cursor)) as Uint32Array)
      : undefined;
    return { count: s.count, x, y, offsets };
  };
  const train = readSplit("train");
  const test = readSplit("test");
  return { info, manifest, train, test };
}

// ---------------------------------------------------------------- batch helpers

/** Image batch scaled to [0, 1]. `out` holds indices.length * (product of the image shape) floats. */
export function imageBatch(
  split: LoadedSplit,
  indices: ArrayLike<number>,
  out: Float32Array,
): Float32Array {
  const size = split.x.length / split.count;
  requireLength(out, indices.length * size, "imageBatch out");
  for (let k = 0; k < indices.length; k++) {
    const src = indices[k] * size;
    const dst = k * size;
    for (let j = 0; j < size; j++) out[dst + j] = split.x[src + j] / 255;
  }
  return out;
}

/** Binary multi-hot bag of words over the first `vocab` token ids. Ids >= vocab are ignored. */
export function bagOfWordsBatch(
  split: LoadedSplit,
  indices: ArrayLike<number>,
  vocab: number,
  out: Float32Array,
): Float32Array {
  if (!split.offsets)
    throw new Error("bagOfWordsBatch needs a text split (offsets)");
  requireLength(out, indices.length * vocab, "bagOfWordsBatch out");
  out.fill(0, 0, indices.length * vocab);
  const offsets = split.offsets;
  for (let k = 0; k < indices.length; k++) {
    const row = k * vocab;
    const i = indices[k];
    for (let t = offsets[i]; t < offsets[i + 1]; t++) {
      const token = split.x[t];
      if (token < vocab) out[row + token] = 1;
    }
  }
  return out;
}

/** Tabular batch normalised with the train-split mean and std. A zero std is treated as 1. */
export function tabularBatch(
  split: LoadedSplit,
  indices: ArrayLike<number>,
  mean: ArrayLike<number>,
  std: ArrayLike<number>,
  out: Float32Array,
): Float32Array {
  const features = mean.length;
  requireLength(out, indices.length * features, "tabularBatch out");
  for (let k = 0; k < indices.length; k++) {
    const src = indices[k] * features;
    const dst = k * features;
    for (let f = 0; f < features; f++) {
      const scale = std[f] === 0 ? 1 : std[f];
      out[dst + f] = (split.x[src + f] - mean[f]) / scale;
    }
  }
  return out;
}

/** Labels for the indices, as a new array of the split's label type. */
export function labels(
  split: LoadedSplit,
  indices: ArrayLike<number>,
): Uint8Array | Float32Array {
  const isFloat = split.y instanceof Float32Array;
  const out = isFloat
    ? new Float32Array(indices.length)
    : new Uint8Array(indices.length);
  for (let k = 0; k < indices.length; k++) out[k] = split.y[indices[k]];
  return out;
}

/** Concatenate buffers in order. Used for the parts of one field. */
export function concatBytes(parts: ArrayBuffer[]): ArrayBuffer {
  const total = parts.reduce((sum, p) => sum + p.byteLength, 0);
  const out = new Uint8Array(total);
  let at = 0;
  for (const part of parts) {
    out.set(new Uint8Array(part), at);
    at += part.byteLength;
  }
  return out.buffer;
}

/** Read little-endian bytes as the typed array of the given dtype. */
export function typed(dtype: FieldRef["dtype"], bytes: ArrayBuffer): Numeric {
  switch (dtype) {
    case "uint8":
      return new Uint8Array(bytes);
    case "uint16":
      return readLittleEndian(
        bytes,
        2,
        (view, o) => view.getUint16(o, true),
        Uint16Array,
      );
    case "uint32":
      return readLittleEndian(
        bytes,
        4,
        (view, o) => view.getUint32(o, true),
        Uint32Array,
      );
    case "float32":
      return readLittleEndian(
        bytes,
        4,
        (view, o) => view.getFloat32(o, true),
        Float32Array,
      );
  }
}

/** Verify a part's length and SHA-256 against the manifest. Throws on mismatch. */
export async function verifyPart(
  buffer: ArrayBuffer,
  ref: PartRef,
): Promise<void> {
  if (buffer.byteLength !== ref.bytes) {
    throw new Error(
      `${ref.file}: expected ${ref.bytes} bytes, got ${buffer.byteLength}`,
    );
  }
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  const hex = Array.from(new Uint8Array(digest), (b) =>
    b.toString(16).padStart(2, "0"),
  ).join("");
  if (hex !== ref.sha256) throw new Error(`${ref.file}: SHA-256 mismatch`);
}

// ---------------------------------------------------------------- internals

function requireLength(out: Float32Array, length: number, name: string): void {
  if (out.length < length)
    throw new Error(`${name} needs ${length} floats, has ${out.length}`);
}

const LITTLE_ENDIAN_HOST = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

function readLittleEndian<T extends Numeric>(
  bytes: ArrayBuffer,
  width: number,
  read: (view: DataView, offset: number) => number,
  Ctor: new (source: number | ArrayBuffer) => T,
): T {
  if (bytes.byteLength % width !== 0)
    throw new Error("byte length is not a multiple of the element size");
  if (LITTLE_ENDIAN_HOST) return new Ctor(bytes);
  const count = bytes.byteLength / width;
  const out = new Ctor(count);
  const view = new DataView(bytes);
  for (let i = 0; i < count; i++)
    (out as unknown as Record<number, number>)[i] = read(view, i * width);
  return out;
}

function takeParts(
  field: FieldRef,
  buffers: ArrayBuffer[],
  cursor: { next: number },
): ArrayBuffer {
  const pieces = buffers.slice(cursor.next, cursor.next + field.parts.length);
  cursor.next += field.parts.length;
  return concatBytes(pieces);
}

async function fetchJson(url: string, signal?: AbortSignal): Promise<unknown> {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}

/** Fetch one part through the Cache API when it is available, and verify it either way. */
async function fetchVerified(
  url: string,
  ref: PartRef,
  signal?: AbortSignal,
): Promise<ArrayBuffer> {
  const cache = await openCache();
  const cached = cache
    ? await cache.match(url).catch(() => undefined)
    : undefined;
  if (cached) {
    try {
      const buffer = await cached.arrayBuffer();
      await verifyPart(buffer, ref);
      return buffer;
    } catch {
      await cache?.delete(url).catch(() => undefined);
    }
  }
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const buffer = await response.arrayBuffer();
  await verifyPart(buffer, ref);
  if (cache) {
    cache
      .put(
        url,
        new Response(buffer.slice(0), {
          headers: { "Content-Type": "application/octet-stream" },
        }),
      )
      .catch(() => undefined);
  }
  return buffer;
}

async function openCache(): Promise<Cache | undefined> {
  try {
    if (typeof caches === "undefined") return undefined;
    return await caches.open(CACHE_NAME);
  } catch {
    return undefined;
  }
}

async function runPool<T, R>(
  items: T[],
  limit: number,
  task: (item: T) => Promise<R>,
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let cursor = 0;
  const workers = Array.from(
    { length: Math.min(limit, items.length) },
    async () => {
      while (cursor < items.length) {
        const index = cursor++;
        results[index] = await task(items[index]);
      }
    },
  );
  await Promise.all(workers);
  return results;
}
