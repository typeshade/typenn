import type { SourceArray } from "./device-dataset.js";

/**
 * The host reference of each batch kind. A device batch equals these values bit for bit for the
 * same indices: the device runs the same formulas. They also fill a batch when a dataset falls back
 * to host batching (`residency`).
 */

/** The normalised input of the samples at `indices`: `(x - mean) / std` for each feature. */
export function hostFeatures(
  x: SourceArray,
  indices: ArrayLike<number>,
  features: number,
  mean: ArrayLike<number>,
  std: ArrayLike<number>,
): Float32Array {
  const out = new Float32Array(indices.length * features);
  // The device computes in f32: round mean and std, and the difference, as f32 does. A double
  // sum or quotient of two f32 values rounds to the f32 result of the operation.
  const center = Float32Array.from(mean);
  const spread = Float32Array.from(std);
  for (let k = 0; k < indices.length; k++) {
    const src = indices[k]! * features;
    for (let f = 0; f < features; f++)
      out[k * features + f] = Math.fround(
        Math.fround(x[src + f]! - center[f]!) / spread[f]!,
      );
  }
  return out;
}

/** The multi-hot bag of words of the samples at `indices` over the first `vocab` token ids. */
export function hostBagOfWords(
  x: SourceArray,
  offsets: Uint32Array,
  indices: ArrayLike<number>,
  vocab: number,
): Float32Array {
  const out = new Float32Array(indices.length * vocab);
  for (let k = 0; k < indices.length; k++) {
    const i = indices[k]!;
    for (let t = offsets[i]!; t < offsets[i + 1]!; t++) {
      const token = x[t]!;
      if (token < vocab) out[k * vocab + token] = 1;
    }
  }
  return out;
}

/** The label values (class ids, or normalised targets) of the samples at `indices`, as f32. */
export function hostLabels(
  y: SourceArray,
  indices: ArrayLike<number>,
  target: { readonly mean: number; readonly std: number } | undefined,
): Float32Array {
  const out = new Float32Array(indices.length);
  for (let k = 0; k < indices.length; k++) {
    const value = y[indices[k]!]!;
    out[k] =
      target === undefined
        ? value
        : Math.fround(
            Math.fround(value - Math.fround(target.mean)) /
              Math.fround(target.std || 1),
          );
  }
  return out;
}
