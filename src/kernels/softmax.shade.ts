"use typeshade";

// Softmax over one axis of an input read as [outer, axisSize, inner]. One invocation for each
// element of the output, so each one finds the greatest element and the sum of exponentials of its
// slice again. A slice has `axisSize` elements, and the work for a slice is `axisSize` times
// `axisSize`: the kernel is made for the short axes of a classifier.

export function softmax(
  out: array<f32>,
  x: array<f32>,
  axisSize: u32,
  inner: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const base = (i / (axisSize * inner)) * axisSize * inner + (i % inner);
    let peak: f32 = x[base];
    for (let k: u32 = 1; k < axisSize; k++) {
      peak = max(peak, x[base + k * inner]);
    }
    let total: f32 = 0.0;
    for (let k: u32 = 0; k < axisSize; k++) {
      total += exp(x[base + k * inner] - peak);
    }
    out[i] = exp(x[i] - peak) / total;
  }
}

export function logSoftmax(
  out: array<f32>,
  x: array<f32>,
  axisSize: u32,
  inner: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const base = (i / (axisSize * inner)) * axisSize * inner + (i % inner);
    let peak: f32 = x[base];
    for (let k: u32 = 1; k < axisSize; k++) {
      peak = max(peak, x[base + k * inner]);
    }
    let total: f32 = 0.0;
    for (let k: u32 = 0; k < axisSize; k++) {
      total += exp(x[base + k * inner] - peak);
    }
    out[i] = x[i] - peak - log(total);
  }
}

/** The gradient of `softmax` from `value`, its output, and `dy`, the gradient of that output. */
export function softmaxGrad(
  out: array<f32>,
  value: array<f32>,
  dy: array<f32>,
  axisSize: u32,
  inner: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const base = (i / (axisSize * inner)) * axisSize * inner + (i % inner);
    let reduction: f32 = 0.0;
    for (let k: u32 = 0; k < axisSize; k++) {
      reduction += dy[base + k * inner] * value[base + k * inner];
    }
    out[i] = value[i] * (dy[i] - reduction);
  }
}

/** The gradient of `logSoftmax` from `value`, its output, and `dy`. */
export function logSoftmaxGrad(
  out: array<f32>,
  value: array<f32>,
  dy: array<f32>,
  axisSize: u32,
  inner: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const base = (i / (axisSize * inner)) * axisSize * inner + (i % inner);
    let reduction: f32 = 0.0;
    for (let k: u32 = 0; k < axisSize; k++) {
      reduction += dy[base + k * inner];
    }
    out[i] = dy[i] - exp(value[i]) * reduction;
  }
}

/**
 * Per-slice statistics for the two-pass path, one invocation per entry of `statistics`.
 * For input viewed as [outer, axisSize, inner], `statistics` has 2 * outer * inner entries:
 * statistics[2 * slice] is the peak of the slice and statistics[2 * slice + 1] is the sum of
 * exp(x - peak) over the slice. The peak and the sum are kept apart so that a large peak does not
 * round away the low bits of the sum (the log of the sum is taken only in the apply kernels).
 * Each invocation reads its slice in O(axisSize); the sum entries read the peak again.
 */
export function softmaxStatistics(
  statistics: array<f32>,
  x: array<f32>,
  axisSize: u32,
  inner: u32,
): void {
  for (let i: u32 = 0; i < statistics.length; i++) {
    const slice = i / 2;
    const base = (slice / inner) * axisSize * inner + (slice % inner);
    let peak: f32 = x[base];
    for (let k: u32 = 1; k < axisSize; k++) {
      peak = max(peak, x[base + k * inner]);
    }
    let statistic: f32 = peak;
    if (i % 2 === 1) {
      let total: f32 = 0.0;
      for (let k: u32 = 0; k < axisSize; k++) {
        total += exp(x[base + k * inner] - peak);
      }
      statistic = total;
    }
    statistics[i] = statistic;
  }
}

/** Apply the cached peak and sum of each slice: exp(x - peak) / total. */
export function softmaxFromStatistics(
  out: array<f32>,
  x: array<f32>,
  statistics: array<f32>,
  axisSize: u32,
  inner: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const slice = (i / (axisSize * inner)) * inner + (i % inner);
    out[i] = exp(x[i] - statistics[2 * slice]) / statistics[2 * slice + 1];
  }
}

/** Apply the cached peak and sum of each slice: (x - peak) - log(total). */
export function logSoftmaxFromStatistics(
  out: array<f32>,
  x: array<f32>,
  statistics: array<f32>,
  axisSize: u32,
  inner: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const slice = (i / (axisSize * inner)) * inner + (i % inner);
    out[i] = x[i] - statistics[2 * slice] - log(statistics[2 * slice + 1]);
  }
}

/** Reduce dy * y of each slice once (softmax). One invocation per slice. */
export function softmaxGradReduction(
  out: array<f32>,
  value: array<f32>,
  dy: array<f32>,
  axisSize: u32,
  inner: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const base = (i / inner) * axisSize * inner + (i % inner);
    let total: f32 = 0.0;
    for (let k: u32 = 0; k < axisSize; k++) {
      total += dy[base + k * inner] * value[base + k * inner];
    }
    out[i] = total;
  }
}

/** Reduce dy of each slice once (log softmax). One invocation per slice. */
export function logSoftmaxGradReduction(
  out: array<f32>,
  dy: array<f32>,
  axisSize: u32,
  inner: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const base = (i / inner) * axisSize * inner + (i % inner);
    let total: f32 = 0.0;
    for (let k: u32 = 0; k < axisSize; k++) {
      total += dy[base + k * inner];
    }
    out[i] = total;
  }
}

/** Apply a cached slice reduction of dy * y to each element (softmax). */
export function softmaxGradFromReduction(
  out: array<f32>,
  value: array<f32>,
  dy: array<f32>,
  reduced: array<f32>,
  axisSize: u32,
  inner: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const slice = (i / (axisSize * inner)) * inner + (i % inner);
    out[i] = value[i] * (dy[i] - reduced[slice]);
  }
}

/** Apply a cached slice reduction of dy to each element (log softmax). */
export function logSoftmaxGradFromReduction(
  out: array<f32>,
  value: array<f32>,
  dy: array<f32>,
  reduced: array<f32>,
  axisSize: u32,
  inner: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const slice = (i / (axisSize * inner)) * inner + (i % inner);
    out[i] = dy[i] - exp(value[i]) * reduced[slice];
  }
}
