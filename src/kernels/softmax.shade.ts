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
    out[i] = x[i] - (peak + log(total));
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
