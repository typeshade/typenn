"use typeshade";

// Reductions over one axis. The input is read as [outer, axisSize, inner] and the output as
// [outer, inner]: one invocation for each output element, which loops over the axis. A reduction
// over every axis is one output element, and one invocation. The loop that runs over the axis is
// serial, so the order of a sum is the order of the axis, on every tier.

export function sumAxis(
  out: array<f32>,
  x: array<f32>,
  axisSize: u32,
  inner: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const base = (i / inner) * axisSize * inner + (i % inner);
    let total: f32 = 0.0;
    for (let k: u32 = 0; k < axisSize; k++) {
      total += x[base + k * inner];
    }
    out[i] = total;
  }
}

export function meanAxis(
  out: array<f32>,
  x: array<f32>,
  axisSize: u32,
  inner: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const base = (i / inner) * axisSize * inner + (i % inner);
    let total: f32 = 0.0;
    for (let k: u32 = 0; k < axisSize; k++) {
      total += x[base + k * inner];
    }
    out[i] = total / f32(axisSize);
  }
}

export function prodAxis(
  out: array<f32>,
  x: array<f32>,
  axisSize: u32,
  inner: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const base = (i / inner) * axisSize * inner + (i % inner);
    let total: f32 = 1.0;
    for (let k: u32 = 0; k < axisSize; k++) {
      total *= x[base + k * inner];
    }
    out[i] = total;
  }
}

export function maxAxis(
  out: array<f32>,
  x: array<f32>,
  axisSize: u32,
  inner: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const base = (i / inner) * axisSize * inner + (i % inner);
    let best: f32 = x[base];
    for (let k: u32 = 1; k < axisSize; k++) {
      best = max(best, x[base + k * inner]);
    }
    out[i] = best;
  }
}

export function minAxis(
  out: array<f32>,
  x: array<f32>,
  axisSize: u32,
  inner: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const base = (i / inner) * axisSize * inner + (i % inner);
    let best: f32 = x[base];
    for (let k: u32 = 1; k < axisSize; k++) {
      best = min(best, x[base + k * inner]);
    }
    out[i] = best;
  }
}

/** The index along the axis of the first greatest element, as a number. */
export function argMaxAxis(
  out: array<f32>,
  x: array<f32>,
  axisSize: u32,
  inner: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const base = (i / inner) * axisSize * inner + (i % inner);
    let bestValue: f32 = x[base];
    let bestIndex: u32 = 0;
    for (let k: u32 = 1; k < axisSize; k++) {
      const v = x[base + k * inner];
      if (v > bestValue) {
        bestValue = v;
        bestIndex = k;
      }
    }
    out[i] = f32(bestIndex);
  }
}

/** The index along the axis of the first least element, as a number. */
export function argMinAxis(
  out: array<f32>,
  x: array<f32>,
  axisSize: u32,
  inner: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const base = (i / inner) * axisSize * inner + (i % inner);
    let bestValue: f32 = x[base];
    let bestIndex: u32 = 0;
    for (let k: u32 = 1; k < axisSize; k++) {
      const v = x[base + k * inner];
      if (v < bestValue) {
        bestValue = v;
        bestIndex = k;
      }
    }
    out[i] = f32(bestIndex);
  }
}

/**
 * The gradient of a sum (`scale` 1) or a mean (`scale` 1 / axisSize) over an axis: each element
 * of the input receives the gradient of the output element it was reduced into. `out` has the
 * shape of the input, and `dy` the shape of the output.
 */
export function reduceGrad(
  out: array<f32>,
  dy: array<f32>,
  axisSize: u32,
  inner: u32,
  scale: f32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = dy[(i / (axisSize * inner)) * inner + (i % inner)] * scale;
  }
}
