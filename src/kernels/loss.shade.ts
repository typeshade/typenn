"use typeshade";

// Softmax cross entropy between logits and labels, both [rows, columns]. A label row is a
// distribution (a one-hot row for a class), and the loss of a row is
// `sum(labels) * logsumexp(logits) - sum(labels * logits)`.

/** The loss of each row: `out` has one element for each row. */
export function softmaxCrossEntropy(
  out: array<f32>,
  logits: array<f32>,
  labels: array<f32>,
  columns: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const base = i * columns;
    let peak: f32 = logits[base];
    for (let c: u32 = 1; c < columns; c++) {
      peak = max(peak, logits[base + c]);
    }
    let expSum: f32 = 0.0;
    let labelLogit: f32 = 0.0;
    let labelSum: f32 = 0.0;
    for (let c: u32 = 0; c < columns; c++) {
      expSum += exp(logits[base + c] - peak);
      labelLogit += labels[base + c] * logits[base + c];
      labelSum += labels[base + c];
    }
    out[i] = labelSum * (peak + log(expSum)) - labelLogit;
  }
}

/** The gradient of the loss with respect to the logits; `dy` has the loss of each row. */
export function crossEntropyLogitsGrad(
  out: array<f32>,
  logits: array<f32>,
  labels: array<f32>,
  dy: array<f32>,
  columns: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const row = i / columns;
    const base = row * columns;
    let peak: f32 = logits[base];
    for (let c: u32 = 1; c < columns; c++) {
      peak = max(peak, logits[base + c]);
    }
    let expSum: f32 = 0.0;
    let labelSum: f32 = 0.0;
    for (let c: u32 = 0; c < columns; c++) {
      expSum += exp(logits[base + c] - peak);
      labelSum += labels[base + c];
    }
    const probability = exp(logits[i] - peak) / expSum;
    out[i] = (probability * labelSum - labels[i]) * dy[row];
  }
}

/** The gradient of the loss with respect to the labels. */
export function crossEntropyLabelsGrad(
  out: array<f32>,
  logits: array<f32>,
  dy: array<f32>,
  columns: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const row = i / columns;
    const base = row * columns;
    let peak: f32 = logits[base];
    for (let c: u32 = 1; c < columns; c++) {
      peak = max(peak, logits[base + c]);
    }
    let expSum: f32 = 0.0;
    for (let c: u32 = 0; c < columns; c++) {
      expSum += exp(logits[base + c] - peak);
    }
    out[i] = (peak + log(expSum) - logits[i]) * dy[row];
  }
}

/**
 * Softmax cross entropy with class ids: `classes[i]` is the class of row `i`, as an f32 value
 * below 2^24. Each row is computed as the one-hot kernel computes it, so the two give one value.
 */
export function sparseSoftmaxCrossEntropy(
  out: array<f32>,
  logits: array<f32>,
  classes: array<f32>,
  columns: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const base = i * columns;
    const label = u32(classes[i]);
    let peak: f32 = logits[base];
    for (let c: u32 = 1; c < columns; c++) {
      peak = max(peak, logits[base + c]);
    }
    let expSum: f32 = 0.0;
    for (let c: u32 = 0; c < columns; c++) {
      expSum += exp(logits[base + c] - peak);
    }
    out[i] = peak + log(expSum) - logits[base + label];
  }
}

/** The gradient of the class-id loss with respect to the logits; `dy` has the loss of each row. */
export function sparseCrossEntropyGrad(
  out: array<f32>,
  logits: array<f32>,
  classes: array<f32>,
  dy: array<f32>,
  columns: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const row = i / columns;
    const column = i % columns;
    const base = row * columns;
    const label = u32(classes[row]);
    let peak: f32 = logits[base];
    for (let c: u32 = 1; c < columns; c++) {
      peak = max(peak, logits[base + c]);
    }
    let expSum: f32 = 0.0;
    for (let c: u32 = 0; c < columns; c++) {
      expSum += exp(logits[base + c] - peak);
    }
    const probability = exp(logits[i] - peak) / expSum;
    const target = select(0.0, 1.0, column === label);
    out[i] = (probability - target) * dy[row];
  }
}
