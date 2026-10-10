"use typeshade";

class Shape {
  rows: u32;
  columns: u32;
}

declare const shape: uniform<Shape>;

declare const logits: storage<array<f32>>;

declare const labels: storage<array<f32>>;

declare const output: storage<array<f32>, "read_write">;

@compute([64])
export function softmaxCrossEntropyWithLogitsKernel(@builtin("global_invocation_id") gid: vec3u) {
  let row: u32 = gid.x;
  if (row >= shape.rows) {
    return;
  }
  let base: u32 = row * shape.columns;
  let peak: f32 = logits[base];
  for (let column: u32 = 1; column < shape.columns; column = column + 1) {
    peak = max(peak, logits[base + column]);
  }
  let expSum: f32 = 0.0;
  let labelLogit: f32 = 0.0;
  let labelSum: f32 = 0.0;
  for (let column: u32 = 0; column < shape.columns; column = column + 1) {
    expSum += exp(logits[base + column] - peak);
    let label: f32 = labels[base + column];
    labelLogit += label * logits[base + column];
    labelSum += label;
  }
  output[row] = labelSum * (peak + log(expSum)) - labelLogit;
}
