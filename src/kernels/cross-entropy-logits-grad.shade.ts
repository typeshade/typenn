"use typeshade";

class CrossEntropyShape {
  columns: u32;
  length: u32;
}

declare const crossEntropyShape: uniform<CrossEntropyShape>;

declare const logits: storage<array<f32>>;

declare const labels: storage<array<f32>>;

declare const upstream: storage<array<f32>>;

declare const gradient: storage<array<f32>, "read_write">;

@compute([64])
export function crossEntropyLogitsGradKernel(@builtin("global_invocation_id") gid: vec3u) {
  if (gid.x >= crossEntropyShape.length) {
    return;
  }
  let row: u32 = gid.x / crossEntropyShape.columns;
  let base: u32 = row * crossEntropyShape.columns;
  let peak: f32 = logits[base];
  for (let i: u32 = 1; i < crossEntropyShape.columns; i = i + 1) {
    peak = max(peak, logits[base + i]);
  }
  let expSum: f32 = 0.0;
  let labelSum: f32 = 0.0;
  for (
    let column: u32 = 0;
    column < crossEntropyShape.columns;
    column = column + 1
  ) {
    expSum += exp(logits[base + column] - peak);
    labelSum += labels[base + column];
  }
  let probability: f32 = exp(logits[gid.x] - peak) / expSum;
  gradient[gid.x] = (probability * labelSum - labels[gid.x]) * upstream[row];
}
