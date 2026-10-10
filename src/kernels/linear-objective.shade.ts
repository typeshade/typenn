"use typeshade";

class Shape {
  count: u32;
  outputFeatures: u32;
}

declare const shape: uniform<Shape>;

declare const labels: storage<array<u32>>;

declare const logits: storage<array<f32>, "read_write">;

declare const delta: storage<array<f32>, "read_write">;

declare const losses: storage<array<f32>, "read_write">;

@compute([64])
export function linearObjectiveKernel(@builtin("global_invocation_id") gid: vec3u) {
  let row: u32 = gid.x;
  if (row >= shape.count) {
    return;
  }
  let base: u32 = row * shape.outputFeatures;
  let peak: f32 = logits[base];
  for (let c: u32 = 1; c < shape.outputFeatures; c = c + 1) {
    peak = max(peak, logits[base + c]);
  }
  let total: f32 = 0.0;
  for (
    let reduceClass: u32 = 0;
    reduceClass < shape.outputFeatures;
    reduceClass = reduceClass + 1
  ) {
    total += exp(logits[base + reduceClass] - peak);
  }
  let label: u32 = labels[row];
  if (label >= shape.outputFeatures) {
    return;
  }
  losses[row] = peak - logits[base + label] + log(total);
  for (
    let outputClass: u32 = 0;
    outputClass < shape.outputFeatures;
    outputClass = outputClass + 1
  ) {
    let probability: f32 = exp(logits[base + outputClass] - peak) / total;
    delta[base + outputClass] =
      (probability - select(0.0, 1.0, outputClass == label)) / f32(shape.count);
  }
}
