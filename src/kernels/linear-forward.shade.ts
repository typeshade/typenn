"use typeshade";

class Shape {
  count: u32;
  inputFeatures: u32;
  outputFeatures: u32;
}

declare const shape: uniform<Shape>;

declare const inputs: storage<array<f32>>;

declare const weights: storage<array<f32>, "read_write">;

declare const bias: storage<array<f32>, "read_write">;

declare const logits: storage<array<f32>, "read_write">;

@compute([64])
export function linearForwardKernel(@builtin("global_invocation_id") gid: vec3u) {
  let length: u32 = shape.count * shape.outputFeatures;
  if (gid.x >= length) {
    return;
  }
  let row: u32 = gid.x / shape.outputFeatures;
  let outputIndex: u32 = gid.x % shape.outputFeatures;
  let value: f32 = bias[outputIndex];
  for (let k: u32 = 0; k < shape.inputFeatures; k = k + 1) {
    value +=
      inputs[row * shape.inputFeatures + k] *
      weights[k * shape.outputFeatures + outputIndex];
  }
  logits[gid.x] = value;
}
