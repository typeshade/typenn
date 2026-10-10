"use typeshade";

class Shape {
  count: u32;
  inputFeatures: u32;
  outputFeatures: u32;
}

declare const shape: uniform<Shape>;

declare const inputs: storage<array<f32>>;

declare const delta: storage<array<f32>, "read_write">;

declare const gradWeights: storage<array<f32>, "read_write">;

declare const gradBias: storage<array<f32>, "read_write">;

@compute([64])
export function linearBackwardKernel(@builtin("global_invocation_id") gid: vec3u) {
  let weightCount: u32 = shape.inputFeatures * shape.outputFeatures;
  let parameterCount: u32 = max(weightCount, shape.outputFeatures);
  if (gid.x >= parameterCount) {
    return;
  }
  if (gid.x < weightCount) {
    let inputIndex: u32 = gid.x / shape.outputFeatures;
    let outputIndex: u32 = gid.x % shape.outputFeatures;
    let sum: f32 = 0.0;
    for (let row: u32 = 0; row < shape.count; row = row + 1) {
      sum +=
        inputs[row * shape.inputFeatures + inputIndex] *
        delta[row * shape.outputFeatures + outputIndex];
    }
    gradWeights[gid.x] = sum;
  }
  if (gid.x < shape.outputFeatures) {
    let biasGradient: f32 = 0.0;
    for (let row: u32 = 0; row < shape.count; row = row + 1) {
      biasGradient += delta[row * shape.outputFeatures + gid.x];
    }
    gradBias[gid.x] = biasGradient;
  }
}
