"use typeshade";

class Shape {
  inputFeatures: u32;
  outputFeatures: u32;
  learningRate: f32;
}

declare const shape: uniform<Shape>;

declare const weights: storage<array<f32>, "read_write">;

declare const bias: storage<array<f32>, "read_write">;

declare const gradWeights: storage<array<f32>, "read_write">;

declare const gradBias: storage<array<f32>, "read_write">;

@compute([64])
export function linearUpdateKernel(@builtin("global_invocation_id") gid: vec3u) {
  let weightCount: u32 = shape.inputFeatures * shape.outputFeatures;
  let parameterCount: u32 = max(weightCount, shape.outputFeatures);
  if (gid.x >= parameterCount) {
    return;
  }
  if (gid.x < weightCount) {
    weights[gid.x] -= shape.learningRate * gradWeights[gid.x];
  }
  if (gid.x < shape.outputFeatures) {
    bias[gid.x] -= shape.learningRate * gradBias[gid.x];
  }
}
