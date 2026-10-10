"use typeshade";

class Shape {
  outer: u32;
  axisSize: u32;
  inner: u32;
}

declare const shape: uniform<Shape>;

declare const input: storage<array<f32>>;

declare const output: storage<array<f32>, "read_write">;

@compute([64])
export function logSoftmaxKernel(@builtin("global_invocation_id") gid: vec3u) {
  let slice: u32 = gid.x;
  let sliceCount: u32 = shape.outer * shape.inner;
  if (slice >= sliceCount) {
    return;
  }
  let outerIndex: u32 = slice / shape.inner;
  let innerIndex: u32 = slice % shape.inner;
  let base: u32 = outerIndex * shape.axisSize * shape.inner + innerIndex;
  let peak: f32 = input[base];
  for (let index: u32 = 1; index < shape.axisSize; index = index + 1) {
    peak = max(peak, input[base + index * shape.inner]);
  }
  let total: f32 = 0.0;
  for (
    let reduceIndex: u32 = 0;
    reduceIndex < shape.axisSize;
    reduceIndex = reduceIndex + 1
  ) {
    total += exp(input[base + reduceIndex * shape.inner] - peak);
  }
  let logNormalizer: f32 = peak + log(total);
  for (
    let outputIndex: u32 = 0;
    outputIndex < shape.axisSize;
    outputIndex = outputIndex + 1
  ) {
    let offset: u32 = base + outputIndex * shape.inner;
    output[offset] = input[offset] - logNormalizer;
  }
}
