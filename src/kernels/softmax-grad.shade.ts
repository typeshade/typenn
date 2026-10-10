"use typeshade";

class Shape {
  outer: u32;
  axisSize: u32;
  inner: u32;
  logMode: u32;
}

declare const shape: uniform<Shape>;

declare const value: storage<array<f32>>;

declare const dy: storage<array<f32>>;

declare const output: storage<array<f32>, "read_write">;

@compute([64])
export function softmaxGradKernel(@builtin("global_invocation_id") gid: vec3u) {
  let slices: u32 = shape.outer * shape.inner;
  if (gid.x >= slices) {
    return;
  }
  let outerIndex: u32 = gid.x / shape.inner;
  let innerIndex: u32 = gid.x % shape.inner;
  let base: u32 = outerIndex * shape.axisSize * shape.inner + innerIndex;
  let reduction: f32 = 0.0;

  for (let i: u32 = 0; i < shape.axisSize; i = i + 1) {
    let index: u32 = base + i * shape.inner;
    if (shape.logMode == 0) {
      reduction += dy[index] * value[index];
    } else {
      reduction += dy[index];
    }
  }
  for (
    let outputAxis: u32 = 0;
    outputAxis < shape.axisSize;
    outputAxis = outputAxis + 1
  ) {
    let outputIndex: u32 = base + outputAxis * shape.inner;
    if (shape.logMode == 0) {
      output[outputIndex] = value[outputIndex] * (dy[outputIndex] - reduction);
    } else {
      output[outputIndex] =
        dy[outputIndex] - exp(value[outputIndex]) * reduction;
    }
  }
}
