"use typeshade";

class Shape {
  outputLength: u32;
  axisSize: u32;
  inner: u32;
}

declare const shape: uniform<Shape>;

declare const input: storage<array<f32>>;

declare const output: storage<array<f32>, "read_write">;

@compute([64])
export function argMinKernel(@builtin("global_invocation_id") gid: vec3u) {
  if (gid.x >= shape.outputLength) {
    return;
  }
  let outerIndex: u32 = gid.x / shape.inner;
  let innerIndex: u32 = gid.x % shape.inner;
  let bestIndex: u32 = 0;

  let bestValue: f32 =
    input[outerIndex * shape.axisSize * shape.inner + innerIndex];
  for (let i: u32 = 1; i < shape.axisSize; i = i + 1) {
    let value: f32 =
      input[(outerIndex * shape.axisSize + i) * shape.inner + innerIndex];
    if (value < bestValue) {
      bestValue = value;
      bestIndex = i;
    }
  }
  output[gid.x] = f32(bestIndex);
}
