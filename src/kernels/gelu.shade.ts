"use typeshade";

class Shape {
  length: u32;
}

declare const shape: uniform<Shape>;

declare const input: storage<array<f32>>;

declare const output: storage<array<f32>, "read_write">;

@compute([64])
export function geluKernel(@builtin("global_invocation_id") gid: vec3u) {
  if (gid.x >= shape.length) {
    return;
  }
  let value: f32 = input[gid.x];
  let scaled: f32 = 0.7978845608 * (value + 0.044715 * value * value * value);
  let tanhValue: f32;
  if (scaled > 10.0) {
    tanhValue = 1.0;
  } else if (scaled < -10.0) {
    tanhValue = -1.0;
  } else {
    tanhValue = 2.0 / (1.0 + exp(-2.0 * scaled)) - 1.0;
  }
  output[gid.x] = 0.5 * value * (1.0 + tanhValue);
}
