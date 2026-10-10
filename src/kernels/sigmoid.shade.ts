"use typeshade";

class Shape {
  length: u32;
}

declare const shape: uniform<Shape>;

declare const input: storage<array<f32>>;

declare const output: storage<array<f32>, "read_write">;

@compute([64])
export function sigmoidKernel(@builtin("global_invocation_id") gid: vec3u) {
  if (gid.x >= shape.length) {
    return;
  }
  let value: f32 = input[gid.x];
  if (value >= 0.0) {
    output[gid.x] = 1.0 / (1.0 + exp(-value));
  } else {
    let exponential: f32 = exp(value);
    output[gid.x] = exponential / (1.0 + exponential);
  }
}
