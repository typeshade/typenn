"use typeshade";

class Shape {
  length: u32;
}

declare const shape: uniform<Shape>;

declare const input: storage<array<f32>>;

declare const output: storage<array<f32>, "read_write">;

@compute([64])
export function log1pKernel(@builtin("global_invocation_id") gid: vec3u) {
  if (gid.x >= shape.length) {
    return;
  }
  let value: f32 = input[gid.x];
  if (value > -0.01 && value < 0.01) {
    let square: f32 = value * value;
    output[gid.x] =
      value -
      square * (0.5 - value * (0.3333333333 - value * (0.25 - value * 0.2)));
  } else {
    output[gid.x] = log(1.0 + value);
  }
}
