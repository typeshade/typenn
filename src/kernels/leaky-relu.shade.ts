"use typeshade";

class Shape {
  length: u32;
  alpha: f32;
}

declare const shape: uniform<Shape>;

declare const input: storage<array<f32>>;

declare const output: storage<array<f32>, "read_write">;

@compute([64])
export function leakyReluKernel(@builtin("global_invocation_id") gid: vec3u) {
  if (gid.x >= shape.length) {
    return;
  }
  let value: f32 = input[gid.x];
  output[gid.x] = select(shape.alpha * value, value, value >= 0.0);
}
