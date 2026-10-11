"use typeshade";

class Shape {
  length: u32;
}

declare const shape: uniform<Shape>;
declare const input: storage<array<f32>>;
declare const output: storage<array<f32>, "read_write">;

@compute([64])
export function acoshKernel(@builtin("global_invocation_id") gid: vec3u) {
  if (gid.x >= shape.length) {
    return;
  }
  output[gid.x] = acosh(input[gid.x]);
}
