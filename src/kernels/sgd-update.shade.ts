"use typeshade";

class Shape {
  count: u32;
  learningRate: f32;
}

declare const shape: uniform<Shape>;
declare const parameter: storage<array<f32>, "read_write">;
declare const gradient: storage<array<f32>>;

@compute([64])
export function sgdUpdateKernel(@builtin("global_invocation_id") gid: vec3u) {
  if (gid.x >= shape.count) return;
  parameter[gid.x] -= shape.learningRate * gradient[gid.x];
}
