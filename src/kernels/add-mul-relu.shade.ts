"use typeshade";

class Shape {
  length: u32;
}

declare const shape: uniform<Shape>;

declare const addA: storage<array<f32>>;

declare const addB: storage<array<f32>>;

declare const multiplier: storage<array<f32>>;

declare const output: storage<array<f32>, "read_write">;

@compute([64])
export function addMulReluKernel(@builtin("global_invocation_id") gid: vec3u) {
  if (gid.x >= shape.length) {
    return;
  }
  let product: f32 = (addA[gid.x] + addB[gid.x]) * multiplier[gid.x];
  output[gid.x] = max(product, 0.0);
}
