"use typeshade";

class Shape {
  length: u32;
  rank: u32;
  dimensions: array<u32, 8>;
  aStrides: array<u32, 8>;
  bStrides: array<u32, 8>;
}

declare const shape: uniform<Shape>;

declare const a: storage<array<f32>>;

declare const b: storage<array<f32>>;

declare const output: storage<array<f32>, "read_write">;

function operandIndex(flatIndex: u32, strides: array<u32, 8>): u32 {
  let remaining: u32 = flatIndex;
  let index: u32 = 0;
  for (let axis: u32 = 0; axis < shape.rank; axis = axis + 1) {
    let coordinate: u32 = remaining % shape.dimensions[axis];
    remaining /= shape.dimensions[axis];
    index += coordinate * strides[axis];
  }
  return index;
}

@compute([64])
export function maximumKernel(@builtin("global_invocation_id") gid: vec3u) {
  if (gid.x >= shape.length) {
    return;
  }
  output[gid.x] = max(
    a[operandIndex(gid.x, shape.aStrides)],
    b[operandIndex(gid.x, shape.bStrides)],
  );
}
