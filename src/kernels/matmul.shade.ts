"use typeshade";

class Shape {
  rows: u32;
  inner: u32;
  columns: u32;
  transposeA: u32;
  transposeB: u32;
}

declare const shape: uniform<Shape>;

declare const a: storage<array<f32>>;

declare const b: storage<array<f32>>;

declare const output: storage<array<f32>, "read_write">;

@compute([64])
export function matmulKernel(@builtin("global_invocation_id") gid: vec3u) {
  let outputLength: u32 = shape.rows * shape.columns;
  if (gid.x >= outputLength) {
    return;
  }
  let row: u32 = gid.x / shape.columns;
  let column: u32 = gid.x % shape.columns;
  let sum: f32 = 0.0;
  for (let k: u32 = 0; k < shape.inner; k = k + 1) {
    let aIndex: u32 = select(
      k * shape.rows + row,
      row * shape.inner + k,
      shape.transposeA == 0,
    );
    let bIndex: u32 = select(
      column * shape.inner + k,
      k * shape.columns + column,
      shape.transposeB == 0,
    );
    sum += a[aIndex] * b[bIndex];
  }
  output[gid.x] = sum;
}
