"use typeshade";

/**
 * `out = a b` for rank-2 matrices: [rows, inner] times [inner, cols]. One invocation for each
 * element of the output, which sums over `inner` in order. `transposeA` says that `a` is stored
 * as [inner, rows], and `transposeB` that `b` is stored as [cols, inner]. Each is 0 or 1.
 */
export function matmul(
  out: array<f32>,
  a: array<f32>,
  b: array<f32>,
  rows: u32,
  inner: u32,
  cols: u32,
  transposeA: u32,
  transposeB: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const row = i / cols;
    const col = i % cols;
    let total: f32 = 0.0;
    for (let k: u32 = 0; k < inner; k++) {
      const ai = select(k * rows + row, row * inner + k, transposeA === 0);
      const bi = select(col * inner + k, k * cols + col, transposeB === 0);
      total += a[ai] * b[bi];
    }
    out[i] = total;
  }
}
