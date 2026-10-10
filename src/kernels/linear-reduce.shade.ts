"use typeshade";

class Shape {
  count: u32;
  outputFeatures: u32;
}

declare const shape: uniform<Shape>;

declare const labels: storage<array<u32>>;

declare const logits: storage<array<f32>, "read_write">;

declare const losses: storage<array<f32>, "read_write">;

declare const stats: storage<array<f32>, "read_write">;

@compute([1])
export function linearReduceKernel() {
  let totalLoss: f32 = 0.0;
  let correct: f32 = 0.0;
  for (let row: u32 = 0; row < shape.count; row = row + 1) {
    totalLoss += losses[row];
    let best: u32 = 0;
    for (let c: u32 = 1; c < shape.outputFeatures; c = c + 1) {
      if (
        logits[row * shape.outputFeatures + c] >
        logits[row * shape.outputFeatures + best]
      ) {
        best = c;
      }
    }
    if (best == labels[row]) {
      correct += 1.0;
    }
  }
  stats[0] = totalLoss / f32(shape.count);
  stats[1] = correct;
}
