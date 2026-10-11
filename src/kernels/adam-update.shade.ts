"use typeshade";

class Shape {
  count: u32;
  learningRate: f32;
  beta1: f32;
  beta2: f32;
  epsilon: f32;
  firstCorrection: f32;
  secondCorrection: f32;
}

declare const shape: uniform<Shape>;
declare const parameter: storage<array<f32>, "read_write">;
declare const gradient: storage<array<f32>>;
declare const firstMoment: storage<array<f32>, "read_write">;
declare const secondMoment: storage<array<f32>, "read_write">;

@compute([64])
export function adamUpdateKernel(@builtin("global_invocation_id") gid: vec3u) {
  if (gid.x >= shape.count) return;
  let g: f32 = gradient[gid.x];
  let first: f32 = shape.beta1 * firstMoment[gid.x] + (1.0 - shape.beta1) * g;
  let second: f32 =
    shape.beta2 * secondMoment[gid.x] + (1.0 - shape.beta2) * g * g;
  firstMoment[gid.x] = first;
  secondMoment[gid.x] = second;
  let correctedFirst: f32 = first / shape.firstCorrection;
  let correctedSecond: f32 = second / shape.secondCorrection;
  parameter[gid.x] -=
    (shape.learningRate * correctedFirst) /
    (sqrt(correctedSecond) + shape.epsilon);
}
