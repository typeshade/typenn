"use typeshade";

class ReduceShape {
  axisSize: u32;
  inner: u32;
  inputLength: u32;
  meanScale: f32;
}

declare const reduceShape: uniform<ReduceShape>;

declare const reduceDy: storage<array<f32>>;

declare const reduceOutput: storage<array<f32>, "read_write">;

@compute([64])
export function reduceGradKernel(@builtin("global_invocation_id") gid: vec3u) {
  if (gid.x >= reduceShape.inputLength) {
    return;
  }
  let inputIndex: u32 = gid.x;
  let outputIndex: u32 =
    (inputIndex / (reduceShape.axisSize * reduceShape.inner)) *
      reduceShape.inner +
    (inputIndex % reduceShape.inner);
  reduceOutput[inputIndex] = reduceDy[outputIndex] * reduceShape.meanScale;
}
