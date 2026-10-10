"use typeshade";

class Shape {
  height: u32;
  width: u32;
  channels: u32;
  filterHeight: u32;
  filterWidth: u32;
  outputHeight: u32;
  outputWidth: u32;
  strideY: u32;
  strideX: u32;
  padTop: u32;
  padLeft: u32;
  outputLength: u32;
}

declare const shape: uniform<Shape>;

declare const input: storage<array<f32>>;

declare const output: storage<array<f32>, "read_write">;

@compute([64])
export function maxPoolKernel(@builtin("global_invocation_id") gid: vec3u) {
  if (gid.x >= shape.outputLength) {
    return;
  }
  let channel: u32 = gid.x % shape.channels;
  let outputPixel: u32 = gid.x / shape.channels;
  let outputX: u32 = outputPixel % shape.outputWidth;
  let outputY: u32 = (outputPixel / shape.outputWidth) % shape.outputHeight;
  let batch: u32 = outputPixel / (shape.outputWidth * shape.outputHeight);
  let value: f32 = 0.0;
  let found: bool = false;
  for (
    let windowY: u32 = 0;
    windowY < shape.filterHeight;
    windowY = windowY + 1
  ) {
    let inputY: i32 =
      i32(outputY * shape.strideY + windowY) - i32(shape.padTop);
    if (inputY < 0 || inputY >= i32(shape.height)) {
      continue;
    }
    for (
      let windowX: u32 = 0;
      windowX < shape.filterWidth;
      windowX = windowX + 1
    ) {
      let inputX: i32 =
        i32(outputX * shape.strideX + windowX) - i32(shape.padLeft);
      if (inputX < 0 || inputX >= i32(shape.width)) {
        continue;
      }
      let index: u32 =
        ((batch * shape.height + u32(inputY)) * shape.width + u32(inputX)) *
          shape.channels +
        channel;
      let candidate: f32 = input[index];
      if (!found || candidate > value) {
        value = candidate;
        found = true;
      }
    }
  }
  output[gid.x] = value;
}
