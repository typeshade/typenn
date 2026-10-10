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
export function avgPoolKernel(@builtin("global_invocation_id") gid: vec3u) {
  if (gid.x >= shape.outputLength) {
    return;
  }

  let channel = gid.x % shape.channels;
  let outputPixel = gid.x / shape.channels;
  let outputX = outputPixel % shape.outputWidth;
  let outputY = (outputPixel / shape.outputWidth) % shape.outputHeight;
  let batch = outputPixel / (shape.outputWidth * shape.outputHeight);
  let total: f32 = 0.0;
  let count: u32 = 0;

  for (let windowY: u32 = 0; windowY < shape.filterHeight; windowY += 1) {
    let inputY = i32(outputY * shape.strideY + windowY) - i32(shape.padTop);
    if (inputY < 0 || inputY >= i32(shape.height)) {
      continue;
    }

    for (let windowX: u32 = 0; windowX < shape.filterWidth; windowX += 1) {
      let inputX = i32(outputX * shape.strideX + windowX) - i32(shape.padLeft);
      if (inputX < 0 || inputX >= i32(shape.width)) {
        continue;
      }

      let inputIndex =
        ((batch * shape.height + u32(inputY)) * shape.width + u32(inputX)) *
          shape.channels +
        channel;
      total += input[inputIndex];
      count += 1;
    }
  }

  output[gid.x] = total / f32(count);
}
