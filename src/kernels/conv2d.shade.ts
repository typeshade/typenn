"use typeshade";

class Shape {
  height: u32;
  width: u32;
  inputChannels: u32;
  kernelHeight: u32;
  kernelWidth: u32;
  outputChannels: u32;
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

declare const weights: storage<array<f32>>;

declare const output: storage<array<f32>, "read_write">;

@compute([64])
export function conv2dKernel(@builtin("global_invocation_id") gid: vec3u) {
  if (gid.x >= shape.outputLength) {
    return;
  }
  let outputChannel: u32 = gid.x % shape.outputChannels;
  let outputPixel: u32 = gid.x / shape.outputChannels;
  let outputX: u32 = outputPixel % shape.outputWidth;
  let outputY: u32 = (outputPixel / shape.outputWidth) % shape.outputHeight;
  let batch: u32 = outputPixel / (shape.outputWidth * shape.outputHeight);
  let value: f32 = 0.0;
  for (
    let kernelY: u32 = 0;
    kernelY < shape.kernelHeight;
    kernelY = kernelY + 1
  ) {
    let inputY: i32 =
      i32(outputY * shape.strideY + kernelY) - i32(shape.padTop);
    if (inputY < 0 || inputY >= i32(shape.height)) {
      continue;
    }
    for (
      let kernelX: u32 = 0;
      kernelX < shape.kernelWidth;
      kernelX = kernelX + 1
    ) {
      let inputX: i32 =
        i32(outputX * shape.strideX + kernelX) - i32(shape.padLeft);
      if (inputX < 0 || inputX >= i32(shape.width)) {
        continue;
      }
      for (
        let inputChannel: u32 = 0;
        inputChannel < shape.inputChannels;
        inputChannel = inputChannel + 1
      ) {
        let inputIndex: u32 =
          ((batch * shape.height + u32(inputY)) * shape.width + u32(inputX)) *
            shape.inputChannels +
          inputChannel;
        let filterIndex: u32 =
          ((kernelY * shape.kernelWidth + kernelX) * shape.inputChannels +
            inputChannel) *
            shape.outputChannels +
          outputChannel;
        value += input[inputIndex] * weights[filterIndex];
      }
    }
  }
  output[gid.x] = value;
}
