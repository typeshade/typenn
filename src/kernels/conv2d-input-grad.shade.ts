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
  inputLength: u32;
}

declare const shape: uniform<Shape>;

declare const weights: storage<array<f32>>;

declare const dy: storage<array<f32>>;

declare const output: storage<array<f32>, "read_write">;

@compute([64])
export function conv2dInputGradKernel(@builtin("global_invocation_id") gid: vec3u) {
  if (gid.x >= shape.inputLength) {
    return;
  }
  let inputChannel: u32 = gid.x % shape.inputChannels;
  let inputPixel: u32 = gid.x / shape.inputChannels;
  let inputX: u32 = inputPixel % shape.width;
  let inputY: u32 = (inputPixel / shape.width) % shape.height;
  let batch: u32 = inputPixel / (shape.width * shape.height);
  let centerY: i32 = i32(inputY) + i32(shape.padTop);
  let centerX: i32 = i32(inputX) + i32(shape.padLeft);
  let strideY: i32 = i32(shape.strideY);
  let strideX: i32 = i32(shape.strideX);
  let minimumKernelY: i32 = max(
    0,
    centerY - i32(shape.outputHeight - 1) * strideY,
  );
  let maximumKernelY: i32 = min(i32(shape.kernelHeight - 1), centerY);
  let minimumKernelX: i32 = max(
    0,
    centerX - i32(shape.outputWidth - 1) * strideX,
  );
  let maximumKernelX: i32 = min(i32(shape.kernelWidth - 1), centerX);
  let outputSpatialSize: u32 = shape.outputHeight * shape.outputWidth;
  let inputBatchOffset: u32 = batch * outputSpatialSize * shape.outputChannels;
  let weightInputChannelOffset: u32 = inputChannel * shape.outputChannels;
  let total: f32 = 0.0;
  for (
    let kernelY: i32 = minimumKernelY;
    kernelY <= maximumKernelY;
    kernelY += 1
  ) {
    if ((centerY - kernelY) % strideY != 0) {
      continue;
    }
    let outputY: u32 = u32((centerY - kernelY) / strideY);
    for (
      let kernelX: i32 = minimumKernelX;
      kernelX <= maximumKernelX;
      kernelX += 1
    ) {
      if ((centerX - kernelX) % strideX != 0) {
        continue;
      }
      let outputX: u32 = u32((centerX - kernelX) / strideX);
      let outputPixel: u32 = outputY * shape.outputWidth + outputX;
      let outputOffset: u32 =
        inputBatchOffset + outputPixel * shape.outputChannels;
      let kernelOffset: u32 =
        (u32(kernelY) * shape.kernelWidth + u32(kernelX)) *
          shape.inputChannels *
          shape.outputChannels +
        weightInputChannelOffset;
      for (
        let outputChannel: u32 = 0;
        outputChannel < shape.outputChannels;
        outputChannel += 1
      ) {
        total +=
          dy[outputOffset + outputChannel] *
          weights[kernelOffset + outputChannel];
      }
    }
  }
  output[gid.x] = total;
}
