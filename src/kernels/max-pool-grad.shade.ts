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
  inputLength: u32;
}

declare const shape: uniform<Shape>;

declare const input: storage<array<f32>>;

declare const dy: storage<array<f32>>;

declare const output: storage<array<f32>, "read_write">;

@compute([64])
export function maxPoolGradKernel(@builtin("global_invocation_id") gid: vec3u) {
  if (gid.x >= shape.inputLength) {
    return;
  }
  let inputChannel: u32 = gid.x % shape.channels;
  let inputPixel: u32 = gid.x / shape.channels;
  let inputX: u32 = inputPixel % shape.width;
  let inputY: u32 = (inputPixel / shape.width) % shape.height;
  let batch: u32 = inputPixel / (shape.width * shape.height);

  let centerY: i32 = i32(inputY) + i32(shape.padTop);

  let centerX: i32 = i32(inputX) + i32(shape.padLeft);
  let strideY: i32 = i32(shape.strideY);
  let strideX: i32 = i32(shape.strideX);
  let minimumOutputYNumerator: i32 = centerY - i32(shape.filterHeight - 1);
  let minimumOutputXNumerator: i32 = centerX - i32(shape.filterWidth - 1);
  let firstOutputY: i32 = max(
    0,
    (minimumOutputYNumerator + strideY - 1) / strideY,
  );
  let firstOutputX: i32 = max(
    0,
    (minimumOutputXNumerator + strideX - 1) / strideX,
  );
  let lastOutputY: i32 = min(i32(shape.outputHeight - 1), centerY / strideY);
  let lastOutputX: i32 = min(i32(shape.outputWidth - 1), centerX / strideX);
  let inputBatchOffset: u32 =
    batch * shape.height * shape.width * shape.channels;
  let outputBatchOffset: u32 =
    batch * shape.outputHeight * shape.outputWidth * shape.channels;
  let total: f32 = 0.0;
  for (let outputY: i32 = firstOutputY; outputY <= lastOutputY; outputY += 1) {
    for (
      let outputX: i32 = firstOutputX;
      outputX <= lastOutputX;
      outputX += 1
    ) {
      let bestValue: f32 = -3.402823e38;
      let bestIndex: u32 = 0;
      let found: bool = false;
      for (let yOffset: u32 = 0; yOffset < shape.filterHeight; yOffset += 1) {
        let y: i32 = outputY * strideY + i32(yOffset) - i32(shape.padTop);
        if (y < 0 || y >= i32(shape.height)) {
          continue;
        }
        for (let xOffset: u32 = 0; xOffset < shape.filterWidth; xOffset += 1) {
          let x: i32 = outputX * strideX + i32(xOffset) - i32(shape.padLeft);
          if (x < 0 || x >= i32(shape.width)) {
            continue;
          }
          let candidateIndex: u32 =
            inputBatchOffset +
            (u32(y) * shape.width + u32(x)) * shape.channels +
            inputChannel;
          let candidate: f32 = input[candidateIndex];
          if (found == false || candidate > bestValue) {
            bestValue = candidate;
            bestIndex = candidateIndex;
            found = true;
          }
        }
      }
      let inputIndex: u32 = gid.x;
      if (found && bestIndex == inputIndex) {
        let outputIndex: u32 =
          outputBatchOffset +
          (u32(outputY) * shape.outputWidth + u32(outputX)) * shape.channels +
          inputChannel;
        total += dy[outputIndex];
      }
    }
  }
  output[gid.x] = total;
}
