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
declare const dy: storage<array<f32>>;
declare const output: storage<array<f32>, "read_write">;

@compute([64])
export function avgPoolGradKernel(@builtin("global_invocation_id") gid: vec3u) {
  if (gid.x >= shape.inputLength) {
    return;
  }

  let channel = gid.x % shape.channels;
  let inputPixel = gid.x / shape.channels;
  let inputX = inputPixel % shape.width;
  let inputY = (inputPixel / shape.width) % shape.height;
  let batch = inputPixel / (shape.width * shape.height);
  let centerY = i32(inputY) + i32(shape.padTop);
  let centerX = i32(inputX) + i32(shape.padLeft);
  let strideY = i32(shape.strideY);
  let strideX = i32(shape.strideX);
  let minimumOutputYNumerator = centerY - i32(shape.filterHeight - 1);
  let minimumOutputXNumerator = centerX - i32(shape.filterWidth - 1);
  let firstOutputY: i32 = 0;
  if (minimumOutputYNumerator > 0) {
    firstOutputY = (minimumOutputYNumerator + strideY - 1) / strideY;
  }
  let firstOutputX: i32 = 0;
  if (minimumOutputXNumerator > 0) {
    firstOutputX = (minimumOutputXNumerator + strideX - 1) / strideX;
  }
  let lastOutputY = min(i32(shape.outputHeight - 1), centerY / strideY);
  let lastOutputX = min(i32(shape.outputWidth - 1), centerX / strideX);
  let outputBatchOffset =
    batch * shape.outputHeight * shape.outputWidth * shape.channels;
  let total: f32 = 0.0;

  for (let outputY = firstOutputY; outputY <= lastOutputY; outputY += 1) {
    let windowStartY = outputY * strideY - i32(shape.padTop);
    let validStartY = max(windowStartY, 0);
    let validEndY = min(
      windowStartY + i32(shape.filterHeight),
      i32(shape.height),
    );
    let validHeight = validEndY - validStartY;

    for (let outputX = firstOutputX; outputX <= lastOutputX; outputX += 1) {
      let windowStartX = outputX * strideX - i32(shape.padLeft);
      let validStartX = max(windowStartX, 0);
      let validEndX = min(
        windowStartX + i32(shape.filterWidth),
        i32(shape.width),
      );
      let validWidth = validEndX - validStartX;
      let count = validHeight * validWidth;
      let outputIndex =
        outputBatchOffset +
        (u32(outputY) * shape.outputWidth + u32(outputX)) * shape.channels +
        channel;
      total += dy[outputIndex] / f32(count);
    }
  }

  output[gid.x] = total;
}
