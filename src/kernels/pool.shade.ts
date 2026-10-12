"use typeshade";

// 2D pooling over NHWC input [batch, height, width, channels]. One invocation for each element of
// the array it writes, which loops over the window. The window is the one `poolingWindow` works
// out on the host. A window reaches over the edge where it is padded, and an element past the
// edge is not in it: a maximum is of the elements inside, and an average divides by their count.

export function maxPool(
  out: array<f32>,
  source: array<f32>,
  height: u32,
  width: u32,
  channels: u32,
  filterHeight: u32,
  filterWidth: u32,
  outHeight: u32,
  outWidth: u32,
  strideY: u32,
  strideX: u32,
  padTop: u32,
  padLeft: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const channel = i % channels;
    const outputPixel = i / channels;
    const outputX = outputPixel % outWidth;
    const outputY = (outputPixel / outWidth) % outHeight;
    const batch = outputPixel / (outWidth * outHeight);
    let value: f32 = 0.0;
    let found: bool = false;
    for (let windowY: u32 = 0; windowY < filterHeight; windowY++) {
      const inputY: i32 = i32(outputY * strideY + windowY) - i32(padTop);
      if (inputY < 0 || inputY >= i32(height)) {
        continue;
      }
      for (let windowX: u32 = 0; windowX < filterWidth; windowX++) {
        const inputX: i32 = i32(outputX * strideX + windowX) - i32(padLeft);
        if (inputX < 0 || inputX >= i32(width)) {
          continue;
        }
        const candidate =
          source[
            ((batch * height + u32(inputY)) * width + u32(inputX)) * channels +
              channel
          ];
        if (!found || candidate > value) {
          value = candidate;
          found = true;
        }
      }
    }
    out[i] = value;
  }
}

export function avgPool(
  out: array<f32>,
  source: array<f32>,
  height: u32,
  width: u32,
  channels: u32,
  filterHeight: u32,
  filterWidth: u32,
  outHeight: u32,
  outWidth: u32,
  strideY: u32,
  strideX: u32,
  padTop: u32,
  padLeft: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const channel = i % channels;
    const outputPixel = i / channels;
    const outputX = outputPixel % outWidth;
    const outputY = (outputPixel / outWidth) % outHeight;
    const batch = outputPixel / (outWidth * outHeight);
    let total: f32 = 0.0;
    let count: u32 = 0;
    for (let windowY: u32 = 0; windowY < filterHeight; windowY++) {
      const inputY: i32 = i32(outputY * strideY + windowY) - i32(padTop);
      if (inputY < 0 || inputY >= i32(height)) {
        continue;
      }
      for (let windowX: u32 = 0; windowX < filterWidth; windowX++) {
        const inputX: i32 = i32(outputX * strideX + windowX) - i32(padLeft);
        if (inputX < 0 || inputX >= i32(width)) {
          continue;
        }
        total +=
          source[
            ((batch * height + u32(inputY)) * width + u32(inputX)) * channels +
              channel
          ];
        count += 1;
      }
    }
    out[i] = total / f32(count);
  }
}

/**
 * The gradient of `maxPool` with respect to its input: one invocation for each input element,
 * which finds the windows that contain it and adds the gradient of each window it is the first
 * greatest element of.
 */
export function maxPoolGrad(
  out: array<f32>,
  source: array<f32>,
  dy: array<f32>,
  height: u32,
  width: u32,
  channels: u32,
  filterHeight: u32,
  filterWidth: u32,
  outHeight: u32,
  outWidth: u32,
  strideY: u32,
  strideX: u32,
  padTop: u32,
  padLeft: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const channel = i % channels;
    const inputPixel = i / channels;
    const inputX = inputPixel % width;
    const inputY = (inputPixel / width) % height;
    const batch = inputPixel / (width * height);
    const centerY: i32 = i32(inputY) + i32(padTop);
    const centerX: i32 = i32(inputX) + i32(padLeft);
    const stepY: i32 = i32(strideY);
    const stepX: i32 = i32(strideX);
    const firstOutputY: i32 = max(
      0,
      (centerY - i32(filterHeight - 1) + stepY - 1) / stepY,
    );
    const firstOutputX: i32 = max(
      0,
      (centerX - i32(filterWidth - 1) + stepX - 1) / stepX,
    );
    const lastOutputY: i32 = min(i32(outHeight - 1), centerY / stepY);
    const lastOutputX: i32 = min(i32(outWidth - 1), centerX / stepX);
    const inputBatchOffset = batch * height * width * channels;
    const outputBatchOffset = batch * outHeight * outWidth * channels;
    let total: f32 = 0.0;
    for (
      let outputY: i32 = firstOutputY;
      outputY <= lastOutputY;
      outputY += 1
    ) {
      for (
        let outputX: i32 = firstOutputX;
        outputX <= lastOutputX;
        outputX += 1
      ) {
        let bestValue: f32 = -3.402823e38;
        let bestIndex: u32 = 0;
        let found: bool = false;
        for (let yOffset: u32 = 0; yOffset < filterHeight; yOffset++) {
          const y: i32 = outputY * stepY + i32(yOffset) - i32(padTop);
          if (y < 0 || y >= i32(height)) {
            continue;
          }
          for (let xOffset: u32 = 0; xOffset < filterWidth; xOffset++) {
            const x: i32 = outputX * stepX + i32(xOffset) - i32(padLeft);
            if (x < 0 || x >= i32(width)) {
              continue;
            }
            const candidateIndex =
              inputBatchOffset + (u32(y) * width + u32(x)) * channels + channel;
            const candidate = source[candidateIndex];
            if (found === false || candidate > bestValue) {
              bestValue = candidate;
              bestIndex = candidateIndex;
              found = true;
            }
          }
        }
        if (found && bestIndex === i) {
          total +=
            dy[
              outputBatchOffset +
                (u32(outputY) * outWidth + u32(outputX)) * channels +
                channel
            ];
        }
      }
    }
    out[i] = total;
  }
}

/** The gradient of `avgPool` with respect to its input: each window shares its gradient. */
export function avgPoolGrad(
  out: array<f32>,
  dy: array<f32>,
  height: u32,
  width: u32,
  channels: u32,
  filterHeight: u32,
  filterWidth: u32,
  outHeight: u32,
  outWidth: u32,
  strideY: u32,
  strideX: u32,
  padTop: u32,
  padLeft: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const channel = i % channels;
    const inputPixel = i / channels;
    const inputX = inputPixel % width;
    const inputY = (inputPixel / width) % height;
    const batch = inputPixel / (width * height);
    const centerY: i32 = i32(inputY) + i32(padTop);
    const centerX: i32 = i32(inputX) + i32(padLeft);
    const stepY: i32 = i32(strideY);
    const stepX: i32 = i32(strideX);
    const numeratorY: i32 = centerY - i32(filterHeight - 1);
    const numeratorX: i32 = centerX - i32(filterWidth - 1);
    let firstOutputY: i32 = 0;
    if (numeratorY > 0) {
      firstOutputY = (numeratorY + stepY - 1) / stepY;
    }
    let firstOutputX: i32 = 0;
    if (numeratorX > 0) {
      firstOutputX = (numeratorX + stepX - 1) / stepX;
    }
    const lastOutputY: i32 = min(i32(outHeight - 1), centerY / stepY);
    const lastOutputX: i32 = min(i32(outWidth - 1), centerX / stepX);
    const outputBatchOffset = batch * outHeight * outWidth * channels;
    let total: f32 = 0.0;
    for (
      let outputY: i32 = firstOutputY;
      outputY <= lastOutputY;
      outputY += 1
    ) {
      const windowStartY: i32 = outputY * stepY - i32(padTop);
      const validHeight: i32 =
        min(windowStartY + i32(filterHeight), i32(height)) -
        max(windowStartY, 0);
      for (
        let outputX: i32 = firstOutputX;
        outputX <= lastOutputX;
        outputX += 1
      ) {
        const windowStartX: i32 = outputX * stepX - i32(padLeft);
        const validWidth: i32 =
          min(windowStartX + i32(filterWidth), i32(width)) -
          max(windowStartX, 0);
        total +=
          dy[
            outputBatchOffset +
              (u32(outputY) * outWidth + u32(outputX)) * channels +
              channel
          ] / f32(validHeight * validWidth);
      }
    }
    out[i] = total;
  }
}
