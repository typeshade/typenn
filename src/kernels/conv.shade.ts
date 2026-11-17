"use typeshade";

// 2D convolution over NHWC input `x` [batch, height, width, inChannels] with an HWIO filter,
// `weights` [kernelHeight, kernelWidth, inChannels, outChannels]. One invocation for each element
// of the array it writes, which loops over the window. `outHeight`, `outWidth`, the strides and
// `padTop` and `padLeft` are the window of the operation, as `spatialWindow` works them out on
// the host. The filter is `weights` and not `filter`: `filter` is a reserved word of WGSL, and
// TypeShade 96dbaa9 names the storage binding of an array parameter after the parameter. Its
// compiler refuses a reserved name on an entry binding (TS8068), but not on a kernel parameter.

export function conv2d(
  out: array<f32>,
  x: array<f32>,
  weights: array<f32>,
  height: u32,
  width: u32,
  inChannels: u32,
  kernelHeight: u32,
  kernelWidth: u32,
  outChannels: u32,
  outHeight: u32,
  outWidth: u32,
  strideY: u32,
  strideX: u32,
  padTop: u32,
  padLeft: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const outputChannel = i % outChannels;
    const outputPixel = i / outChannels;
    const outputX = outputPixel % outWidth;
    const outputY = (outputPixel / outWidth) % outHeight;
    const batch = outputPixel / (outWidth * outHeight);
    let value: f32 = 0.0;
    for (let kernelY: u32 = 0; kernelY < kernelHeight; kernelY++) {
      const inputY: i32 = i32(outputY * strideY + kernelY) - i32(padTop);
      if (inputY < 0 || inputY >= i32(height)) {
        continue;
      }
      for (let kernelX: u32 = 0; kernelX < kernelWidth; kernelX++) {
        const inputX: i32 = i32(outputX * strideX + kernelX) - i32(padLeft);
        if (inputX < 0 || inputX >= i32(width)) {
          continue;
        }
        for (let channel: u32 = 0; channel < inChannels; channel++) {
          const inputIndex =
            ((batch * height + u32(inputY)) * width + u32(inputX)) *
              inChannels +
            channel;
          const filterIndex =
            ((kernelY * kernelWidth + kernelX) * inChannels + channel) *
              outChannels +
            outputChannel;
          value += x[inputIndex] * weights[filterIndex];
        }
      }
    }
    out[i] = value;
  }
}

/** The gradient of `conv2d` with respect to its input: one invocation for each input element. */
export function conv2dInputGrad(
  out: array<f32>,
  weights: array<f32>,
  dy: array<f32>,
  height: u32,
  width: u32,
  inChannels: u32,
  kernelHeight: u32,
  kernelWidth: u32,
  outChannels: u32,
  outHeight: u32,
  outWidth: u32,
  strideY: u32,
  strideX: u32,
  padTop: u32,
  padLeft: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const inputChannel = i % inChannels;
    const inputPixel = i / inChannels;
    const inputX = inputPixel % width;
    const inputY = (inputPixel / width) % height;
    const batch = inputPixel / (width * height);
    const centerY: i32 = i32(inputY) + i32(padTop);
    const centerX: i32 = i32(inputX) + i32(padLeft);
    const stepY: i32 = i32(strideY);
    const stepX: i32 = i32(strideX);
    const firstKernelY: i32 = max(0, centerY - i32(outHeight - 1) * stepY);
    const lastKernelY: i32 = min(i32(kernelHeight - 1), centerY);
    const firstKernelX: i32 = max(0, centerX - i32(outWidth - 1) * stepX);
    const lastKernelX: i32 = min(i32(kernelWidth - 1), centerX);
    const batchOffset = batch * outHeight * outWidth * outChannels;
    const channelOffset = inputChannel * outChannels;
    let total: f32 = 0.0;
    for (
      let kernelY: i32 = firstKernelY;
      kernelY <= lastKernelY;
      kernelY += 1
    ) {
      if ((centerY - kernelY) % stepY != 0) {
        continue;
      }
      const outputY = u32((centerY - kernelY) / stepY);
      for (
        let kernelX: i32 = firstKernelX;
        kernelX <= lastKernelX;
        kernelX += 1
      ) {
        if ((centerX - kernelX) % stepX != 0) {
          continue;
        }
        const outputX = u32((centerX - kernelX) / stepX);
        const outputOffset =
          batchOffset + (outputY * outWidth + outputX) * outChannels;
        const kernelOffset =
          (u32(kernelY) * kernelWidth + u32(kernelX)) *
            inChannels *
            outChannels +
          channelOffset;
        for (let channel: u32 = 0; channel < outChannels; channel++) {
          total += dy[outputOffset + channel] * weights[kernelOffset + channel];
        }
      }
    }
    out[i] = total;
  }
}

/** The gradient of `conv2d` with respect to its filter: one invocation for each filter element. */
export function conv2dFilterGrad(
  out: array<f32>,
  x: array<f32>,
  dy: array<f32>,
  batch: u32,
  height: u32,
  width: u32,
  inChannels: u32,
  kernelWidth: u32,
  outChannels: u32,
  outHeight: u32,
  outWidth: u32,
  strideY: u32,
  strideX: u32,
  padTop: u32,
  padLeft: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const oc = i % outChannels;
    const ic = (i / outChannels) % inChannels;
    const kx = (i / (outChannels * inChannels)) % kernelWidth;
    const ky = i / (outChannels * inChannels * kernelWidth);
    let total: f32 = 0.0;
    for (let b: u32 = 0; b < batch; b++) {
      for (let oy: u32 = 0; oy < outHeight; oy++) {
        const iy: i32 = i32(oy * strideY + ky) - i32(padTop);
        if (iy < 0 || iy >= i32(height)) {
          continue;
        }
        for (let ox: u32 = 0; ox < outWidth; ox++) {
          const ix: i32 = i32(ox * strideX + kx) - i32(padLeft);
          if (ix < 0 || ix >= i32(width)) {
            continue;
          }
          const inputIndex =
            ((b * height + u32(iy)) * width + u32(ix)) * inChannels + ic;
          const outputIndex =
            ((b * outHeight + oy) * outWidth + ox) * outChannels + oc;
          total += x[inputIndex] * dy[outputIndex];
        }
      }
    }
    out[i] = total;
  }
}
