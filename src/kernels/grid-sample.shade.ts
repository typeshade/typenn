"use typeshade";

// Inference-only spatial sampling for NHWC tensors. Grid coordinates are normalized x/y pairs.
// Out-of-bounds taps contribute zero, as in torch.nn.functional.grid_sample.
export function gridSampleNHWC(
  out: array<f32>,
  source: array<f32>,
  grid: array<f32>,
  height: u32,
  width: u32,
  channels: u32,
  outHeight: u32,
  outWidth: u32,
  alignCorners: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const channel = i % channels;
    const outPixel = i / channels;
    const outX = outPixel % outWidth;
    const outY = (outPixel / outWidth) % outHeight;
    const b = outPixel / (outWidth * outHeight);
    const gridIndex = (b * outHeight * outWidth + outY * outWidth + outX) * 2;
    const gx = grid[gridIndex];
    const gy = grid[gridIndex + 1];

    // Classify exponent-all-ones values by their IEEE-754 bits. Arithmetic NaN checks can be
    // folded under GPU fast-math, and subtraction does not reliably manufacture a NaN there.
    const gxBits = bitcast<u32>(gx);
    const gyBits = bitcast<u32>(gy);
    const gxExponent = gxBits & 0x7f800000;
    const gyExponent = gyBits & 0x7f800000;
    if (gxExponent === 0x7f800000 || gyExponent === 0x7f800000) {
      const nonFiniteBits = select(gyBits, gxBits, gxExponent === 0x7f800000);
      out[i] = bitcast<f32>(nonFiniteBits | 0x00400000);
      continue;
    }

    let x: f32 = 0.0;
    let y: f32 = 0.0;
    if (alignCorners !== 0) {
      x = (gx + 1.0) * f32(width - 1) * 0.5;
      y = (gy + 1.0) * f32(height - 1) * 0.5;
    } else {
      x = ((gx + 1.0) * f32(width) - 1.0) * 0.5;
      y = ((gy + 1.0) * f32(height) - 1.0) * 0.5;
    }

    // Avoid undefined integer conversion for finite coordinates far outside the image.
    if (x < -1.0 || x > f32(width) || y < -1.0 || y > f32(height)) {
      out[i] = 0.0;
      continue;
    }

    const x0 = i32(floor(x));
    const y0 = i32(floor(y));
    const x1 = x0 + 1;
    const y1 = y0 + 1;
    const wx = x - f32(x0);
    const wy = y - f32(y0);
    const bOffset = b * height * width * channels;
    let value: f32 = 0.0;
    if (x0 >= 0 && x0 < i32(width) && y0 >= 0 && y0 < i32(height)) {
      value +=
        source[bOffset + (u32(y0) * width + u32(x0)) * channels + channel] *
        (1.0 - wx) *
        (1.0 - wy);
    }
    if (x1 >= 0 && x1 < i32(width) && y0 >= 0 && y0 < i32(height)) {
      value +=
        source[bOffset + (u32(y0) * width + u32(x1)) * channels + channel] *
        wx *
        (1.0 - wy);
    }
    if (x0 >= 0 && x0 < i32(width) && y1 >= 0 && y1 < i32(height)) {
      value +=
        source[bOffset + (u32(y1) * width + u32(x0)) * channels + channel] *
        (1.0 - wx) *
        wy;
    }
    if (x1 >= 0 && x1 < i32(width) && y1 >= 0 && y1 < i32(height)) {
      value +=
        source[bOffset + (u32(y1) * width + u32(x1)) * channels + channel] *
        wx *
        wy;
    }
    out[i] = value;
  }
}

// PyTorch F.interpolate(..., mode="bilinear") uses edge-value padding. The caller supplies the
// output dimensions; coordinate transformation follows align_corners exactly.
export function resizeBilinear(
  out: array<f32>,
  source: array<f32>,
  height: u32,
  width: u32,
  channels: u32,
  outHeight: u32,
  outWidth: u32,
  alignCorners: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const channel = i % channels;
    const outPixel = i / channels;
    const outX = outPixel % outWidth;
    const outY = (outPixel / outWidth) % outHeight;
    const b = outPixel / (outWidth * outHeight);
    let x: f32 = 0.0;
    let y: f32 = 0.0;
    if (alignCorners !== 0) {
      if (outWidth > 1) x = (f32(outX) * f32(width - 1)) / f32(outWidth - 1);
      if (outHeight > 1) y = (f32(outY) * f32(height - 1)) / f32(outHeight - 1);
    } else {
      x = max(((f32(outX) + 0.5) * f32(width)) / f32(outWidth) - 0.5, 0.0);
      y = max(((f32(outY) + 0.5) * f32(height)) / f32(outHeight) - 0.5, 0.0);
    }
    x = min(x, f32(width - 1));
    y = min(y, f32(height - 1));
    const x0 = u32(floor(x));
    const y0 = u32(floor(y));
    const x1 = min(x0 + 1, width - 1);
    const y1 = min(y0 + 1, height - 1);
    const wx = x - f32(x0);
    const wy = y - f32(y0);
    const bOffset = b * height * width * channels;
    const v00 = source[bOffset + (y0 * width + x0) * channels + channel];
    const v10 = source[bOffset + (y0 * width + x1) * channels + channel];
    const v01 = source[bOffset + (y1 * width + x0) * channels + channel];
    const v11 = source[bOffset + (y1 * width + x1) * channels + channel];
    out[i] =
      (v00 * (1.0 - wx) + v10 * wx) * (1.0 - wy) +
      (v01 * (1.0 - wx) + v11 * wx) * wy;
  }
}
