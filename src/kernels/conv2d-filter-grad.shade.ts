"use typeshade";

class Shape {
  batch: u32;
  height: u32;
  width: u32;
  inputChannels: u32;
  kernelWidth: u32;
  outputChannels: u32;
  outputHeight: u32;
  outputWidth: u32;
  strideY: u32;
  strideX: u32;
  padTop: u32;
  padLeft: u32;
  weightLength: u32;
}

declare const shape: uniform<Shape>;

declare const input: storage<array<f32>>;

declare const dy: storage<array<f32>>;

declare const output: storage<array<f32>, "read_write">;

@compute([64])
export function conv2dFilterGradKernel(@builtin("global_invocation_id") gid: vec3u) {
  if (gid.x >= shape.weightLength) {
    return;
  }
  let oc: u32 = gid.x % shape.outputChannels;
  let ic: u32 = (gid.x / shape.outputChannels) % shape.inputChannels;

  let kx: u32 =
    (gid.x / (shape.outputChannels * shape.inputChannels)) % shape.kernelWidth;
  let ky: u32 =
    gid.x / (shape.outputChannels * shape.inputChannels * shape.kernelWidth);
  let total: f32 = 0.0;
  for (let b: u32 = 0; b < shape.batch; b = b + 1) {
    for (let oy: u32 = 0; oy < shape.outputHeight; oy = oy + 1) {
      let iy: i32 = i32(oy * shape.strideY + ky) - i32(shape.padTop);
      if (iy < 0 || iy >= i32(shape.height)) {
        continue;
      }
      for (let ox: u32 = 0; ox < shape.outputWidth; ox = ox + 1) {
        let ix: i32 = i32(ox * shape.strideX + kx) - i32(shape.padLeft);
        if (ix < 0 || ix >= i32(shape.width)) {
          continue;
        }
        let ii: u32 =
          ((b * shape.height + u32(iy)) * shape.width + u32(ix)) *
            shape.inputChannels +
          ic;
        let oi: u32 =
          ((b * shape.outputHeight + oy) * shape.outputWidth + ox) *
            shape.outputChannels +
          oc;
        total += input[ii] * dy[oi];
      }
    }
  }
  output[gid.x] = total;
}
