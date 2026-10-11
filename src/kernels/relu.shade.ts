"use typeshade";

class Shape {
  length: u32;
}

declare const shape: uniform<Shape>;

declare const input: storage<array<f32>>;

declare const output: storage<array<f32>, "read_write">;

function typennReluF32(@inout result: f32, inputValue: f32): void {
  result = max(inputValue, 0.0);
}

@compute([64])
export function reluKernel(@builtin("global_invocation_id") gid: vec3u) {
  if (gid.x >= shape.length) {
    return;
  }
  let value: f32 = 0.0;
  typennReluF32(value, input[gid.x]);
  output[gid.x] = value;
}
