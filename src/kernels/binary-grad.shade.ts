"use typeshade";

export enum BinaryOperation {
  Add = 0,
  Subtract = 1,
  Multiply = 2,
  Divide = 3,
  Maximum = 4,
  Minimum = 5,
  Power = 6,
  PRelu = 7,
}

class Shape {
  rank: u32;
  dimensions: array<u32, 8>;
  aStrides: array<u32, 8>;
  bStrides: array<u32, 8>;
  targetLength: u32;
  repeatCount: u32;
  operation: BinaryOperation;
  wrtA: u32;
}

declare const shape: uniform<Shape>;

declare const a: storage<array<f32>>;

declare const b: storage<array<f32>>;

declare const dy: storage<array<f32>>;

declare const output: storage<array<f32>, "read_write">;

function operandIndex(flatIndex: u32, strides: array<u32, 8>): u32 {
  let remaining: u32 = flatIndex;
  let index: u32 = 0;
  for (let axis: u32 = 0; axis < shape.rank; axis = axis + 1) {
    let coordinate: u32 = remaining % shape.dimensions[axis];
    remaining /= shape.dimensions[axis];
    index += coordinate * strides[axis];
  }
  return index;
}

@compute([64])
export function binaryGradKernel(@builtin("global_invocation_id") gid: vec3u) {
  if (gid.x >= shape.targetLength) {
    return;
  }
  let targetStrides: array<u32, 8>;
  if (shape.wrtA == 1) {
    targetStrides = shape.aStrides;
  } else {
    targetStrides = shape.bStrides;
  }
  let baseOutput: u32 = 0;
  let outputStride: u32 = 1;
  for (let axis: u32 = 0; axis < shape.rank; axis = axis + 1) {
    if (targetStrides[axis] > 0) {
      let coordinate: u32 =
        (gid.x / targetStrides[axis]) % shape.dimensions[axis];
      baseOutput += coordinate * outputStride;
    }
    outputStride *= shape.dimensions[axis];
  }
  let total: f32 = 0.0;

  for (let repeat: u32 = 0; repeat < shape.repeatCount; repeat = repeat + 1) {
    let remaining: u32 = repeat;
    let outputIndex: u32 = baseOutput;
    outputStride = 1;
    for (
      let repeatAxis: u32 = 0;
      repeatAxis < shape.rank;
      repeatAxis = repeatAxis + 1
    ) {
      if (targetStrides[repeatAxis] == 0) {
        let repeatCoordinate: u32 = remaining % shape.dimensions[repeatAxis];
        remaining /= shape.dimensions[repeatAxis];
        outputIndex += repeatCoordinate * outputStride;
      }
      outputStride *= shape.dimensions[repeatAxis];
    }
    let aIndex: u32 = operandIndex(outputIndex, shape.aStrides);
    let bIndex: u32 = operandIndex(outputIndex, shape.bStrides);
    let av: f32 = a[aIndex];
    let bv: f32 = b[bIndex];
    let upstream: f32 = dy[outputIndex];
    switch (shape.operation) {
      case BinaryOperation.Add:
        total += upstream;
        break;
      case BinaryOperation.Subtract:
        if (shape.wrtA == 1) {
          total += upstream;
        } else {
          total -= upstream;
        }
        break;
      case BinaryOperation.Multiply:
        if (shape.wrtA == 1) {
          total += upstream * bv;
        } else {
          total += upstream * av;
        }
        break;
      case BinaryOperation.Divide:
        if (shape.wrtA == 1) {
          total += upstream / bv;
        } else {
          total -= (upstream * av) / (bv * bv);
        }
        break;
      case BinaryOperation.Maximum:
        if (shape.wrtA == 1 && av >= bv) {
          total += upstream;
        } else if (shape.wrtA == 0 && bv > av) {
          total += upstream;
        }
        break;
      case BinaryOperation.Minimum:
        if (shape.wrtA == 1 && av <= bv) {
          total += upstream;
        } else if (shape.wrtA == 0 && bv < av) {
          total += upstream;
        }
        break;
      case BinaryOperation.Power:
        if (shape.wrtA == 1) {
          total += upstream * bv * pow(av, bv - 1.0);
        } else {
          total += upstream * pow(av, bv) * log(av);
        }
        break;
      case BinaryOperation.PRelu:
        if (shape.wrtA == 1 && av >= 0.0) {
          total += upstream;
        } else if (shape.wrtA == 1) {
          total += upstream * bv;
        } else if (av < 0.0) {
          total += upstream * av;
        }
        break;
    }
  }
  output[gid.x] = total;
}
