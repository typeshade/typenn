"use typeshade";

// Binary operations. Each operation has three kernel functions, one for each way its operands
// meet the shape of the output:
//
//   add(out, a, b)                                          a, b and out have one shape
//   addExpand(out, a, b, rank, dims, bStrides)              b is broadcast to the shape of a
//   addStrided(out, a, b, rank, dims, aStrides, bStrides)   both operands are broadcast
//
// `dims` is the shape of the output, innermost axis first, and `strides` is the stride of each
// operand along each of those axes (0 where the operand is broadcast), both padded to 8 axes.
// A function reads an array at exactly `i` where it can. A call that reads the array another call
// wrote at any other index cannot share a dispatch with it (typeshade/fuse, F3), so `a` of
// `addExpand` flows through a chain of calls and `a` of `addStrided` does not. `power` is the
// name of `pow`, which is a TypeShade builtin.

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

/** The index of an operand for element `flat` of the output. */
function operandIndex(
  flat: u32,
  rank: u32,
  dims: array<u32, 8>,
  strides: array<u32, 8>,
): u32 {
  let remaining: u32 = flat;
  let index: u32 = 0;
  for (let axis: u32 = 0; axis < rank; axis++) {
    index += (remaining % dims[axis]) * strides[axis];
    remaining /= dims[axis];
  }
  return index;
}

/** PReLU: `value` where it is at least 0, else `alpha * value`. */
function prelu1(value: f32, alpha: f32): f32 {
  return select(alpha * value, value, value >= 0.0);
}

export function add(out: array<f32>, a: array<f32>, b: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = a[i] + b[i];
  }
}

export function addExpand(
  out: array<f32>,
  a: array<f32>,
  b: array<f32>,
  rank: u32,
  dims: array<u32, 8>,
  bStrides: array<u32, 8>,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = a[i] + b[operandIndex(i, rank, dims, bStrides)];
  }
}

export function addStrided(
  out: array<f32>,
  a: array<f32>,
  b: array<f32>,
  rank: u32,
  dims: array<u32, 8>,
  aStrides: array<u32, 8>,
  bStrides: array<u32, 8>,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] =
      a[operandIndex(i, rank, dims, aStrides)] +
      b[operandIndex(i, rank, dims, bStrides)];
  }
}

export function sub(out: array<f32>, a: array<f32>, b: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = a[i] - b[i];
  }
}

export function subExpand(
  out: array<f32>,
  a: array<f32>,
  b: array<f32>,
  rank: u32,
  dims: array<u32, 8>,
  bStrides: array<u32, 8>,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = a[i] - b[operandIndex(i, rank, dims, bStrides)];
  }
}

export function subStrided(
  out: array<f32>,
  a: array<f32>,
  b: array<f32>,
  rank: u32,
  dims: array<u32, 8>,
  aStrides: array<u32, 8>,
  bStrides: array<u32, 8>,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] =
      a[operandIndex(i, rank, dims, aStrides)] -
      b[operandIndex(i, rank, dims, bStrides)];
  }
}

export function mul(out: array<f32>, a: array<f32>, b: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = a[i] * b[i];
  }
}

export function mulExpand(
  out: array<f32>,
  a: array<f32>,
  b: array<f32>,
  rank: u32,
  dims: array<u32, 8>,
  bStrides: array<u32, 8>,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = a[i] * b[operandIndex(i, rank, dims, bStrides)];
  }
}

export function mulStrided(
  out: array<f32>,
  a: array<f32>,
  b: array<f32>,
  rank: u32,
  dims: array<u32, 8>,
  aStrides: array<u32, 8>,
  bStrides: array<u32, 8>,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] =
      a[operandIndex(i, rank, dims, aStrides)] *
      b[operandIndex(i, rank, dims, bStrides)];
  }
}

export function div(out: array<f32>, a: array<f32>, b: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = a[i] / b[i];
  }
}

export function divExpand(
  out: array<f32>,
  a: array<f32>,
  b: array<f32>,
  rank: u32,
  dims: array<u32, 8>,
  bStrides: array<u32, 8>,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = a[i] / b[operandIndex(i, rank, dims, bStrides)];
  }
}

export function divStrided(
  out: array<f32>,
  a: array<f32>,
  b: array<f32>,
  rank: u32,
  dims: array<u32, 8>,
  aStrides: array<u32, 8>,
  bStrides: array<u32, 8>,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] =
      a[operandIndex(i, rank, dims, aStrides)] /
      b[operandIndex(i, rank, dims, bStrides)];
  }
}

export function maximum(out: array<f32>, a: array<f32>, b: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = max(a[i], b[i]);
  }
}

export function maximumExpand(
  out: array<f32>,
  a: array<f32>,
  b: array<f32>,
  rank: u32,
  dims: array<u32, 8>,
  bStrides: array<u32, 8>,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = max(a[i], b[operandIndex(i, rank, dims, bStrides)]);
  }
}

export function maximumStrided(
  out: array<f32>,
  a: array<f32>,
  b: array<f32>,
  rank: u32,
  dims: array<u32, 8>,
  aStrides: array<u32, 8>,
  bStrides: array<u32, 8>,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = max(
      a[operandIndex(i, rank, dims, aStrides)],
      b[operandIndex(i, rank, dims, bStrides)],
    );
  }
}

export function minimum(out: array<f32>, a: array<f32>, b: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = min(a[i], b[i]);
  }
}

export function minimumExpand(
  out: array<f32>,
  a: array<f32>,
  b: array<f32>,
  rank: u32,
  dims: array<u32, 8>,
  bStrides: array<u32, 8>,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = min(a[i], b[operandIndex(i, rank, dims, bStrides)]);
  }
}

export function minimumStrided(
  out: array<f32>,
  a: array<f32>,
  b: array<f32>,
  rank: u32,
  dims: array<u32, 8>,
  aStrides: array<u32, 8>,
  bStrides: array<u32, 8>,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = min(
      a[operandIndex(i, rank, dims, aStrides)],
      b[operandIndex(i, rank, dims, bStrides)],
    );
  }
}

export function power(out: array<f32>, a: array<f32>, b: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = pow(a[i], b[i]);
  }
}

export function powerExpand(
  out: array<f32>,
  a: array<f32>,
  b: array<f32>,
  rank: u32,
  dims: array<u32, 8>,
  bStrides: array<u32, 8>,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = pow(a[i], b[operandIndex(i, rank, dims, bStrides)]);
  }
}

export function powerStrided(
  out: array<f32>,
  a: array<f32>,
  b: array<f32>,
  rank: u32,
  dims: array<u32, 8>,
  aStrides: array<u32, 8>,
  bStrides: array<u32, 8>,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = pow(
      a[operandIndex(i, rank, dims, aStrides)],
      b[operandIndex(i, rank, dims, bStrides)],
    );
  }
}

export function prelu(out: array<f32>, a: array<f32>, b: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = prelu1(a[i], b[i]);
  }
}

export function preluExpand(
  out: array<f32>,
  a: array<f32>,
  b: array<f32>,
  rank: u32,
  dims: array<u32, 8>,
  bStrides: array<u32, 8>,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = prelu1(a[i], b[operandIndex(i, rank, dims, bStrides)]);
  }
}

export function preluStrided(
  out: array<f32>,
  a: array<f32>,
  b: array<f32>,
  rank: u32,
  dims: array<u32, 8>,
  aStrides: array<u32, 8>,
  bStrides: array<u32, 8>,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = prelu1(
      a[operandIndex(i, rank, dims, aStrides)],
      b[operandIndex(i, rank, dims, bStrides)],
    );
  }
}

/**
 * The gradient of a binary operation with respect to `a` (`wrtA` is 1) or to `b` (0). `out` has
 * the shape of that operand, and `dy` the shape of the output: an element of `out` sums the
 * contributions of the `repeats` elements of the output that its operand element was broadcast to.
 */
export function binaryGrad(
  out: array<f32>,
  a: array<f32>,
  b: array<f32>,
  dy: array<f32>,
  rank: u32,
  dims: array<u32, 8>,
  aStrides: array<u32, 8>,
  bStrides: array<u32, 8>,
  repeats: u32,
  operation: BinaryOperation,
  wrtA: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    // The output element this element of the target meets first: the axes the target spans.
    let base: u32 = 0;
    let stride: u32 = 1;
    for (let axis: u32 = 0; axis < rank; axis++) {
      const targetStride = select(bStrides[axis], aStrides[axis], wrtA === 1);
      if (targetStride > 0) {
        base += ((i / targetStride) % dims[axis]) * stride;
      }
      stride *= dims[axis];
    }
    let total: f32 = 0.0;
    for (let repeat: u32 = 0; repeat < repeats; repeat++) {
      // Then the axes it was broadcast along, counted by `repeat`.
      let remaining: u32 = repeat;
      let at: u32 = base;
      stride = 1;
      for (let axis: u32 = 0; axis < rank; axis++) {
        const targetStride = select(bStrides[axis], aStrides[axis], wrtA === 1);
        if (targetStride === 0) {
          at += (remaining % dims[axis]) * stride;
          remaining /= dims[axis];
        }
        stride *= dims[axis];
      }
      const av = a[operandIndex(at, rank, dims, aStrides)];
      const bv = b[operandIndex(at, rank, dims, bStrides)];
      const upstream = dy[at];
      if (operation === BinaryOperation.Add) {
        total += upstream;
      } else if (operation === BinaryOperation.Subtract) {
        total += select(-upstream, upstream, wrtA === 1);
      } else if (operation === BinaryOperation.Multiply) {
        total += upstream * select(av, bv, wrtA === 1);
      } else if (operation === BinaryOperation.Divide) {
        if (wrtA === 1) {
          total += upstream / bv;
        } else {
          total -= (upstream * av) / (bv * bv);
        }
      } else if (operation === BinaryOperation.Maximum) {
        if (wrtA === 1 && av >= bv) {
          total += upstream;
        } else if (wrtA === 0 && bv > av) {
          total += upstream;
        }
      } else if (operation === BinaryOperation.Minimum) {
        if (wrtA === 1 && av <= bv) {
          total += upstream;
        } else if (wrtA === 0 && bv < av) {
          total += upstream;
        }
      } else if (operation === BinaryOperation.Power) {
        if (wrtA === 1) {
          total += upstream * bv * pow(av, bv - 1.0);
        } else {
          total += upstream * pow(av, bv) * log(av);
        }
      } else if (operation === BinaryOperation.PRelu) {
        if (wrtA === 1 && av >= 0.0) {
          total += upstream;
        } else if (wrtA === 1) {
          total += upstream * bv;
        } else if (av < 0.0) {
          total += upstream * av;
        }
      }
    }
    out[i] = total;
  }
}
