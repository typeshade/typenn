// Plain-JavaScript references for the operations of TypeNN, and the seeded inputs they run on.
// The references round to f32 after every operation (`Math.fround`), as the kernel functions do
// on the CPU tier, and they share nothing with the library. Pure ES module: the Node tests and
// the browser test import it alike.

export const f32 = Math.fround;

/** A seeded generator of numbers in [0, 1): mulberry32. */
export function random(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** `count` seeded f32 values in [low, high]. */
export function values(seed, count, low, high) {
  const next = random(seed);
  return Float32Array.from({ length: count }, () =>
    f32(low + (high - low) * next()),
  );
}

/** Values that pin the edges: zeros, halves, integers and a few tiny numbers. */
export const EDGES = [
  0, -0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 1, -1, 3, 0.005, -0.005,
];

export function sizeOf(shape) {
  return shape.reduce((size, axis) => size * axis, 1);
}

/** The first difference between two arrays, or undefined. A tolerance of 0 asks for the same
 *  value (a NaN equals a NaN, and a zero equals a zero of the other sign); a number is a
 *  relative bound with a floor of 1 on the size of the expected value, so that values near 0
 *  have an absolute bound. */
export function difference(actual, expected, tolerance = 0, label = "value") {
  if (actual.length !== expected.length)
    return `${label}: length ${actual.length}, expected ${expected.length}`;
  for (let index = 0; index < expected.length; index++) {
    const a = actual[index];
    const e = expected[index];
    const same =
      a === e ||
      (a !== a && e !== e) ||
      (tolerance !== 0 &&
        Math.abs(a - e) <= tolerance * Math.max(1, Math.abs(e)));
    if (!same) return `${label}[${index}]: got ${a}, expected ${e}`;
  }
  return undefined;
}

// ─── maps ─────────────────────────────────────────────────────────────────────────────────────

const roundHalfEven = (x) => {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
};
const sign = (x) => (x > 0 ? 1 : x < 0 ? -1 : 0);

/**
 * Each map of one array: its reference, the range of inputs it is defined on, its relative
 * tolerance on the CPU tier (0: the same bits) and on a GPU (`gpu`, when it differs: a GPU
 * divides and takes roots to a few ULP, and its `exp` and `sin` are not the CPU's).
 */
export const MAPS = {
  neg: { f: (x) => -x, range: [-4, 4], tolerance: 0 },
  abs: { f: (x) => Math.abs(x), range: [-4, 4], tolerance: 0 },
  square: { f: (x) => f32(x * x), range: [-4, 4], tolerance: 0 },
  sqrt: {
    f: (x) => f32(Math.sqrt(x)),
    range: [0, 16],
    tolerance: 0,
    gpu: 1e-5,
  },
  exp: { f: (x) => f32(Math.exp(x)), range: [-5, 5], tolerance: 1e-5 },
  log: { f: (x) => f32(Math.log(x)), range: [0.01, 16], tolerance: 1e-5 },
  sigmoid: {
    f: (x) => f32(1 / (1 + Math.exp(-x))),
    range: [-8, 8],
    tolerance: 1e-5,
  },
  tanh: { f: (x) => f32(Math.tanh(x)), range: [-4, 4], tolerance: 1e-5 },
  floor: { f: (x) => Math.floor(x), range: [-4, 4], tolerance: 0 },
  ceil: { f: (x) => Math.ceil(x), range: [-4, 4], tolerance: 0 },
  round: { f: roundHalfEven, range: [-4, 4], tolerance: 0 },
  sign: { f: sign, range: [-4, 4], tolerance: 0 },
  reciprocal: {
    f: (x) => f32(1 / x),
    range: [0.1, 4],
    tolerance: 0,
    gpu: 1e-5,
  },
  rsqrt: { f: (x) => f32(1 / Math.sqrt(x)), range: [0.1, 16], tolerance: 1e-5 },
  sin: { f: (x) => f32(Math.sin(x)), range: [-3, 3], tolerance: 1e-5 },
  cos: { f: (x) => f32(Math.cos(x)), range: [-3, 3], tolerance: 1e-5 },
  tan: { f: (x) => f32(Math.tan(x)), range: [-1.2, 1.2], tolerance: 1e-5 },
  expm1: { f: (x) => f32(Math.expm1(x)), range: [-2, 2], tolerance: 1e-5 },
  log1p: { f: (x) => f32(Math.log1p(x)), range: [-0.9, 3], tolerance: 1e-5 },
  sinh: { f: (x) => f32(Math.sinh(x)), range: [-3, 3], tolerance: 1e-5 },
  cosh: { f: (x) => f32(Math.cosh(x)), range: [-3, 3], tolerance: 1e-5 },
  acos: { f: (x) => f32(Math.acos(x)), range: [-0.99, 0.99], tolerance: 1e-5 },
  acosh: { f: (x) => f32(Math.acosh(x)), range: [1.01, 8], tolerance: 1e-5 },
  asin: { f: (x) => f32(Math.asin(x)), range: [-0.99, 0.99], tolerance: 1e-5 },
  asinh: { f: (x) => f32(Math.asinh(x)), range: [-5, 5], tolerance: 1e-5 },
  atan: { f: (x) => f32(Math.atan(x)), range: [-5, 5], tolerance: 1e-5 },
  atanh: {
    f: (x) => f32(Math.atanh(x)),
    range: [-0.99, 0.99],
    tolerance: 1e-5,
  },
  step: { f: (x) => (x < 0 ? 0 : 1), range: [-2, 2], tolerance: 0 },
  relu: { f: (x) => Math.max(x, 0), range: [-4, 4], tolerance: 0 },
  relu6: {
    f: (x) => Math.min(Math.max(x, 0), 6),
    range: [-3, 9],
    tolerance: 0,
  },
  softplus: {
    f: (x) => f32(Math.max(x, 0) + Math.log(1 + Math.exp(-Math.abs(x)))),
    range: [-6, 6],
    tolerance: 1e-5,
  },
  gelu: {
    f: (x) => {
      const scaled = 0.7978845608 * (x + 0.044715 * x * x * x);
      const t =
        scaled > 10
          ? 1
          : scaled < -10
            ? -1
            : 2 / (1 + Math.exp(-2 * scaled)) - 1;
      return f32(0.5 * x * (1 + t));
    },
    range: [-5, 5],
    tolerance: 1e-5,
  },
};

/** The maps with a scalar: the call that makes the tensor, and its reference. */
export const SCALED_MAPS = {
  "leakyRelu(0.2)": {
    call: (t) => t.leakyRelu(0.2),
    f: (x) => (x >= 0 ? x : f32(f32(0.2) * x)),
    range: [-4, 4],
    tolerance: 0,
  },
  "elu(1.5)": {
    call: (t) => t.elu(1.5),
    f: (x) => (x >= 0 ? x : f32(1.5 * Math.expm1(x))),
    range: [-4, 4],
    tolerance: 1e-5,
  },
  "clipByValue(-1, 2)": {
    call: (t) => t.clipByValue(-1, 2),
    f: (x) => Math.min(Math.max(x, f32(-1)), f32(2)),
    range: [-4, 4],
    tolerance: 0,
  },
};

/** The derivative of each map at `x`, in double precision, for the gradient references. */
export const DERIVATIVES = {
  neg: () => -1,
  abs: (x) => sign(x),
  square: (x) => 2 * x,
  sqrt: (x) => 0.5 / Math.sqrt(x),
  exp: (x) => Math.exp(x),
  log: (x) => 1 / x,
  sigmoid: (x) => {
    const s = 1 / (1 + Math.exp(-x));
    return s * (1 - s);
  },
  tanh: (x) => 1 - Math.tanh(x) ** 2,
  floor: () => 0,
  ceil: () => 0,
  round: () => 0,
  sign: () => 0,
  reciprocal: (x) => -1 / (x * x),
  rsqrt: (x) => -0.5 * x ** -1.5,
  sin: (x) => Math.cos(x),
  cos: (x) => -Math.sin(x),
  tan: (x) => 1 + Math.tan(x) ** 2,
  expm1: (x) => Math.exp(x),
  log1p: (x) => 1 / (1 + x),
  sinh: (x) => Math.cosh(x),
  cosh: (x) => Math.sinh(x),
  acos: (x) => -1 / Math.sqrt(1 - x * x),
  acosh: (x) => 1 / Math.sqrt(x * x - 1),
  asin: (x) => 1 / Math.sqrt(1 - x * x),
  asinh: (x) => 1 / Math.sqrt(x * x + 1),
  atan: (x) => 1 / (1 + x * x),
  atanh: (x) => 1 / (1 - x * x),
  step: () => 0,
  relu: (x) => (x > 0 ? 1 : 0),
};

// ─── binary operations and broadcasting ───────────────────────────────────────────────────────

/** Each binary operation: its reference on two f32 values, the range of its operands, and its
 *  tolerances (see `MAPS`). `power` takes a positive base. */
export const BINARY = {
  add: { f: (x, y) => f32(x + y), a: [-4, 4], b: [-4, 4], tolerance: 0 },
  sub: { f: (x, y) => f32(x - y), a: [-4, 4], b: [-4, 4], tolerance: 0 },
  mul: { f: (x, y) => f32(x * y), a: [-4, 4], b: [-4, 4], tolerance: 0 },
  div: {
    f: (x, y) => f32(x / y),
    a: [-4, 4],
    b: [0.5, 4],
    tolerance: 0,
    gpu: 1e-5,
  },
  maximum: {
    f: (x, y) => Math.max(x, y),
    a: [-4, 4],
    b: [-4, 4],
    tolerance: 0,
  },
  minimum: {
    f: (x, y) => Math.min(x, y),
    a: [-4, 4],
    b: [-4, 4],
    tolerance: 0,
  },
  pow: {
    f: (x, y) => f32(Math.pow(x, y)),
    a: [0.5, 3],
    b: [-2, 3],
    tolerance: 1e-5,
  },
  prelu: {
    f: (x, y) => (x >= 0 ? x : f32(y * x)),
    a: [-4, 4],
    b: [0, 1],
    tolerance: 0,
  },
};

/** The shapes the operands of a binary operation meet in: one shape, a row, a column, a scalar
 *  on either side, both operands broadcast, and rank 3. */
export const BROADCASTS = {
  "same shape": [
    [2, 3],
    [2, 3],
  ],
  "row to matrix": [[2, 3], [3]],
  "column to matrix": [
    [2, 3],
    [2, 1],
  ],
  "scalar to matrix": [[2, 3], []],
  "matrix to row": [[3], [2, 3]],
  "scalar first": [[], [2, 3]],
  "both operands": [
    [3, 1],
    [1, 4],
  ],
  "rank three": [
    [2, 1, 4],
    [3, 1],
  ],
};

export function broadcastShape(a, b) {
  const rank = Math.max(a.length, b.length);
  return Array.from({ length: rank }, (_, axis) => {
    const x = a[axis - (rank - a.length)] ?? 1;
    const y = b[axis - (rank - b.length)] ?? 1;
    return Math.max(x, y);
  });
}

/** The index in an operand of `shape` of element `flat` of an output of `outShape`. */
export function operandIndex(outShape, shape, flat) {
  const offset = outShape.length - shape.length;
  let index = 0;
  let stride = 1;
  for (let axis = shape.length - 1; axis >= 0; axis--) {
    const outAxis = axis + offset;
    const coordinate =
      Math.floor(flat / sizeOf(outShape.slice(outAxis + 1))) %
      outShape[outAxis];
    index += (shape[axis] === 1 ? 0 : coordinate) * stride;
    stride *= shape[axis];
  }
  return index;
}

export function binaryReference(op, aShape, a, bShape, b) {
  const shape = broadcastShape(aShape, bShape);
  const out = new Float32Array(sizeOf(shape));
  for (let flat = 0; flat < out.length; flat++)
    out[flat] = BINARY[op].f(
      a[operandIndex(shape, aShape, flat)],
      b[operandIndex(shape, bShape, flat)],
    );
  return out;
}

/** The gradient of a binary operation with respect to both operands, summed over the axes each
 *  was broadcast on, in double precision. */
export function binaryGradReference(op, aShape, a, bShape, b, dy) {
  const shape = broadcastShape(aShape, bShape);
  const ga = new Float64Array(a.length);
  const gb = new Float64Array(b.length);
  for (let flat = 0; flat < dy.length; flat++) {
    const i = operandIndex(shape, aShape, flat);
    const j = operandIndex(shape, bShape, flat);
    const x = a[i];
    const y = b[j];
    const d = dy[flat];
    switch (op) {
      case "add":
        ga[i] += d;
        gb[j] += d;
        break;
      case "sub":
        ga[i] += d;
        gb[j] -= d;
        break;
      case "mul":
        ga[i] += d * y;
        gb[j] += d * x;
        break;
      case "div":
        ga[i] += d / y;
        gb[j] -= (d * x) / (y * y);
        break;
      case "maximum":
        if (x >= y) ga[i] += d;
        if (y > x) gb[j] += d;
        break;
      case "minimum":
        if (x <= y) ga[i] += d;
        if (y < x) gb[j] += d;
        break;
      case "pow":
        ga[i] += d * y * Math.pow(x, y - 1);
        gb[j] += d * Math.pow(x, y) * Math.log(x);
        break;
      case "prelu":
        ga[i] += x >= 0 ? d : d * y;
        if (x < 0) gb[j] += d * x;
        break;
    }
  }
  return [Float32Array.from(ga), Float32Array.from(gb)];
}

// ─── reductions and matmul ────────────────────────────────────────────────────────────────────

/** The input of a reduction as [outer, axisSize, inner]; `axis` undefined reduces every axis. */
export function axisLayout(shape, axis) {
  if (axis === undefined)
    return { outer: 1, axisSize: sizeOf(shape), inner: 1, outShape: [] };
  const at = axis < 0 ? axis + shape.length : axis;
  return {
    outer: sizeOf(shape.slice(0, at)),
    axisSize: shape[at],
    inner: sizeOf(shape.slice(at + 1)),
    outShape: shape.filter((_, index) => index !== at),
  };
}

/** Each reduction: the reference step along the axis, in index order, as the kernel adds. */
export const REDUCTIONS = {
  sum: { tolerance: 0, gpu: 1e-5, range: [-3, 3] },
  mean: { tolerance: 0, gpu: 1e-5, range: [-3, 3] },
  prod: { tolerance: 0, gpu: 1e-5, range: [0.5, 1.5] },
  max: { tolerance: 0, range: [-3, 3] },
  min: { tolerance: 0, range: [-3, 3] },
  argMax: { tolerance: 0, range: [-3, 3] },
  argMin: { tolerance: 0, range: [-3, 3] },
};

export function reduceReference(op, shape, x, axis) {
  const { outer, axisSize, inner } = axisLayout(shape, axis);
  const out = new Float32Array(outer * inner);
  for (let o = 0; o < outer; o++) {
    for (let n = 0; n < inner; n++) {
      const at = (k) => x[(o * axisSize + k) * inner + n];
      let total = op === "prod" ? 1 : 0;
      let best = at(0);
      let bestIndex = 0;
      for (let k = 0; k < axisSize; k++) {
        const v = at(k);
        if (op === "sum" || op === "mean") total = f32(total + v);
        else if (op === "prod") total = f32(total * v);
        else if (op === "max") best = Math.max(best, v);
        else if (op === "min") best = Math.min(best, v);
        else if (op === "argMax" && v > best) [best, bestIndex] = [v, k];
        else if (op === "argMin" && v < best) [best, bestIndex] = [v, k];
      }
      out[o * inner + n] =
        op === "mean"
          ? f32(total / axisSize)
          : op === "sum" || op === "prod"
            ? total
            : op === "argMax" || op === "argMin"
              ? bestIndex
              : best;
    }
  }
  return out;
}

/** The gradient of a sum or a mean over `axis`, from the gradient of the output. */
export function reduceGradReference(op, shape, dy, axis) {
  const { axisSize, inner } = axisLayout(shape, axis);
  const scale = op === "mean" ? f32(1 / axisSize) : 1;
  return Float32Array.from({ length: sizeOf(shape) }, (_, i) =>
    f32(dy[Math.floor(i / (axisSize * inner)) * inner + (i % inner)] * scale),
  );
}

/** `op(a) op(b)` for rank-2 matrices; a transposed matrix is stored transposed. */
export function matmulShape(aShape, bShape, transposeA, transposeB) {
  const rows = transposeA ? aShape[1] : aShape[0];
  const inner = transposeA ? aShape[0] : aShape[1];
  const cols = transposeB ? bShape[0] : bShape[1];
  return { rows, inner, cols };
}

const entryA = (a, { rows, inner }, transposeA, i, k) =>
  a[transposeA ? k * rows + i : i * inner + k];
const entryB = (b, { inner, cols }, transposeB, k, j) =>
  b[transposeB ? j * inner + k : k * cols + j];

export function matmulReference(
  a,
  aShape,
  b,
  bShape,
  transposeA = false,
  transposeB = false,
) {
  const dims = matmulShape(aShape, bShape, transposeA, transposeB);
  const out = new Float32Array(dims.rows * dims.cols);
  for (let i = 0; i < dims.rows; i++)
    for (let j = 0; j < dims.cols; j++) {
      let total = 0;
      for (let k = 0; k < dims.inner; k++)
        total = f32(
          total +
            f32(
              entryA(a, dims, transposeA, i, k) *
                entryB(b, dims, transposeB, k, j),
            ),
        );
      out[i * dims.cols + j] = total;
    }
  return out;
}

/** The gradients of a matmul with respect to a and b, in the layouts a and b are stored in. */
export function matmulGradReference(
  a,
  aShape,
  b,
  bShape,
  transposeA,
  transposeB,
  dc,
) {
  const dims = matmulShape(aShape, bShape, transposeA, transposeB);
  const da = new Float64Array(a.length);
  const db = new Float64Array(b.length);
  for (let i = 0; i < dims.rows; i++)
    for (let j = 0; j < dims.cols; j++)
      for (let k = 0; k < dims.inner; k++) {
        const d = dc[i * dims.cols + j];
        da[transposeA ? k * dims.rows + i : i * dims.inner + k] +=
          d * entryB(b, dims, transposeB, k, j);
        db[transposeB ? j * dims.inner + k : k * dims.cols + j] +=
          d * entryA(a, dims, transposeA, i, k);
      }
  return [Float32Array.from(da), Float32Array.from(db)];
}

// ─── softmax and cross entropy ────────────────────────────────────────────────────────────────

/** The softmax, or the log softmax, of `x` over `axis`, in double precision. */
export function softmaxReference(x, shape, axis, log = false) {
  const { outer, axisSize, inner } = axisLayout(shape, axis);
  const out = new Float32Array(x.length);
  for (let o = 0; o < outer; o++)
    for (let n = 0; n < inner; n++) {
      const at = (k) => (o * axisSize + k) * inner + n;
      let peak = -Infinity;
      for (let k = 0; k < axisSize; k++) peak = Math.max(peak, x[at(k)]);
      let sum = 0;
      for (let k = 0; k < axisSize; k++) sum += Math.exp(x[at(k)] - peak);
      for (let k = 0; k < axisSize; k++)
        out[at(k)] = log
          ? x[at(k)] - peak - Math.log(sum)
          : Math.exp(x[at(k)] - peak) / sum;
    }
  return out;
}

/** The gradient of a softmax or a log softmax from its output `y` and the gradient `dy` of it. */
export function softmaxGradReference(y, dy, shape, axis, log = false) {
  const { outer, axisSize, inner } = axisLayout(shape, axis);
  const out = new Float32Array(y.length);
  for (let o = 0; o < outer; o++)
    for (let n = 0; n < inner; n++) {
      const at = (k) => (o * axisSize + k) * inner + n;
      let reduction = 0;
      for (let k = 0; k < axisSize; k++)
        reduction += log ? dy[at(k)] : dy[at(k)] * y[at(k)];
      for (let k = 0; k < axisSize; k++)
        out[at(k)] = log
          ? dy[at(k)] - Math.exp(y[at(k)]) * reduction
          : y[at(k)] * (dy[at(k)] - reduction);
    }
  return out;
}

/** The loss of each row of `logits` [rows, columns] against `labels` of the same shape, and the
 *  gradients of the loss with respect to both, from the gradient `dy` of each row's loss. */
export function crossEntropyReference(logits, labels, rows, columns, dy) {
  const losses = new Float32Array(rows);
  const gradientLogits = new Float32Array(logits.length);
  const gradientLabels = new Float32Array(logits.length);
  for (let row = 0; row < rows; row++) {
    const at = (c) => row * columns + c;
    let peak = -Infinity;
    for (let c = 0; c < columns; c++) peak = Math.max(peak, logits[at(c)]);
    let sum = 0;
    let labelSum = 0;
    let labelLogit = 0;
    for (let c = 0; c < columns; c++) {
      sum += Math.exp(logits[at(c)] - peak);
      labelSum += labels[at(c)];
      labelLogit += labels[at(c)] * logits[at(c)];
    }
    const logSumExp = peak + Math.log(sum);
    losses[row] = labelSum * logSumExp - labelLogit;
    if (dy)
      for (let c = 0; c < columns; c++) {
        gradientLogits[at(c)] =
          (Math.exp(logits[at(c)] - logSumExp) * labelSum - labels[at(c)]) *
          dy[row];
        gradientLabels[at(c)] = (logSumExp - logits[at(c)]) * dy[row];
      }
  }
  return { losses, gradientLogits, gradientLabels };
}
