"use typeshade";

// Maps: `out[i]` is a function of `x[i]` alone. Each kernel function is one loop that reads and
// writes only element `i`, so a map joins the call before it and the call after it in one
// dispatch (typeshade/fuse). A TypeShade builtin (`abs`, `exp`) cannot also name a kernel function
// that calls it, so a map is called `<operation>Map`. `copy`, `relu` and `reluGrad` are no builtins.

/** `out = x`: reshape, and the copy that makes a repeated operand a second array. */
export function copy(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = x[i];
  }
}

export function relu(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = max(x[i], 0.0);
  }
}

/** The gradient of `relu` at `x`: `dy` where `x` is positive, 0 elsewhere. */
export function reluGrad(out: array<f32>, x: array<f32>, dy: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = select(0.0, dy[i], x[i] > 0.0);
  }
}

export function negMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = -x[i];
  }
}

export function absMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = abs(x[i]);
  }
}

export function squareMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = x[i] * x[i];
  }
}

export function sqrtMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = sqrt(x[i]);
  }
}

export function expMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = exp(x[i]);
  }
}

export function logMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = log(x[i]);
  }
}

/** The logistic function, written so that `exp` never sees a large positive argument. */
export function sigmoidMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const v = x[i];
    let y: f32 = 0.0;
    if (v >= 0.0) {
      y = 1.0 / (1.0 + exp(-v));
    } else {
      const e = exp(v);
      y = e / (1.0 + e);
    }
    out[i] = y;
  }
}

export function tanhMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = tanh(x[i]);
  }
}

export function floorMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = floor(x[i]);
  }
}

export function ceilMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = ceil(x[i]);
  }
}

export function roundMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = round(x[i]);
  }
}

export function signMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = sign(x[i]);
  }
}

export function reciprocalMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = 1.0 / x[i];
  }
}

export function rsqrtMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = inverseSqrt(x[i]);
  }
}

export function sinMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = sin(x[i]);
  }
}

export function cosMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = cos(x[i]);
  }
}

export function tanMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = tan(x[i]);
  }
}

/** `exp(x) - 1`, with a series near 0 where the subtraction would lose digits. */
export function expm1Map(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const v = x[i];
    let y: f32 = 0.0;
    if (v > -0.01 && v < 0.01) {
      const square = v * v;
      y =
        v +
        square *
          (0.5 + v * (0.1666666667 + v * (0.0416666667 + v * 0.0083333333)));
    } else {
      y = exp(v) - 1.0;
    }
    out[i] = y;
  }
}

/** `log(1 + x)`, with a series near 0 where the addition would lose digits. */
export function log1pMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const v = x[i];
    let y: f32 = 0.0;
    if (v > -0.01 && v < 0.01) {
      const square = v * v;
      y = v - square * (0.5 - v * (0.3333333333 - v * (0.25 - v * 0.2)));
    } else {
      y = log(1.0 + v);
    }
    out[i] = y;
  }
}

export function sinhMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = sinh(x[i]);
  }
}

export function coshMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = cosh(x[i]);
  }
}

export function acosMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = acos(x[i]);
  }
}

export function acoshMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = acosh(x[i]);
  }
}

export function asinMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = asin(x[i]);
  }
}

export function asinhMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = asinh(x[i]);
  }
}

export function atanMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = atan(x[i]);
  }
}

export function atanhMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = atanh(x[i]);
  }
}

/** 1 where `x` is at least 0, 0 below it. */
export function stepMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = step(0.0, x[i]);
  }
}

// Maps with a scalar: the scalar follows the arrays.

export function clipMap(
  out: array<f32>,
  x: array<f32>,
  minimum: f32,
  maximum: f32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = min(max(x[i], minimum), maximum);
  }
}

export function leakyReluMap(out: array<f32>, x: array<f32>, alpha: f32): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const v = x[i];
    out[i] = select(alpha * v, v, v >= 0.0);
  }
}

export function eluMap(out: array<f32>, x: array<f32>, alpha: f32): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const v = x[i];
    let y: f32 = v;
    if (v < 0.0) {
      y = alpha * (exp(v) - 1.0);
    }
    out[i] = y;
  }
}

export function relu6Map(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = min(max(x[i], 0.0), 6.0);
  }
}

/** `log(1 + exp(x))`, written so that `exp` never sees a large positive argument. */
export function softplusMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const v = x[i];
    out[i] = max(v, 0.0) + log(1.0 + exp(-abs(v)));
  }
}

/** The tanh approximation of GELU. */
export function geluMap(out: array<f32>, x: array<f32>): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const v = x[i];
    const scaled = 0.7978845608 * (v + 0.044715 * v * v * v);
    let tanhValue: f32 = 0.0;
    if (scaled > 10.0) {
      tanhValue = 1.0;
    } else if (scaled < -10.0) {
      tanhValue = -1.0;
    } else {
      tanhValue = 2.0 / (1.0 + exp(-2.0 * scaled)) - 1.0;
    }
    out[i] = 0.5 * v * (1.0 + tanhValue);
  }
}
