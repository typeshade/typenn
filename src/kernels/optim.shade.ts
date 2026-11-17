"use typeshade";

// Optimizer updates. Each kernel function is one loop that writes `out[i]` from element `i` of its
// arrays, so the calls of one optimizer step join one dispatch where the compiler allows it
// (typeshade/fuse). `adamUpdate` reads the new first and second moments, which `adamFirst` and
// `adamSecond` wrote earlier in the same plan. The arithmetic follows the order of the host
// update this replaces, so the results match it to f32 rounding.

/** `out = param - lr * grad`: stochastic gradient descent. */
export function sgdUpdate(
  out: array<f32>,
  param: array<f32>,
  grad: array<f32>,
  lr: f32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = param[i] - lr * grad[i];
  }
}

// `keep` is `1 - beta`, computed in double on the host. In f32, `1 - beta2` for beta2 = 0.999
// loses about 1e-5 relative, so the complement is a value of its own and not a subtraction here.

/** The new first moment: `beta1 * m + keep * grad`, where `keep` is `1 - beta1`. */
export function adamFirst(
  out: array<f32>,
  m: array<f32>,
  grad: array<f32>,
  beta1: f32,
  keep: f32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = beta1 * m[i] + keep * grad[i];
  }
}

/** The new second moment: `beta2 * v + keep * grad * grad`, where `keep` is `1 - beta2`. */
export function adamSecond(
  out: array<f32>,
  v: array<f32>,
  grad: array<f32>,
  beta2: f32,
  keep: f32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = beta2 * v[i] + keep * grad[i] * grad[i];
  }
}

/**
 * `out = param - lr * (m / c1) / (sqrt(v / c2) + eps)`, where `c1` and `c2` are the bias
 * corrections of the step and `m` and `v` are the new moments.
 */
export function adamUpdate(
  out: array<f32>,
  param: array<f32>,
  m: array<f32>,
  v: array<f32>,
  lr: f32,
  c1: f32,
  c2: f32,
  eps: f32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    out[i] = param[i] - (lr * (m[i] / c1)) / (sqrt(v[i] / c2) + eps);
  }
}
