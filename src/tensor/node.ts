import type { Tensor } from "./tensor.js";

/**
 * A kernel function of `src/kernels`, called as `kernel(out, ...arrays, ...values)`: `out` and
 * the arrays are `Resident` arrays, and the values follow them.
 */
export type Kernel = (out: any, ...rest: any[]) => unknown;

/** A scalar, or a fixed-size array of scalars, that a kernel function takes by value. */
export type KernelValue = number | readonly number[];

/**
 * The gradient of each input of an operation, given the gradient of its output (`upstream`) and
 * its output (`value`). One entry for each input in order; `undefined` where an input has none.
 */
export type Backward = (
  upstream: Tensor,
  value: Tensor,
) => readonly (Tensor | undefined)[];

/** @internal A tensor the caller made: it holds its values on the host. */
export interface InputNode {
  readonly kind: "input";
  readonly value: Float32Array;
}

/**
 * @internal A tensor one kernel function call computes: one operation of the graph. The
 * executor calls `kernel(out, ...inputs, ...values)` for it, where `out` is the array of this
 * tensor and each input is the array of that tensor.
 */
export interface OpNode {
  readonly kind: "op";
  /** What an error and a report call the operation. */
  readonly op: string;
  readonly kernel: Kernel;
  readonly inputs: readonly Tensor[];
  readonly values: readonly KernelValue[];
  /** Absent where the operation has no gradient rule. */
  readonly backward?: Backward;
}

/** @internal */
export type TensorNode = InputNode | OpNode;
