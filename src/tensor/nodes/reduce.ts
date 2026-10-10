import type { Tensor } from "../tensor.js";

export type ReductionOperation =
  "sum" | "mean" | "prod" | "max" | "min" | "argMax" | "argMin";

/** @internal */
export interface ReduceNode {
  readonly kind: "reduce";
  readonly op: ReductionOperation;
  readonly input: Tensor;
  readonly outer: number;
  readonly axisSize: number;
  readonly inner: number;
}
