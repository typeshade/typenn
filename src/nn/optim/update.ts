import type { Tensor } from "../../tensor/tensor.js";
import { operation } from "../../tensor/operations/common.js";
import {
  adamFirst,
  adamSecond,
  adamUpdate,
  sgdUpdate,
} from "../../kernels/optim.shade.ts";

// The update operations of the optimizers. They are graph nodes without a gradient rule: an
// optimizer computes them in one plan and moves the results into the parameters and moments.

export function sgdUpdateOf(
  parameter: Tensor,
  gradient: Tensor,
  lr: number,
): Tensor {
  return operation(
    parameter.shape,
    "sgdUpdate",
    sgdUpdate,
    [parameter, gradient],
    [lr],
  );
}

export function adamFirstOf(
  first: Tensor,
  gradient: Tensor,
  beta1: number,
): Tensor {
  return operation(
    first.shape,
    "adamFirst",
    adamFirst,
    [first, gradient],
    [beta1, 1 - beta1],
  );
}

export function adamSecondOf(
  second: Tensor,
  gradient: Tensor,
  beta2: number,
): Tensor {
  return operation(
    second.shape,
    "adamSecond",
    adamSecond,
    [second, gradient],
    [beta2, 1 - beta2],
  );
}

export function adamUpdateOf(
  parameter: Tensor,
  first: Tensor,
  second: Tensor,
  values: readonly [number, number, number, number],
): Tensor {
  return operation(
    parameter.shape,
    "adamUpdate",
    adamUpdate,
    [parameter, first, second],
    values,
  );
}
