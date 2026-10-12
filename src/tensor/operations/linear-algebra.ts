import { matmul as matmulKernel } from "../../kernels/matmul.shade.ts";
import type { Tensor } from "../tensor.js";
import { assertCompatible, operation } from "./common.js";

export interface MatmulOptions {
  readonly transposeA?: boolean;
  readonly transposeB?: boolean;
}

export function matmul(
  a: Tensor,
  b: Tensor,
  options: MatmulOptions = {},
): Tensor {
  assertCompatible(a, b);
  if (a.shape.length !== 2 || b.shape.length !== 2)
    throw new RangeError("matmul currently supports rank-2 tensors only.");

  const transposeA = options.transposeA ?? false;
  const transposeB = options.transposeB ?? false;
  const rows = transposeA ? a.shape[1]! : a.shape[0]!;
  const inner = transposeA ? a.shape[0]! : a.shape[1]!;
  const otherInner = transposeB ? b.shape[1]! : b.shape[0]!;
  const columns = transposeB ? b.shape[0]! : b.shape[1]!;
  if (inner !== otherInner)
    throw new RangeError(
      `matmul inner dimensions must match; got ${inner} and ${otherInner}.`,
    );

  return operation(
    [rows, columns],
    "matmul",
    matmulKernel,
    [a, b],
    [rows, inner, columns, transposeA ? 1 : 0, transposeB ? 1 : 0],
    (upstream) => [
      transposeA
        ? b.matmul(upstream, { transposeA: transposeB, transposeB: true })
        : upstream.matmul(b, { transposeB: !transposeB }),
      transposeB
        ? upstream.matmul(a, { transposeA: true, transposeB: transposeA })
        : a.matmul(upstream, { transposeA: !transposeA }),
    ],
  );
}
