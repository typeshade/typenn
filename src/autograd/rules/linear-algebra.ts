import { addGradient } from "../accumulate.js";
import type { GradientRule } from "./types.js";

export const matmulGradient: GradientRule<"matmul"> = ({
  node,
  upstream,
  gradients,
}) => {
  const { a, b, transposeA, transposeB } = node;
  const aGradient = transposeA
    ? b.matmul(upstream, { transposeA: transposeB, transposeB: true })
    : upstream.matmul(b, { transposeB: !transposeB });
  const bGradient = transposeB
    ? upstream.matmul(a, { transposeA: true, transposeB: transposeA })
    : a.matmul(upstream, { transposeA: !transposeA });

  addGradient(gradients, a, aGradient);
  addGradient(gradients, b, bGradient);
};
