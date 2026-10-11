import type { Tensor } from "../../tensor/tensor.js";
import { BinaryOperation } from "../../kernels/binary-grad.shade.ts";
import { absKernel } from "../../kernels/abs.shade.ts";
import { acosKernel } from "../../kernels/acos.shade.ts";
import { acoshKernel } from "../../kernels/acosh.shade.ts";
import { asinKernel } from "../../kernels/asin.shade.ts";
import { asinhKernel } from "../../kernels/asinh.shade.ts";
import { atanKernel } from "../../kernels/atan.shade.ts";
import { atanhKernel } from "../../kernels/atanh.shade.ts";
import { ceilKernel } from "../../kernels/ceil.shade.ts";
import { cosKernel } from "../../kernels/cos.shade.ts";
import { coshKernel } from "../../kernels/cosh.shade.ts";
import { expKernel } from "../../kernels/exp.shade.ts";
import { expm1Kernel } from "../../kernels/expm1.shade.ts";
import { floorKernel } from "../../kernels/floor.shade.ts";
import { logKernel } from "../../kernels/log.shade.ts";
import { log1pKernel } from "../../kernels/log1p.shade.ts";
import { negKernel } from "../../kernels/neg.shade.ts";
import { reciprocalKernel } from "../../kernels/reciprocal.shade.ts";
import { roundKernel } from "../../kernels/round.shade.ts";
import { rsqrtKernel } from "../../kernels/rsqrt.shade.ts";
import { sigmoidKernel } from "../../kernels/sigmoid.shade.ts";
import { signKernel } from "../../kernels/sign.shade.ts";
import { sinKernel } from "../../kernels/sin.shade.ts";
import { sinhKernel } from "../../kernels/sinh.shade.ts";
import { sqrtKernel } from "../../kernels/sqrt.shade.ts";
import { squareKernel } from "../../kernels/square.shade.ts";
import { stepKernel } from "../../kernels/step.shade.ts";
import { tanKernel } from "../../kernels/tan.shade.ts";
import { tanhKernel } from "../../kernels/tanh.shade.ts";
import { addKernel } from "../../kernels/add.shade.ts";
import { divKernel } from "../../kernels/div.shade.ts";
import { maximumKernel } from "../../kernels/maximum.shade.ts";
import { minimumKernel } from "../../kernels/minimum.shade.ts";
import { mulKernel } from "../../kernels/mul.shade.ts";
import { powKernel } from "../../kernels/pow.shade.ts";
import { preluKernel } from "../../kernels/prelu.shade.ts";
import { subKernel } from "../../kernels/sub.shade.ts";
import { clipByValueKernel } from "../../kernels/clip-by-value.shade.ts";
import { eluKernel } from "../../kernels/elu.shade.ts";
import { geluKernel } from "../../kernels/gelu.shade.ts";
import { leakyReluKernel } from "../../kernels/leaky-relu.shade.ts";
import { relu6Kernel } from "../../kernels/relu6.shade.ts";
import { softplusKernel } from "../../kernels/softplus.shade.ts";
import { reluKernel as relu } from "../../kernels/relu.shade.ts";
import { reluGradKernel as reluGrad } from "../../kernels/relu-grad.shade.ts";
import { binaryGradKernel as binaryGrad } from "../../kernels/binary-grad.shade.ts";
import { createOutput, dispatchCpuKernel } from "./common.js";
import type { CpuNodeExecutor } from "./types.js";

const UNARY_KERNELS = {
  acos: acosKernel,
  acosh: acoshKernel,
  asin: asinKernel,
  asinh: asinhKernel,
  atan: atanKernel,
  atanh: atanhKernel,
  step: stepKernel,
  neg: negKernel,
  abs: absKernel,
  square: squareKernel,
  sqrt: sqrtKernel,
  exp: expKernel,
  log: logKernel,
  sigmoid: sigmoidKernel,
  tanh: tanhKernel,
  floor: floorKernel,
  ceil: ceilKernel,
  round: roundKernel,
  sign: signKernel,
  reciprocal: reciprocalKernel,
  rsqrt: rsqrtKernel,
  sin: sinKernel,
  cos: cosKernel,
  tan: tanKernel,
  expm1: expm1Kernel,
  log1p: log1pKernel,
  sinh: sinhKernel,
  cosh: coshKernel,
};

const SPECIAL_KERNELS = {
  clipByValue: clipByValueKernel,
  gelu: geluKernel,
  leakyRelu: leakyReluKernel,
  elu: eluKernel,
  relu6: relu6Kernel,
  softplus: softplusKernel,
};

const BINARY_KERNELS = {
  add: addKernel,
  sub: subKernel,
  mul: mulKernel,
  div: divKernel,
  maximum: maximumKernel,
  minimum: minimumKernel,
  pow: powKernel,
  prelu: preluKernel,
};

export const executeInput: CpuNodeExecutor<"input"> = async (_tensor, node) =>
  node.value;

export const executeRelu: CpuNodeExecutor<"relu"> = async (
  tensor,
  node,
  evaluate,
) => {
  const input = await evaluate(node.input);
  const output = createOutput(tensor);
  await dispatchCpuKernel(
    relu,
    { shape: { length: output.length }, input, output },
    output.length,
  );
  return output;
};

export const executeReluGrad: CpuNodeExecutor<"reluGrad"> = async (
  tensor,
  node,
  evaluate,
) => {
  const [input, dy] = await Promise.all([
    evaluate(node.input),
    evaluate(node.dy),
  ]);
  const output = createOutput(tensor);
  await dispatchCpuKernel(
    reluGrad,
    { shape: { length: output.length }, input, dy, output },
    output.length,
  );
  return output;
};

export const executeUnary: CpuNodeExecutor<"unary"> = async (
  tensor,
  node,
  evaluate,
) => {
  const input = await evaluate(node.input);
  const output = createOutput(tensor);
  await dispatchCpuKernel(
    UNARY_KERNELS[node.op],
    { shape: { length: output.length }, input, output },
    output.length,
  );
  return output;
};

export const executeSpecial: CpuNodeExecutor<"special"> = async (
  tensor,
  node,
  evaluate,
) => {
  const input = await evaluate(node.input);
  const output = createOutput(tensor);
  await dispatchCpuKernel(
    SPECIAL_KERNELS[node.op],
    {
      shape: {
        length: output.length,
        alpha: node.alpha,
        minimum: node.minimum,
        maximum: node.maximum,
      },
      input,
      output,
    },
    output.length,
  );
  return output;
};

export const executeBinary: CpuNodeExecutor<"binary"> = async (
  tensor,
  node,
  evaluate,
) => {
  const [a, b] = await Promise.all([evaluate(node.a), evaluate(node.b)]);
  const output = createOutput(tensor);
  await dispatchCpuKernel(
    BINARY_KERNELS[node.op],
    {
      shape: {
        length: output.length,
        rank: tensor.shape.length,
        dimensions: node.dimensions,
        aStrides: node.aStrides,
        bStrides: node.bStrides,
      },
      a,
      b,
      output,
    },
    output.length,
  );
  return output;
};

export const executeBinaryGrad: CpuNodeExecutor<"binaryGrad"> = async (
  tensor,
  node,
  evaluate,
) => {
  const [a, b, dy] = await Promise.all([
    evaluate(node.a),
    evaluate(node.b),
    evaluate(node.dy),
  ]);
  const output = createOutput(tensor);
  const operation = {
    add: BinaryOperation.Add,
    sub: BinaryOperation.Subtract,
    mul: BinaryOperation.Multiply,
    div: BinaryOperation.Divide,
    maximum: BinaryOperation.Maximum,
    minimum: BinaryOperation.Minimum,
    pow: BinaryOperation.Power,
    prelu: BinaryOperation.PRelu,
  }[node.op];
  await dispatchCpuKernel(
    binaryGrad,
    {
      shape: {
        rank: Math.max(node.a.shape.length, node.b.shape.length),
        dimensions: node.dimensions,
        aStrides: node.aStrides,
        bStrides: node.bStrides,
        targetLength: output.length,
        repeatCount: dy.length / output.length,
        operation,
        wrtA: node.wrtA ? 1 : 0,
      },
      a,
      b,
      dy,
      output,
    },
    output.length,
  );
  return output;
};
