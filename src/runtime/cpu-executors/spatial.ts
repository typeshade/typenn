import { avgPoolGradKernel } from "../../kernels/avg-pool-grad.shade.ts";
import { avgPoolKernel } from "../../kernels/avg-pool.shade.ts";
import { conv2dFilterGradKernel } from "../../kernels/conv2d-filter-grad.shade.ts";
import { conv2dInputGradKernel } from "../../kernels/conv2d-input-grad.shade.ts";
import { conv2dKernel } from "../../kernels/conv2d.shade.ts";
import { maxPoolGradKernel } from "../../kernels/max-pool-grad.shade.ts";
import { maxPoolKernel } from "../../kernels/max-pool.shade.ts";
import { sizeOf } from "../../tensor/shape.js";
import { createOutput, dispatchCpuKernel } from "./common.js";
import type { CpuNodeExecutor } from "./types.js";

const POOL_KERNELS = {
  maxPool: maxPoolKernel,
  avgPool: avgPoolKernel,
};

const POOL_GRAD_KERNELS = {
  maxPoolGrad: maxPoolGradKernel,
  avgPoolGrad: avgPoolGradKernel,
};

const CONVOLUTION_GRAD_KERNELS = {
  conv2dInputGrad: conv2dInputGradKernel,
  conv2dFilterGrad: conv2dFilterGradKernel,
};

interface Conv2dGradientContext {
  readonly shape: {
    readonly batch: number;
    readonly height: number;
    readonly width: number;
    readonly inputChannels: number;
    readonly kernelHeight: number;
    readonly kernelWidth: number;
    readonly outputChannels: number;
    readonly outputHeight: number;
    readonly outputWidth: number;
    readonly strideY: number;
    readonly strideX: number;
    readonly padTop: number;
    readonly padLeft: number;
    readonly inputLength: number;
    readonly weightLength: number;
  };
  readonly input: Float32Array;
  readonly weights: Float32Array;
  readonly dy: Float32Array;
  readonly output: Float32Array;
}

const CONVOLUTION_GRADIENT_EXECUTORS = {
  conv2dInputGrad: (context: Conv2dGradientContext) =>
    dispatchCpuKernel(
      CONVOLUTION_GRAD_KERNELS.conv2dInputGrad,
      {
        shape: {
          height: context.shape.height,
          width: context.shape.width,
          inputChannels: context.shape.inputChannels,
          kernelHeight: context.shape.kernelHeight,
          kernelWidth: context.shape.kernelWidth,
          outputChannels: context.shape.outputChannels,
          outputHeight: context.shape.outputHeight,
          outputWidth: context.shape.outputWidth,
          strideY: context.shape.strideY,
          strideX: context.shape.strideX,
          padTop: context.shape.padTop,
          padLeft: context.shape.padLeft,
          inputLength: context.shape.inputLength,
        },
        weights: context.weights,
        dy: context.dy,
        output: context.output,
      },
      context.output.length,
    ),
  conv2dFilterGrad: (context: Conv2dGradientContext) =>
    dispatchCpuKernel(
      CONVOLUTION_GRAD_KERNELS.conv2dFilterGrad,
      {
        shape: {
          batch: context.shape.batch,
          height: context.shape.height,
          width: context.shape.width,
          inputChannels: context.shape.inputChannels,
          kernelWidth: context.shape.kernelWidth,
          outputChannels: context.shape.outputChannels,
          outputHeight: context.shape.outputHeight,
          outputWidth: context.shape.outputWidth,
          strideY: context.shape.strideY,
          strideX: context.shape.strideX,
          padTop: context.shape.padTop,
          padLeft: context.shape.padLeft,
          weightLength: context.shape.weightLength,
        },
        input: context.input,
        dy: context.dy,
        output: context.output,
      },
      context.output.length,
    ),
};

export const executePool2d: CpuNodeExecutor<"pool2d"> = async (
  tensor,
  node,
  evaluate,
) => {
  const input = await evaluate(node.input);
  const output = createOutput(tensor);
  const [, height, width, channels] = node.input.shape as readonly [
    number,
    number,
    number,
    number,
  ];
  await dispatchCpuKernel(
    POOL_KERNELS[node.op],
    {
      shape: {
        height,
        width,
        channels,
        filterHeight: node.filterHeight,
        filterWidth: node.filterWidth,
        outputHeight: node.outputHeight,
        outputWidth: node.outputWidth,
        strideY: node.strideY,
        strideX: node.strideX,
        padTop: node.padTop,
        padLeft: node.padLeft,
        outputLength: output.length,
      },
      input,
      output,
    },
    output.length,
  );
  return output;
};

export const executePool2dGrad: CpuNodeExecutor<"pool2dGrad"> = async (
  tensor,
  node,
  evaluate,
) => {
  const [input, dy] = await Promise.all([
    evaluate(node.input),
    evaluate(node.dy),
  ]);
  const output = createOutput(tensor);
  const [, height, width, channels] = node.input.shape as readonly [
    number,
    number,
    number,
    number,
  ];
  await dispatchCpuKernel(
    POOL_GRAD_KERNELS[node.op],
    {
      shape: {
        height,
        width,
        channels,
        filterHeight: node.filterHeight,
        filterWidth: node.filterWidth,
        outputHeight: node.outputHeight,
        outputWidth: node.outputWidth,
        strideY: node.strideY,
        strideX: node.strideX,
        padTop: node.padTop,
        padLeft: node.padLeft,
        inputLength: output.length,
      },
      input,
      dy,
      output,
    },
    output.length,
  );
  return output;
};

export const executeConv2d: CpuNodeExecutor<"conv2d"> = async (
  tensor,
  node,
  evaluate,
) => {
  const [input, weights] = await Promise.all([
    evaluate(node.input),
    evaluate(node.filter),
  ]);
  const output = createOutput(tensor);
  const [, height, width] = node.input.shape as readonly [
    number,
    number,
    number,
    number,
  ];
  await dispatchCpuKernel(
    conv2dKernel,
    {
      shape: {
        height,
        width,
        inputChannels: node.inputChannels,
        kernelHeight: node.kernelHeight,
        kernelWidth: node.kernelWidth,
        outputChannels: node.outputChannels,
        outputHeight: node.outputHeight,
        outputWidth: node.outputWidth,
        strideY: node.strideY,
        strideX: node.strideX,
        padTop: node.padTop,
        padLeft: node.padLeft,
        outputLength: output.length,
      },
      input,
      weights,
      output,
    },
    output.length,
  );
  return output;
};

export const executeConv2dGrad: CpuNodeExecutor<"conv2dGrad"> = async (
  tensor,
  node,
  evaluate,
) => {
  const [input, weights, dy] = await Promise.all([
    evaluate(node.input),
    evaluate(node.filter),
    evaluate(node.dy),
  ]);
  const output = createOutput(tensor);
  const [, height, width] = node.input.shape as readonly [
    number,
    number,
    number,
    number,
  ];
  const shape: Conv2dGradientContext["shape"] = {
    batch: node.input.shape[0]!,
    height,
    width,
    inputChannels: node.inputChannels,
    kernelHeight: node.kernelHeight,
    kernelWidth: node.kernelWidth,
    outputChannels: node.outputChannels,
    outputHeight: node.outputHeight,
    outputWidth: node.outputWidth,
    strideY: node.strideY,
    strideX: node.strideX,
    padTop: node.padTop,
    padLeft: node.padLeft,
    inputLength: sizeOf(node.input.shape),
    weightLength: weights.length,
  };
  const context = {
    shape,
    input,
    weights,
    dy,
    output,
  };
  await CONVOLUTION_GRADIENT_EXECUTORS[node.op](context);
  return output;
};
