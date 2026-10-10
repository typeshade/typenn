import { sizeOf } from "../../tensor/shape.js";
import type { GpuNodeExecutor } from "./types.js";

export const executeConv2d: GpuNodeExecutor<"conv2d"> = async ({
  tensor,
  node,
  output,
  schedule,
}) => {
  const [input, filter] = await Promise.all([
    schedule(node.input),
    schedule(node.filter),
  ]);
  const [batch, height, width] = node.input.shape as readonly [
    number,
    number,
    number,
    number,
  ];
  return {
    operation: "conv2d",
    bindings: {
      shape: {
        batch,
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
        outputLength: sizeOf(tensor.shape),
      },
      input,
      weights: filter,
      output,
    },
    workgroups: Math.ceil(sizeOf(tensor.shape) / 64),
  };
};

export const executePool2d: GpuNodeExecutor<"pool2d"> = async ({
  tensor,
  node,
  output,
  schedule,
}) => {
  const input = await schedule(node.input);
  const [batch, height, width, channels] = node.input.shape as readonly [
    number,
    number,
    number,
    number,
  ];
  return {
    operation: node.op,
    bindings: {
      shape: {
        batch,
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
        outputLength: sizeOf(tensor.shape),
      },
      input,
      output,
    },
    workgroups: Math.ceil(sizeOf(tensor.shape) / 64),
  };
};

export const executeConv2dGrad: GpuNodeExecutor<"conv2dGrad"> = async ({
  tensor,
  node,
  output,
  schedule,
}) => {
  const [filter, dy] = await Promise.all([
    schedule(node.filter),
    schedule(node.dy),
  ]);
  const [batch, height, width] = node.input.shape as readonly [
    number,
    number,
    number,
    number,
  ];
  const shape = {
    batch,
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
    channels: 0,
    filterHeight: 0,
    filterWidth: 0,
    inputLength: sizeOf(node.input.shape),
    outputLength: sizeOf(node.dy.shape),
    weightLength: sizeOf(node.filter.shape),
  };
  if (node.op === "conv2dInputGrad") {
    return {
      operation: node.op,
      bindings: { shape, weights: filter, dy, output },
      workgroups: Math.ceil(sizeOf(tensor.shape) / 64),
    };
  }
  return {
    operation: node.op,
    bindings: {
      shape,
      input: await schedule(node.input),
      dy,
      output,
    },
    workgroups: Math.ceil(sizeOf(tensor.shape) / 64),
  };
};

export const executePool2dGrad: GpuNodeExecutor<"pool2dGrad"> = async ({
  tensor,
  node,
  output,
  schedule,
}) => {
  const [input, dy] = await Promise.all([
    schedule(node.input),
    schedule(node.dy),
  ]);
  const [batch, height, width, channels] = node.input.shape as readonly [
    number,
    number,
    number,
    number,
  ];
  return {
    operation: node.op,
    bindings: {
      shape: {
        batch,
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
        inputChannels: 0,
        kernelHeight: 0,
        kernelWidth: 0,
        outputChannels: 0,
        inputLength: sizeOf(node.input.shape),
        outputLength: sizeOf(node.dy.shape),
        weightLength: 0,
      },
      input,
      dy,
      output,
    },
    workgroups: Math.ceil(sizeOf(tensor.shape) / 64),
  };
};
