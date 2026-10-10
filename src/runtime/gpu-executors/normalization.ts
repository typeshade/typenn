import type { GpuNodeExecutor } from "./types.js";

export const executeSoftmax: GpuNodeExecutor<"softmax"> = async ({
  node,
  output,
  schedule,
}) => ({
  operation: node.op,
  bindings: {
    shape: {
      outer: node.outer,
      axisSize: node.axisSize,
      inner: node.inner,
    },
    input: await schedule(node.input),
    output,
  },
  workgroups: Math.ceil((node.outer * node.inner) / 64),
});

export const executeSoftmaxGrad: GpuNodeExecutor<"softmaxGrad"> = async ({
  node,
  output,
  schedule,
}) => {
  const [value, dy] = await Promise.all([
    schedule(node.value),
    schedule(node.dy),
  ]);
  return {
    operation: node.op,
    bindings: {
      shape: {
        outer: node.outer,
        axisSize: node.axisSize,
        inner: node.inner,
        logMode: node.op === "logSoftmaxGrad" ? 1 : 0,
      },
      value,
      dy,
      output,
    },
    workgroups: Math.ceil((node.outer * node.inner) / 64),
  };
};
