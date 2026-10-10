import { BinaryOperation } from "../../kernels/binary-grad.shade.ts";
import { sizeOf } from "../../tensor/shape.js";
import type { GpuNodeExecutor } from "./types.js";
import { elementwiseWorkgroups, outputLength } from "./common.js";

export const executeRelu: GpuNodeExecutor<"relu"> = async ({
  tensor,
  node,
  output,
  schedule,
}) => ({
  operation: "relu",
  bindings: {
    shape: { length: outputLength({ tensor }) },
    input: await schedule(node.input),
    output,
  },
  workgroups: elementwiseWorkgroups({ tensor }),
});

export const executeReluGrad: GpuNodeExecutor<"reluGrad"> = async ({
  tensor,
  node,
  output,
  schedule,
}) => {
  const [input, dy] = await Promise.all([
    schedule(node.input),
    schedule(node.dy),
  ]);
  return {
    operation: "reluGrad",
    bindings: {
      shape: { length: outputLength({ tensor }) },
      input,
      dy,
      output,
    },
    workgroups: elementwiseWorkgroups({ tensor }),
  };
};

export const executeUnary: GpuNodeExecutor<"unary"> = async ({
  tensor,
  node,
  output,
  schedule,
}) => ({
  operation: node.op,
  bindings: {
    shape: { length: outputLength({ tensor }) },
    input: await schedule(node.input),
    output,
  },
  workgroups: elementwiseWorkgroups({ tensor }),
});

export const executeSpecial: GpuNodeExecutor<"special"> = async ({
  tensor,
  node,
  output,
  schedule,
}) => ({
  operation: node.op,
  bindings: {
    shape: {
      length: outputLength({ tensor }),
      alpha: node.alpha,
      minimum: node.minimum,
      maximum: node.maximum,
    },
    input: await schedule(node.input),
    output,
  },
  workgroups: elementwiseWorkgroups({ tensor }),
});

export const executeBinary: GpuNodeExecutor<"binary"> = async ({
  tensor,
  node,
  output,
  schedule,
}) => {
  const [a, b] = await Promise.all([schedule(node.a), schedule(node.b)]);
  return {
    operation: node.op,
    bindings: {
      shape: {
        length: outputLength({ tensor }),
        rank: tensor.shape.length,
        dimensions: node.dimensions,
        aStrides: node.aStrides,
        bStrides: node.bStrides,
      },
      a,
      b,
      output,
    },
    workgroups: elementwiseWorkgroups({ tensor }),
  };
};

export const executeBinaryGrad: GpuNodeExecutor<"binaryGrad"> = async ({
  tensor,
  node,
  output,
  schedule,
}) => {
  const [a, b, dy] = await Promise.all([
    schedule(node.a),
    schedule(node.b),
    schedule(node.dy),
  ]);
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
  return {
    operation: "binaryGrad",
    bindings: {
      shape: {
        length: sizeOf(node.dy.shape),
        rank: Math.max(node.a.shape.length, node.b.shape.length),
        dimensions: node.dimensions,
        aStrides: node.aStrides,
        bStrides: node.bStrides,
        targetLength: outputLength({ tensor }),
        repeatCount: sizeOf(node.dy.shape) / outputLength({ tensor }),
        operation,
        wrtA: node.wrtA ? 1 : 0,
      },
      a,
      b,
      dy,
      output,
    },
    workgroups: elementwiseWorkgroups({ tensor }),
  };
};
