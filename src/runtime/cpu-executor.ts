import type { TensorNode } from "../tensor/node.js";
import { sizeOf } from "../tensor/shape.js";
import { Tensor } from "../tensor/tensor.js";
import { analyzeElementwiseFusion } from "./elementwise-fusion.js";
import {
  dispatchElementwiseFusionCpu,
  elementwiseFusionCpuProgram,
} from "./elementwise-fusion-ir.js";
import {
  executeBinary,
  executeBinaryGrad,
  executeInput,
  executeRelu,
  executeReluGrad,
  executeSpecial,
  executeUnary,
} from "./cpu-executors/elementwise.js";
import { executeMatmul } from "./cpu-executors/linear-algebra.js";
import { executeReshape } from "./cpu-executors/shape.js";
import {
  executeSoftmaxCrossEntropy,
  executeSoftmaxCrossEntropyGrad,
} from "./cpu-executors/loss.js";
import {
  executeSoftmax,
  executeSoftmaxGrad,
} from "./cpu-executors/normalization.js";
import {
  executeReduction,
  executeReductionGrad,
} from "./cpu-executors/reductions.js";
import {
  executeConv2d,
  executeConv2dGrad,
  executePool2d,
  executePool2dGrad,
} from "./cpu-executors/spatial.js";
import type {
  CpuNodeExecutor,
  CpuNodeExecutorMap,
  CpuNodeKind,
} from "./cpu-executors/types.js";
import type { ElementwiseFusionDecision } from "./elementwise-fusion.js";

const NODE_EXECUTORS: CpuNodeExecutorMap = {
  input: executeInput,
  relu: executeRelu,
  reluGrad: executeReluGrad,
  unary: executeUnary,
  special: executeSpecial,
  binary: executeBinary,
  binaryGrad: executeBinaryGrad,
  softmax: executeSoftmax,
  softmaxGrad: executeSoftmaxGrad,
  matmul: executeMatmul,
  conv2d: executeConv2d,
  pool2d: executePool2d,
  conv2dGrad: executeConv2dGrad,
  pool2dGrad: executePool2dGrad,
  reduceGrad: executeReductionGrad,
  softmaxCrossEntropyGrad: executeSoftmaxCrossEntropyGrad,
  reduce: executeReduction,
  softmaxCrossEntropy: executeSoftmaxCrossEntropy,
  reshape: executeReshape,
};

export interface CpuGraphExecution {
  readonly value: Float32Array;
  readonly decision: ElementwiseFusionDecision | undefined;
  /** Count of actual fused TypeShade CPU compute entry dispatches. */
  readonly dispatches: number;
}

interface CpuExecutionContext {
  readonly fusionTarget: Tensor | undefined;
  readonly readMaterialized: boolean;
  readonly writeMaterialized: boolean;
  readonly pending: Map<Tensor, Promise<Float32Array>>;
  dispatches: number;
}

const pendingGraphEvaluations = new WeakMap<
  Tensor,
  Promise<CpuGraphExecution>
>();

/** Execute a deferred Tensor graph, lowering eligible subgraphs to emitted TypeShade CPU code. */
export async function evaluateCpu(tensor: Tensor): Promise<Float32Array> {
  return (await evaluateCpuWithStats(tensor)).value;
}

/** Internal execution mode with counters from actual CPU compute dispatches. */
export async function evaluateCpuWithStats(
  tensor: Tensor,
): Promise<CpuGraphExecution> {
  const pending = pendingGraphEvaluations.get(tensor);
  if (pending) return pending;

  const task = executeCpuGraph(tensor, true).finally(() =>
    pendingGraphEvaluations.delete(tensor),
  );
  pendingGraphEvaluations.set(tensor, task);
  return task;
}

/** Explicit nodewise TypeNN CPU reference path for parity checks and diagnostics. */
export async function evaluateCpuUnfused(
  tensor: Tensor,
): Promise<Float32Array> {
  return (await evaluateCpuUnfusedWithStats(tensor)).value;
}

/** Internal nodewise reference with actual per-kernel dispatch accounting. */
export async function evaluateCpuUnfusedWithStats(
  tensor: Tensor,
): Promise<CpuGraphExecution> {
  return executeCpuGraph(tensor, false);
}

async function executeCpuGraph(
  tensor: Tensor,
  allowFusion: boolean,
): Promise<CpuGraphExecution> {
  const analysis = analyzeElementwiseFusion(tensor);
  const decision = allowFusion
    ? "decision" in analysis
      ? analysis.decision
      : analysis
    : undefined;
  const context: CpuExecutionContext = {
    fusionTarget: allowFusion ? tensor : undefined,
    readMaterialized: allowFusion,
    writeMaterialized: allowFusion,
    pending: new Map(),
    dispatches: 0,
  };
  const value = await evaluateNode(tensor, context);
  return { value, decision, dispatches: context.dispatches };
}

function evaluateNode(
  tensor: Tensor,
  context: CpuExecutionContext,
): Promise<Float32Array> {
  const existing = context.readMaterialized
    ? tensor.getMaterialized()
    : undefined;
  if (existing instanceof Float32Array) return Promise.resolve(existing);

  const pending = context.pending.get(tensor);
  if (pending) return pending;

  const task = (async () => {
    if (context.fusionTarget === tensor) {
      const analysis = analyzeElementwiseFusion(tensor);
      if ("decision" in analysis) {
        const program = elementwiseFusionCpuProgram(analysis);
        const length = sizeOf(tensor.shape);
        const leaves = analysis.leaves.map((leaf) => {
          if (leaf.node.kind !== "input")
            throw new Error("Fusion planner returned a non-input leaf.");
          return leaf.node.value;
        });
        const value = dispatchElementwiseFusionCpu(
          program.cpu,
          program.inputNames,
          leaves,
          length,
        );
        if (length > 0) context.dispatches++;
        return value;
      }
    }
    return executeNode(
      tensor,
      tensor.node,
      (input) => evaluateNode(input, context),
      () => context.dispatches++,
    );
  })()
    .then((value) => {
      if (context.writeMaterialized) tensor.setMaterialized(value);
      return value;
    })
    .finally(() => context.pending.delete(tensor));
  context.pending.set(tensor, task);
  return task;
}

function executeNode<K extends CpuNodeKind>(
  tensor: Tensor,
  node: Extract<TensorNode, { kind: K }>,
  evaluate: (input: Tensor) => Promise<Float32Array>,
  onDispatch: () => void,
): Promise<Float32Array> {
  const execute = NODE_EXECUTORS[node.kind] as CpuNodeExecutor<K>;
  return execute(tensor, node, evaluate, onDispatch);
}
