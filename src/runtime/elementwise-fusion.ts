import { sameShape, sizeOf } from "../tensor/shape.js";
import { Tensor } from "../tensor/tensor.js";

export type ElementwiseFusionRefusalCode =
  | "unsupported-output"
  | "unsupported-chain"
  | "materialized-output"
  | "requires-grad"
  | "unsupported-leaf"
  | "shape-mismatch"
  | "aliased-leaf"
  | "observable-intermediate"
  | "materialized-intermediate";

export type ElementwiseFusionDecision =
  | {
      readonly status: "fused";
      readonly operations: 3;
      readonly dispatches: 1;
    }
  | {
      readonly status: "refused";
      readonly code: ElementwiseFusionRefusalCode;
      readonly reason: string;
    };

export type ElementwiseFusionExpression =
  | { readonly kind: "input"; readonly slot: number }
  | {
      readonly kind: "add" | "mul";
      readonly left: ElementwiseFusionExpression;
      readonly right: ElementwiseFusionExpression;
    }
  | { readonly kind: "relu"; readonly input: ElementwiseFusionExpression };

export interface ElementwiseFusionCandidate {
  readonly decision: Extract<ElementwiseFusionDecision, { status: "fused" }>;
  readonly output: Tensor;
  /** Lowered expression derived by walking the original Tensor nodes. */
  readonly expression: ElementwiseFusionExpression;
  /** Distinct input leaves in first-visit graph order. */
  readonly leaves: readonly Tensor[];
  /** Operation order as observed by the graph walk. */
  readonly operations: readonly ("add" | "mul" | "relu")[];
}

const consumers = new WeakMap<Tensor, WeakRef<Tensor>[]>();

/** Record forward edges without keeping otherwise-dead graph branches alive. */
export function registerTensorConsumers(output: Tensor, node: object): void {
  const parents = new Set(
    Object.values(node).filter(
      (value): value is Tensor => value instanceof Tensor,
    ),
  );
  for (const parent of parents) {
    const refs = liveConsumers(parent);
    refs.push(new WeakRef(output));
    consumers.set(parent, refs);
  }
}

function liveConsumers(tensor: Tensor): WeakRef<Tensor>[] {
  const refs = consumers.get(tensor) ?? [];
  const live = refs.filter((ref) => ref.deref() !== undefined);
  if (live.length !== refs.length) consumers.set(tensor, live);
  return live;
}

function refuse(
  code: ElementwiseFusionRefusalCode,
  reason: string,
): ElementwiseFusionDecision {
  return { status: "refused", code, reason };
}

function walkExpression(
  tensor: Tensor,
  leaves: Tensor[],
  operations: ("add" | "mul" | "relu")[],
): ElementwiseFusionExpression | undefined {
  const node = tensor.node;
  if (node.kind === "input") {
    const slot = leaves.indexOf(tensor);
    if (slot >= 0) return { kind: "input", slot };
    leaves.push(tensor);
    return { kind: "input", slot: leaves.length - 1 };
  }
  if (node.kind === "relu") {
    const input = walkExpression(node.input, leaves, operations);
    if (!input) return undefined;
    operations.push("relu");
    return { kind: "relu", input };
  }
  if (node.kind === "binary" && (node.op === "add" || node.op === "mul")) {
    const left = walkExpression(node.a, leaves, operations);
    const right = walkExpression(node.b, leaves, operations);
    if (!left || !right) return undefined;
    operations.push(node.op);
    return { kind: node.op, left, right };
  }
  return undefined;
}

/** Analyze the narrow same-shape add -> mul -> relu pattern for safe fusion. */
export function analyzeElementwiseFusion(
  output: Tensor,
): ElementwiseFusionCandidate | ElementwiseFusionDecision {
  output.assertUsable();
  if (output.getMaterialized() !== undefined)
    return refuse(
      "materialized-output",
      "The requested output is already materialized and will be reused without fusion.",
    );

  if (output.node.kind !== "relu")
    return refuse(
      "unsupported-output",
      "Fusion currently supports only a ReLU output.",
    );

  const multiply = output.node.input;
  if (multiply.node.kind !== "binary" || multiply.node.op !== "mul")
    return refuse(
      "unsupported-chain",
      "Fusion requires ReLU(Multiply(Add(a, b), c)).",
    );

  const addOperands = [multiply.node.a, multiply.node.b].filter(
    (operand) => operand.node.kind === "binary" && operand.node.op === "add",
  );
  if (addOperands.length !== 1)
    return refuse(
      "unsupported-chain",
      "Fusion requires exactly one add intermediate consumed by multiply.",
    );

  const add = addOperands[0]!;
  const multiplier =
    add === multiply.node.a ? multiply.node.b : multiply.node.a;
  if (add.node.kind !== "binary")
    return refuse("unsupported-chain", "The add intermediate is malformed.");
  const { a: addA, b: addB } = add.node;
  const candidateNodes = [output, multiply, add, addA, addB, multiplier];
  if (candidateNodes.some((tensor) => tensor.requiresGrad))
    return refuse(
      "requires-grad",
      "Fusion is disabled when any candidate tensor requires gradients because backward activations are not retained.",
    );

  if (
    addA.node.kind !== "input" ||
    addB.node.kind !== "input" ||
    multiplier.node.kind !== "input"
  )
    return refuse(
      "unsupported-leaf",
      "Fusion currently requires three input leaves; nested or computed operands use the unfused path.",
    );

  if (new Set([addA, addB, multiplier]).size !== 3)
    return refuse(
      "aliased-leaf",
      "Fusion requires three distinct input tensors; aliased operands use the unfused path.",
    );

  if (
    !sameShape(addA.shape, addB.shape) ||
    !sameShape(addA.shape, multiplier.shape) ||
    !sameShape(addA.shape, add.shape) ||
    !sameShape(addA.shape, multiply.shape) ||
    !sameShape(addA.shape, output.shape)
  )
    return refuse(
      "shape-mismatch",
      "Fusion supports only exact same-shape operands without broadcasting.",
    );

  if (
    add.getMaterialized() !== undefined ||
    multiply.getMaterialized() !== undefined
  )
    return refuse(
      "materialized-intermediate",
      "An add or multiply intermediate is already materialized; the unfused path preserves its observable value.",
    );

  if (liveConsumers(add).length !== 1 || liveConsumers(multiply).length !== 1)
    return refuse(
      "observable-intermediate",
      "The add and multiply intermediates must each have exactly one live consumer.",
    );

  try {
    sizeOf(output.shape);
  } catch {
    return refuse(
      "shape-mismatch",
      "The output shape cannot be represented safely.",
    );
  }

  const leaves: Tensor[] = [];
  const operations: ("add" | "mul" | "relu")[] = [];
  const expression = walkExpression(output, leaves, operations);
  if (!expression || operations.length !== 3 || leaves.length !== 3)
    return refuse(
      "unsupported-chain",
      "The candidate graph could not be lowered as one add, one multiply, and one ReLU.",
    );

  return {
    decision: { status: "fused", operations: operations.length, dispatches: 1 },
    output,
    expression,
    leaves,
    operations,
  };
}

/** Return the public, data-free decision used by the TypeShade graph executor. */
export function inspectElementwiseFusion(
  output: Tensor,
): ElementwiseFusionDecision {
  const analysis = analyzeElementwiseFusion(output);
  return "decision" in analysis ? analysis.decision : analysis;
}
