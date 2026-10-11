import { sameShape, sizeOf } from "../tensor/shape.js";
import type { TensorNode } from "../tensor/node.js";
import { Tensor } from "../tensor/tensor.js";
import {
  ELEMENTWISE_FUSION_OPS,
  type ElementwiseFusionOperation,
} from "./kernel-registry.js";

export type ElementwiseFusionRefusalCode =
  | "unsupported-output"
  | "unsupported-chain"
  | "materialized-output"
  | "requires-grad"
  | "unsupported-leaf"
  | "shape-mismatch"
  | "aliased-leaf"
  | "observable-intermediate"
  | "materialized-intermediate"
  | "storage-buffer-limit";

export type ElementwiseFusionDecision =
  | {
      readonly status: "fused";
      readonly operations: number;
      readonly dispatches: 1;
    }
  | {
      readonly status: "refused";
      readonly code: ElementwiseFusionRefusalCode;
      readonly reason: string;
    };

export type ElementwiseFusionValueRef =
  | { readonly kind: "leaf"; readonly slot: number }
  | { readonly kind: "value"; readonly id: number };

export interface ElementwiseFusionStep {
  readonly id: number;
  readonly operation: ElementwiseFusionOperation;
  readonly inputs: readonly ElementwiseFusionValueRef[];
}

export interface ElementwiseFusionCandidate {
  readonly decision: Extract<ElementwiseFusionDecision, { status: "fused" }>;
  readonly output: Tensor;
  /** Pure operation calls in postorder, derived from Tensor node references. */
  readonly steps: readonly ElementwiseFusionStep[];
  /** Operation names in the same postorder, exposed for diagnostics. */
  readonly operations: readonly string[];
  readonly outputValue: ElementwiseFusionValueRef;
  /** Distinct input leaves in first-visit graph order. */
  readonly leaves: readonly Tensor[];
  /** Structural cache key; contains no Tensor identities or element values. */
  readonly signature: string;
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
    const refs = liveConsumerRefs(parent);
    refs.push(new WeakRef(output));
    consumers.set(parent, refs);
  }
}

function liveConsumerRefs(tensor: Tensor): WeakRef<Tensor>[] {
  const refs = consumers.get(tensor) ?? [];
  const liveRefs = refs.filter((ref) => ref.deref() !== undefined);
  if (liveRefs.length !== refs.length) consumers.set(tensor, liveRefs);
  return liveRefs;
}

function liveConsumers(tensor: Tensor): Tensor[] {
  return liveConsumerRefs(tensor).flatMap((ref) => {
    const value = ref.deref();
    return value ? [value] : [];
  });
}

function refuse(
  code: ElementwiseFusionRefusalCode,
  reason: string,
): ElementwiseFusionDecision {
  return { status: "refused", code, reason };
}

function operationForNode(
  node: TensorNode,
): ElementwiseFusionOperation | undefined {
  const operation = "op" in node ? node.op : node.kind;
  const definition = ELEMENTWISE_FUSION_OPS[operation as string];
  if (!definition || definition.nodeKind !== node.kind) return undefined;
  return definition;
}

function inputTensors(
  node: TensorNode,
  operation: ElementwiseFusionOperation,
): Tensor[] | undefined {
  const record = node as unknown as Record<string, unknown>;
  const inputs: Tensor[] = [];
  for (const field of operation.inputFields) {
    const input = record[field];
    if (!(input instanceof Tensor)) return undefined;
    inputs.push(input);
  }
  return inputs;
}

function structuralSignature(
  steps: readonly ElementwiseFusionStep[],
  outputValue: ElementwiseFusionValueRef,
): string {
  const topology = steps.map((step) => ({
    operation: step.operation.name,
    inputs: step.inputs.map((input) =>
      input.kind === "leaf" ? ["leaf", input.slot] : ["value", input.id],
    ),
  }));
  const output =
    outputValue.kind === "leaf"
      ? ["leaf", outputValue.slot]
      : ["value", outputValue.id];
  return `f32:${JSON.stringify({ topology, output })}`;
}

/** Walk a pure same-shape Tensor DAG and lower registered elementwise operations. */
export function analyzeElementwiseFusion(
  output: Tensor,
  options: { readonly maxStorageBuffers?: number } = {},
): ElementwiseFusionCandidate | ElementwiseFusionDecision {
  output.assertUsable();
  if (output.getMaterialized() !== undefined)
    return refuse(
      "materialized-output",
      "The requested output is already materialized and will be reused without fusion.",
    );

  const leaves: Tensor[] = [];
  const leafSlots = new Map<Tensor, number>();
  const values = new Map<Tensor, ElementwiseFusionValueRef>();
  const candidateTensors = new Set<Tensor>();
  const visiting = new Set<Tensor>();
  const steps: ElementwiseFusionStep[] = [];
  let refusal: ElementwiseFusionDecision | undefined;

  const walk = (tensor: Tensor): ElementwiseFusionValueRef | undefined => {
    tensor.assertUsable();
    const previous = values.get(tensor);
    if (previous) return previous;
    if (visiting.has(tensor)) {
      refusal ??= refuse(
        "unsupported-chain",
        "The Tensor graph contains a cycle and cannot be lowered safely.",
      );
      return undefined;
    }
    if (tensor.requiresGrad) {
      refusal ??= refuse(
        "requires-grad",
        "Fusion is disabled when any candidate tensor requires gradients because backward activations are not retained.",
      );
      return undefined;
    }
    if (!sameShape(output.shape, tensor.shape)) {
      refusal ??= refuse(
        "shape-mismatch",
        "Fusion supports only exact same-shape operands without broadcasting.",
      );
      return undefined;
    }

    const node = tensor.node;
    if (node.kind === "input") {
      let slot = leafSlots.get(tensor);
      if (slot === undefined) {
        slot = leaves.length;
        leafSlots.set(tensor, slot);
        leaves.push(tensor);
      }
      const reference: ElementwiseFusionValueRef = { kind: "leaf", slot };
      values.set(tensor, reference);
      return reference;
    }

    const operation = operationForNode(node);
    if (!operation) {
      refusal ??= refuse(
        "unsupported-chain",
        `Fusion has no registered pure elementwise implementation for Tensor node ${node.kind}${"op" in node ? `/${node.op}` : ""}.`,
      );
      return undefined;
    }

    if (tensor.getMaterialized() !== undefined) {
      refusal ??= refuse(
        "materialized-intermediate",
        "A computed intermediate is already materialized; the unfused path preserves its observable value.",
      );
      return undefined;
    }

    const inputs = inputTensors(node, operation);
    if (!inputs) {
      refusal ??= refuse(
        "unsupported-chain",
        `Tensor node ${operation.name} does not provide the registered input fields.`,
      );
      return undefined;
    }

    visiting.add(tensor);
    candidateTensors.add(tensor);
    const references: ElementwiseFusionValueRef[] = [];
    for (const input of inputs) {
      if (!sameShape(output.shape, input.shape)) {
        refusal ??= refuse(
          "shape-mismatch",
          "Fusion supports only exact same-shape operands without broadcasting.",
        );
        visiting.delete(tensor);
        return undefined;
      }
      const reference = walk(input);
      if (!reference) {
        visiting.delete(tensor);
        return undefined;
      }
      references.push(reference);
    }
    visiting.delete(tensor);

    const step: ElementwiseFusionStep = {
      id: steps.length,
      operation,
      inputs: references,
    };
    steps.push(step);
    const reference: ElementwiseFusionValueRef = { kind: "value", id: step.id };
    values.set(tensor, reference);
    return reference;
  };

  let outputValue: ElementwiseFusionValueRef | undefined;
  try {
    sizeOf(output.shape);
    outputValue = walk(output);
  } catch {
    return refuse(
      "shape-mismatch",
      "The output shape cannot be represented safely.",
    );
  }
  if (refusal) return refusal;
  if (!outputValue || steps.length < 2 || leaves.length === 0)
    return refuse(
      "unsupported-output",
      "Fusion requires an output graph with at least two registered elementwise operations and one input leaf.",
    );

  for (const intermediate of candidateTensors) {
    if (intermediate === output) continue;
    if (
      liveConsumers(intermediate).some(
        (consumer) => !candidateTensors.has(consumer),
      )
    )
      return refuse(
        "observable-intermediate",
        "A computed intermediate has a consumer outside the candidate graph; the unfused path preserves its observable value.",
      );
  }

  const storageBufferCount = leaves.length + 1;
  if (
    options.maxStorageBuffers !== undefined &&
    storageBufferCount > options.maxStorageBuffers
  )
    return refuse(
      "storage-buffer-limit",
      `The fused entry needs ${storageBufferCount} storage buffers (${leaves.length} distinct input leaves plus one output), exceeding this WebGPU device limit of ${options.maxStorageBuffers}; the unfused path will be used.`,
    );

  return {
    decision: {
      status: "fused",
      operations: steps.length,
      dispatches: 1,
    },
    output,
    steps,
    operations: steps.map((step) => step.operation.name),
    outputValue,
    leaves,
    signature: structuralSignature(steps, outputValue),
  };
}

/** Return the public, data-free decision used by the TypeShade graph executor. */
export function inspectElementwiseFusion(
  output: Tensor,
): ElementwiseFusionDecision {
  const analysis = analyzeElementwiseFusion(output);
  return "decision" in analysis ? analysis.decision : analysis;
}
