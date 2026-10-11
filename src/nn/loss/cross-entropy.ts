import type { Tensor } from "../../tensor/tensor.js";

export interface CrossEntropyOptions {
  readonly reduction?: "none" | "sum" | "mean";
}

/** Sparse integer-label cross entropy over a rank-2 `[batch, classes]` logits tensor. */
export class CrossEntropyLoss {
  readonly reduction: "none" | "sum" | "mean";

  constructor(options: CrossEntropyOptions = {}) {
    this.reduction = options.reduction ?? "mean";
  }

  forward(logits: Tensor, labels: ArrayLike<number>): Tensor {
    logits.assertUsable();
    if (logits.shape.length !== 2)
      throw new RangeError("CrossEntropyLoss expects rank-2 logits.");
    const [batch, classes] = logits.shape;
    if (labels.length !== batch)
      throw new RangeError(
        "CrossEntropyLoss label count must match the batch.",
      );
    const oneHot = new Float32Array(batch! * classes!);
    for (let row = 0; row < batch!; row++) {
      const label = labels[row]!;
      if (!Number.isSafeInteger(label) || label < 0 || label >= classes!)
        throw new RangeError(`Class label ${label} is out of range.`);
      oneHot[row * classes! + label] = 1;
    }
    const labelTensor = logits.device.tensor(oneHot, { shape: logits.shape });
    return logits.softmaxCrossEntropyWithLogits(labelTensor, {
      reduction: this.reduction,
    });
  }
}

export function crossEntropyLoss(
  options: CrossEntropyOptions = {},
): CrossEntropyLoss {
  return new CrossEntropyLoss(options);
}
