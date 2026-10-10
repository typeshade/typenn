import type { Tensor } from "../tensor.js";

/** @internal */
export interface Conv2dGradNode {
  readonly kind: "conv2dGrad";
  readonly op: "conv2dInputGrad" | "conv2dFilterGrad";
  readonly input: Tensor;
  readonly filter: Tensor;
  readonly dy: Tensor;
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
}
