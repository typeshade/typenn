// Declarations for the vendored TypeNN browser runtime (index.js). The website keeps this file
// narrow: it lists the exports the Playground and the Lab import, with the signatures of the
// compiled declarations in the TypeNN build (dist/types) that they use.

export interface Tensor {
  readonly device: Device;
  readonly shape: readonly number[];
  readonly requiresGrad: boolean;
  /** The gradient that backward() accumulated on a parameter. Lazy until it is read. */
  grad?: Tensor;
  add(other: Tensor): Tensor;
  sub(other: Tensor): Tensor;
  mul(other: Tensor): Tensor;
  matmul(other: Tensor): Tensor;
  relu(): Tensor;
  softmax(axis?: number): Tensor;
  conv2d(
    filter: Tensor,
    options?: {
      readonly strides?: number;
      readonly padding?: "valid" | "same";
    },
  ): Tensor;
  maxPool(options: {
    readonly filterSize: number;
    readonly strides?: number;
  }): Tensor;
  reshape(shape: readonly number[]): Tensor;
  data(): Promise<Float32Array>;
  backward(upstream?: Tensor): Promise<void>;
  disposeGraph(preserve?: readonly Tensor[]): Promise<void>;
}

export interface FuseReport {
  readonly tier: string;
  readonly dispatches: number;
  readonly submissions: number;
  readonly groups: readonly {
    readonly calls: readonly string[];
    readonly fused: boolean;
    /** Why the group stands apart from the group before it, in the author's words. */
    readonly why?: string;
  }[];
}

export interface DevicePlan {
  readonly report: FuseReport;
  readonly nodes: readonly number[];
}

export interface Device {
  readonly backend: {
    readonly provider: "typeshade";
    readonly name: "cpu" | "webgpu" | "webgl2";
    readonly fallbackReason?: string;
  };
  /** The report of the last plan a `data()` ran. */
  readonly lastReport?: FuseReport;
  /** The last plan, with the graph id of each kernel call. */
  readonly lastPlan?: DevicePlan;
  tensor(
    values: ArrayLike<number>,
    options: { readonly shape: readonly number[] },
  ): Tensor;
  dispose(): Promise<void>;
}

export type DevicePreference = "auto" | "cpu" | "webgpu" | "webgl2";

export function device(
  options?:
    | {
        readonly device?: DevicePreference;
        /** The most bytes of released arrays kept for reuse. 0 turns the reuse off. Default 64 MiB. */
        readonly poolBytes?: number;
      }
    | DevicePreference,
): Promise<Device>;

export abstract class Module {
  abstract forward(input: Tensor): Tensor;
  to(device: Device): this;
  parameters(): Tensor[];
  /** Refresh the host arrays of the layers (`weight`, `bias`) from the device, in one plan. */
  sync(): Promise<void>;
  train(): this;
  eval(): this;
  isTraining(): boolean;
}

export class Sequential extends Module {
  readonly layers: readonly Module[];
  constructor(...layers: Module[]);
  forward(input: Tensor): Tensor;
}

export class Linear extends Module {
  readonly inFeatures: number;
  readonly outFeatures: number;
  constructor(
    inFeatures: number,
    outFeatures: number,
    options?: { readonly seed?: number; readonly weightScale?: number },
  );
  forward(input: Tensor): Tensor;
}

export class Dense extends Linear {}

export interface Conv2dOptions {
  readonly kernelSize: number | readonly [number, number];
  readonly strides?: number | readonly [number, number];
  readonly padding?: "same" | "valid";
  readonly bias?: boolean;
  readonly seed?: number;
}

export class Conv2d extends Module {
  readonly inChannels: number;
  readonly outChannels: number;
  readonly kernelSize: readonly [number, number];
  readonly strides: readonly [number, number];
  readonly padding: "same" | "valid";
  constructor(inChannels: number, outChannels: number, options: Conv2dOptions);
  forward(input: Tensor): Tensor;
}

export interface Pool2DOptions {
  readonly filterSize: number | readonly [number, number];
  readonly strides?: number | readonly [number, number];
  readonly padding?: "same" | "valid";
}

/** `strides` defaults to `filterSize`. */
export class MaxPool2d extends Module {
  constructor(options: Pool2DOptions);
  forward(input: Tensor): Tensor;
}

/** `strides` defaults to `filterSize`. */
export class AvgPool2d extends Module {
  constructor(options: Pool2DOptions);
  forward(input: Tensor): Tensor;
}

export class ReLU extends Module {
  forward(input: Tensor): Tensor;
}

export class Tanh extends Module {
  forward(input: Tensor): Tensor;
}

export class Sigmoid extends Module {
  forward(input: Tensor): Tensor;
}

export class Dropout extends Module {
  readonly rate: number;
  constructor(rate: number, options?: { readonly seed?: number });
  forward(input: Tensor): Tensor;
}

export class Flatten extends Module {
  forward(input: Tensor): Tensor;
}

export class CrossEntropyLoss {
  readonly reduction: "none" | "sum" | "mean";
  constructor(options?: { readonly reduction?: "none" | "sum" | "mean" });
  forward(logits: Tensor, labels: ArrayLike<number>): Tensor;
}

export class MSELoss {
  forward(prediction: Tensor, target: Tensor | Float32Array): Tensor;
}

export interface Optimizer {
  readonly parameters: readonly Tensor[];
  zeroGrad(): Promise<void>;
  /**
   * Update every parameter in one plan. The values of `alsoRead` are computed in that plan, from
   * the parameters before the update, and returned in the order given.
   */
  step(options?: {
    readonly alsoRead?: readonly Tensor[];
  }): Promise<Float32Array[]>;
  /** Free the arrays the optimizer holds on the device besides the parameters. */
  dispose(): Promise<void>;
}

export const optim: {
  SGD(
    parameters: readonly Tensor[],
    options: { readonly lr: number },
  ): Optimizer;
  Adam(
    parameters: readonly Tensor[],
    options: {
      readonly lr: number;
      readonly beta1?: number;
      readonly beta2?: number;
      readonly epsilon?: number;
    },
  ): Optimizer;
};

export interface GraphNode {
  readonly id: number;
  readonly kind: "input" | "op";
  readonly op: string;
  readonly shape: readonly number[];
  readonly inputs: readonly number[];
  readonly parameter: boolean;
  readonly computed: boolean;
}

export interface GraphSnapshot {
  readonly output: number;
  readonly nodes: readonly GraphNode[];
}

export function graph(tensor: Tensor): GraphSnapshot;
