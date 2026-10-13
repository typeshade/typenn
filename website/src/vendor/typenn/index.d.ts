export interface Tensor {
  readonly shape: readonly number[];
  matmul(other: Tensor): Tensor;
  add(other: Tensor): Tensor;
  relu(): Tensor;
  data(): Promise<Float32Array>;
  disposeGraph(preserve?: readonly Tensor[]): Promise<void>;
}

export interface Device {
  readonly backend: {
    readonly name: "cpu" | "webgpu" | "webgl2";
    readonly fallbackReason?: string;
  };
  /** The plan of the last data(): tier, dispatches, submissions and groups of calls. */
  readonly lastReport?: { readonly tier: string; readonly dispatches: number; readonly submissions: number };
  tensor(values: ArrayLike<number>, options: { readonly shape: readonly number[] }): Tensor;
  dispose(): Promise<void>;
}

export type DevicePreference = "auto" | "cpu" | "webgpu" | "webgl2";

export function device(options?: { readonly device?: DevicePreference } | DevicePreference): Promise<Device>;
