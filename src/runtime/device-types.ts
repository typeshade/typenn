import type { FuseReport } from "typeshade/fuse";

/** The last plan of a device: its report, and the graph id of each kernel call, in call order. */
export interface DevicePlan {
  readonly report: FuseReport;
  readonly nodes: readonly number[];
}

export type DevicePreference = "auto" | "webgpu" | "webgl2" | "cpu";
export type ExecutionTier = "webgpu" | "webgl2" | "cpu";

export interface DeviceOptions {
  /** `auto` tries WebGPU, then the TypeShade CPU tier. WebGL2 is explicit. */
  readonly device?: DevicePreference;
  /**
   * A `GPUDevice` the application already has. TypeNN runs its kernel functions on it and never
   * destroys it. Only `auto` and `webgpu` take it.
   */
  readonly gpuDevice?: object;
  /**
   * The most bytes of released arrays the device keeps to reuse for the next plan's outputs. `0`
   * turns the reuse off. The default is 64 MiB. A pooled array is freed when the device is.
   */
  readonly poolBytes?: number;
}

export interface BackendInfo {
  readonly provider: "typeshade";
  readonly name: ExecutionTier;
  readonly fallbackReason?: string;
}
