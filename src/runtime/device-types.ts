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
}

export interface BackendInfo {
  readonly provider: "typeshade";
  readonly name: ExecutionTier;
  readonly fallbackReason?: string;
}
