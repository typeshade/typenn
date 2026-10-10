import type { LinearTrainingEntry, Operation } from "./kernel-registry.js";

export type DevicePreference = "auto" | "webgpu" | "webgl2" | "cpu";
export type ExecutionTier = "webgpu" | "webgl2" | "cpu";

export interface DeviceOptions {
  /** `auto` tries WebGPU, then the TypeShade CPU tier. WebGL2 is explicit. */
  readonly device?: DevicePreference;
}

export interface BackendInfo {
  readonly provider: "typeshade";
  readonly name: ExecutionTier;
  readonly fallbackReason?: string;
}

export interface LinearTrainingCommand {
  readonly entry: LinearTrainingEntry;
  readonly bindings: Record<string, unknown>;
  readonly workgroups: number;
}

export type PipelineMap = ReadonlyMap<Operation, unknown>;
