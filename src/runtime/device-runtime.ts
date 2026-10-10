import { createRuntime, type Runtime } from "typeshade/runtime";
import { PACKS } from "./kernel-registry.js";
import type { DevicePreference } from "./device-types.js";

export interface RuntimeSelection {
  readonly runtime?: Runtime;
  readonly fallbackReason?: string;
}

export function isDevicePreference(value: string): value is DevicePreference {
  return ["auto", "webgpu", "webgl2", "cpu"].includes(value);
}

export async function selectRuntime(
  preference: DevicePreference,
): Promise<RuntimeSelection> {
  if (preference === "cpu") return {};
  const tier = preference === "auto" ? "webgpu" : preference;
  try {
    const runtime = await createRuntime({
      prefer: [tier],
      programs: [...new Set(Object.values(PACKS))],
    });
    return { runtime };
  } catch (error) {
    if (preference !== "auto") throw error;
    return {
      fallbackReason: error instanceof Error ? error.message : String(error),
    };
  }
}
