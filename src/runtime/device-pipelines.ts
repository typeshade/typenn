import { type Pack, type Runtime } from "typeshade/runtime";
import { ENTRYPOINT, PACKS, type Operation } from "./kernel-registry.js";

export async function preparePipelines(
  runtime: Runtime,
): Promise<Map<Operation, unknown>> {
  const pipelines = new Map<Operation, unknown>();
  for (const [operation, pack] of Object.entries(PACKS) as [
    Operation,
    Pack,
  ][]) {
    const program = runtime.load(pack);
    pipelines.set(operation, await program.compute(ENTRYPOINT[operation]));
  }
  return pipelines;
}
