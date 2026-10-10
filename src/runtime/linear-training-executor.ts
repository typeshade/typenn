import type { Runtime } from "typeshade/runtime";
import { runCpuLinearTraining } from "./cpu-training.js";
import type { LinearTrainingCommand, PipelineMap } from "./device-types.js";

export async function runLinearTraining(
  runtime: Runtime | undefined,
  pipelines: PipelineMap,
  commands: readonly LinearTrainingCommand[],
): Promise<void> {
  if (!runtime) return runCpuLinearTraining(commands);

  const frame = runtime.frame();
  for (const command of commands) {
    const pipeline = pipelines.get(command.entry);
    if (!pipeline)
      throw new Error(`TypeShade pipeline not prepared: ${command.entry}.`);
    frame.dispatch(
      pipeline as never,
      command.bindings as never,
      command.workgroups,
    );
  }
  await frame.submit();
}
