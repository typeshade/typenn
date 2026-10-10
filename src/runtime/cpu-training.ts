import type { LinearTrainingEntry } from "./kernel-registry.js";
import { linearForwardKernel } from "../kernels/linear-forward.shade.ts";
import { linearObjectiveKernel } from "../kernels/linear-objective.shade.ts";
import { linearReduceKernel } from "../kernels/linear-reduce.shade.ts";
import { linearBackwardKernel } from "../kernels/linear-backward.shade.ts";
import { linearUpdateKernel } from "../kernels/linear-update.shade.ts";

interface TrainingCommand {
  readonly entry: LinearTrainingEntry;
  readonly bindings: Record<string, unknown>;
  readonly workgroups: number;
}

const CPU_TRAINING_ENTRIES = {
  linearForward: linearForwardKernel,
  linearObjective: linearObjectiveKernel,
  linearReduce: linearReduceKernel,
  linearBackward: linearBackwardKernel,
  linearUpdate: linearUpdateKernel,
};

/** Run the Linear Trainer's ordered operations using TypeShade's generated CPU functions. */
export async function runCpuLinearTraining(
  commands: readonly TrainingCommand[],
): Promise<void> {
  for (const command of commands) {
    const entry = CPU_TRAINING_ENTRIES[command.entry] as unknown as (
      bindings: Record<string, unknown>,
      workgroups: number,
    ) => Promise<void>;
    await entry(command.bindings, command.workgroups);
  }
}
