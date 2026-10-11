import {
  resident,
  type Pack,
  type Resident,
  type Runtime,
} from "typeshade/runtime";
import { sizeOf } from "../tensor/shape.js";
import type { Tensor } from "../tensor/tensor.js";
import type { Operation } from "./kernel-registry.js";
import { executorFor } from "./gpu-executors/index.js";
import { DeviceBuffers } from "./device-buffers.js";
import {
  analyzeElementwiseFusion,
  type ElementwiseFusionCandidate,
  type ElementwiseFusionDecision,
} from "./elementwise-fusion.js";
import { elementwiseFusionGpuProgram } from "./elementwise-fusion-ir.js";

const WORKGROUP_SIZE = 64;

/** Internal observation/injection points for deterministic planner boundary tests. */
export interface GpuFusionExecutorHooks {
  readonly maxStorageBuffers?: number;
  readonly onAnalysis?: (decision: ElementwiseFusionDecision) => void;
  readonly onResource?: (resource: "output-allocation" | "fusion-pack") => void;
}

export class GpuGraphExecutor {
  private readonly fusionPipelines = new Map<string, Promise<unknown>>();

  constructor(
    private readonly runtime: Runtime,
    private readonly pipelines: ReadonlyMap<Operation, unknown>,
    private readonly buffers: DeviceBuffers,
    private readonly onFusionDecision?: (
      decision: ElementwiseFusionDecision,
      dispatches: number,
    ) => void,
    private readonly fusionHooks?: GpuFusionExecutorHooks,
  ) {}

  async execute(tensor: Tensor): Promise<Float32Array> {
    const frame = this.runtime.frame();
    let dispatches = 0;
    const scheduled = new Map<Tensor, Promise<Resident<Float32Array>>>();
    const deviceStorageLimit =
      this.runtime.tier === "webgpu"
        ? (
            this.runtime.device as {
              readonly limits?: {
                readonly maxStorageBuffersPerShaderStage?: number;
              };
            }
          ).limits?.maxStorageBuffersPerShaderStage
        : undefined;
    const analysis = analyzeElementwiseFusion(tensor, {
      maxStorageBuffers:
        this.fusionHooks?.maxStorageBuffers ?? deviceStorageLimit,
    });
    const candidate = "decision" in analysis ? analysis : undefined;
    const decision: ElementwiseFusionDecision = candidate
      ? candidate.decision
      : (analysis as ElementwiseFusionDecision);
    this.fusionHooks?.onAnalysis?.(decision);
    const dispatched = () => dispatches++;
    const result = candidate
      ? await this.scheduleFused(
          tensor,
          candidate,
          frame,
          dispatched,
          scheduled,
        )
      : await this.schedule(tensor, frame, dispatched, scheduled);
    if (dispatches > 0) await frame.submit();
    this.onFusionDecision?.(decision, dispatches);
    return result.read();
  }

  private async scheduleFused(
    tensor: Tensor,
    candidate: ElementwiseFusionCandidate,
    frame: ReturnType<Runtime["frame"]>,
    dispatched: () => void,
    scheduled: Map<Tensor, Promise<Resident<Float32Array>>>,
  ): Promise<Resident<Float32Array>> {
    const length = sizeOf(tensor.shape);
    this.fusionHooks?.onResource?.("output-allocation");
    const output = this.buffers.track(resident(new Float32Array(length)));
    if (length === 0) {
      tensor.setMaterialized(output);
      return output;
    }

    try {
      this.fusionHooks?.onResource?.("fusion-pack");
      const program = elementwiseFusionGpuProgram(candidate);
      const [inputs, pipeline] = await Promise.all([
        Promise.all(
          candidate.leaves.map((leaf) =>
            this.schedule(leaf, frame, dispatched, scheduled),
          ),
        ),
        this.fusionPipeline(program.signature, program.pack, program.entry),
      ]);
      const bindings: Record<string, unknown> = {
        shape: { length },
        output,
      };
      for (let slot = 0; slot < inputs.length; slot++)
        bindings[program.inputNames[slot]!] = inputs[slot]!;

      frame.dispatch(
        pipeline as never,
        bindings as never,
        Math.ceil(length / WORKGROUP_SIZE),
      );
      dispatched();
      tensor.setMaterialized(output);
      return output;
    } catch (error) {
      this.buffers.release(output);
      throw error;
    }
  }

  private fusionPipeline(
    signature: string,
    pack: Pack,
    entry: string,
  ): Promise<unknown> {
    const cached = this.fusionPipelines.get(signature);
    if (cached) return cached;
    const pending = this.runtime.load(pack).compute(entry);
    this.fusionPipelines.set(signature, pending);
    void pending.catch(() => {
      if (this.fusionPipelines.get(signature) === pending)
        this.fusionPipelines.delete(signature);
    });
    return pending;
  }

  private schedule(
    tensor: Tensor,
    frame: ReturnType<Runtime["frame"]>,
    dispatched: () => void,
    scheduled: Map<Tensor, Promise<Resident<Float32Array>>>,
  ): Promise<Resident<Float32Array>> {
    const pending = scheduled.get(tensor);
    if (pending) return pending;
    const result = this.scheduleNode(tensor, frame, dispatched, scheduled);
    scheduled.set(tensor, result);
    return result;
  }

  private async scheduleNode(
    tensor: Tensor,
    frame: ReturnType<Runtime["frame"]>,
    dispatched: () => void,
    scheduled: Map<Tensor, Promise<Resident<Float32Array>>>,
  ): Promise<Resident<Float32Array>> {
    const existing = tensor.getMaterialized();
    if (existing && !(existing instanceof Float32Array))
      return existing as Resident<Float32Array>;

    const node = tensor.node;
    if (node.kind === "input") {
      const value = this.buffers.track(resident(node.value));
      tensor.setMaterialized(value);
      return value;
    }

    const output = this.buffers.track(
      resident(new Float32Array(sizeOf(tensor.shape))),
    );
    try {
      const dispatch = await executorFor(node.kind)({
        tensor,
        node,
        output,
        schedule: (input) => this.schedule(input, frame, dispatched, scheduled),
      });
      const pipeline = this.pipelines.get(dispatch.operation);
      if (!pipeline)
        throw new Error(
          `TypeShade pipeline not prepared: ${dispatch.operation}.`,
        );
      frame.dispatch(
        pipeline as never,
        dispatch.bindings as never,
        dispatch.workgroups,
      );
      dispatched();
      tensor.setMaterialized(output);
      return output;
    } catch (error) {
      this.buffers.release(output);
      throw error;
    }
  }
}
