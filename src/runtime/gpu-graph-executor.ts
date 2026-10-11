import { resident, type Resident, type Runtime } from "typeshade/runtime";
import { sizeOf } from "../tensor/shape.js";
import type { Tensor } from "../tensor/tensor.js";
import type { Operation } from "./kernel-registry.js";
import { executorFor } from "./gpu-executors/index.js";
import { DeviceBuffers } from "./device-buffers.js";
import {
  analyzeElementwiseFusion,
  type ElementwiseFusionDecision,
} from "./elementwise-fusion.js";

export class GpuGraphExecutor {
  constructor(
    private readonly runtime: Runtime,
    private readonly pipelines: ReadonlyMap<Operation, unknown>,
    private readonly buffers: DeviceBuffers,
    private readonly onFusionDecision?: (
      decision: ElementwiseFusionDecision,
      dispatches: number,
    ) => void,
  ) {}

  async execute(tensor: Tensor): Promise<Float32Array> {
    const frame = this.runtime.frame();
    let dispatches = 0;
    const scheduled = new Map<Tensor, Promise<Resident<Float32Array>>>();
    const analysis = analyzeElementwiseFusion(tensor);
    const candidate = "decision" in analysis ? analysis : undefined;
    const decision =
      candidate?.decision ?? (analysis as ElementwiseFusionDecision);
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
    candidate: Extract<
      ReturnType<typeof analyzeElementwiseFusion>,
      { readonly decision: { readonly status: "fused" } }
    >,
    frame: ReturnType<Runtime["frame"]>,
    dispatched: () => void,
    scheduled: Map<Tensor, Promise<Resident<Float32Array>>>,
  ): Promise<Resident<Float32Array>> {
    const output = this.buffers.track(
      resident(new Float32Array(sizeOf(tensor.shape))),
    );
    try {
      const [addA, addB, multiplier] = await Promise.all([
        this.schedule(candidate.addA, frame, dispatched, scheduled),
        this.schedule(candidate.addB, frame, dispatched, scheduled),
        this.schedule(candidate.multiplier, frame, dispatched, scheduled),
      ]);
      const pipeline = this.pipelines.get("addMulRelu");
      if (!pipeline)
        throw new Error("TypeShade pipeline not prepared: addMulRelu.");
      frame.dispatch(
        pipeline as never,
        {
          shape: { length: sizeOf(tensor.shape) },
          addA,
          addB,
          multiplier,
          output,
        } as never,
        Math.ceil(sizeOf(tensor.shape) / 64),
      );
      dispatched();
      tensor.setMaterialized(output);
      return output;
    } catch (error) {
      this.buffers.release(output);
      throw error;
    }
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
