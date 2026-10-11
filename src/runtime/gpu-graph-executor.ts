import { resident, type Resident, type Runtime } from "typeshade/runtime";
import { sizeOf } from "../tensor/shape.js";
import type { Tensor } from "../tensor/tensor.js";
import type { Operation } from "./kernel-registry.js";
import { executorFor } from "./gpu-executors/index.js";
import { DeviceBuffers } from "./device-buffers.js";

export class GpuGraphExecutor {
  constructor(
    private readonly runtime: Runtime,
    private readonly pipelines: ReadonlyMap<Operation, unknown>,
    private readonly buffers: DeviceBuffers,
  ) {}

  async execute(tensor: Tensor): Promise<Float32Array> {
    const frame = this.runtime.frame();
    let dispatches = 0;
    const scheduled = new Map<Tensor, Promise<Resident<Float32Array>>>();
    const result = await this.schedule(
      tensor,
      frame,
      () => dispatches++,
      scheduled,
    );
    if (dispatches > 0) await frame.submit();
    return result.read();
  }

  async runOptimizerUpdates(
    kind: "adam" | "sgd",
    updates: readonly {
      readonly parameter: Tensor;
      readonly gradient: Tensor;
      readonly firstMoment?: Resident<Float32Array>;
      readonly secondMoment?: Resident<Float32Array>;
    }[],
    options: {
      readonly learningRate: number;
      readonly beta1?: number;
      readonly beta2?: number;
      readonly epsilon?: number;
      readonly firstCorrection?: number;
      readonly secondCorrection?: number;
    },
  ): Promise<void> {
    const frame = this.runtime.frame();
    const scheduled = new Map<Tensor, Promise<Resident<Float32Array>>>();
    for (const update of updates) {
      const parameter = await this.schedule(
        update.parameter,
        frame,
        () => undefined,
        scheduled,
      );
      const gradient = await this.schedule(
        update.gradient,
        frame,
        () => undefined,
        scheduled,
      );
      const operation = kind === "adam" ? "adamUpdate" : "sgdUpdate";
      const pipeline = this.pipelines.get(operation);
      if (!pipeline)
        throw new Error(`TypeShade pipeline not prepared: ${operation}.`);
      const count = update.parameter.shape.reduce((size, dim) => size * dim, 1);
      let bindings: Record<string, unknown>;
      if (kind === "adam") {
        const firstMoment = update.firstMoment;
        const secondMoment = update.secondMoment;
        const beta1 = options.beta1;
        const beta2 = options.beta2;
        const epsilon = options.epsilon;
        const firstCorrection = options.firstCorrection;
        const secondCorrection = options.secondCorrection;
        if (
          !firstMoment ||
          !secondMoment ||
          beta1 === undefined ||
          beta2 === undefined ||
          epsilon === undefined ||
          firstCorrection === undefined ||
          secondCorrection === undefined
        )
          throw new Error(
            "Adam GPU update is missing moment state or hyperparameters.",
          );
        bindings = {
          shape: {
            count,
            learningRate: options.learningRate,
            beta1,
            beta2,
            epsilon,
            firstCorrection,
            secondCorrection,
          },
          parameter,
          gradient,
          firstMoment,
          secondMoment,
        };
      } else {
        bindings = {
          shape: { count, learningRate: options.learningRate },
          parameter,
          gradient,
        };
      }
      frame.dispatch(
        pipeline as never,
        bindings as never,
        Math.ceil(count / 64),
      );
    }
    await frame.submit();
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
