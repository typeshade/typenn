import { configure, type Resident, type Runtime } from "typeshade/runtime";
import { checkedShape, sizeOf, type Shape } from "../tensor/shape.js";
import type { TensorNode } from "../tensor/node.js";
import type { Operation } from "./kernel-registry.js";
import { Tensor } from "../tensor/tensor.js";
import { evaluateCpu } from "./cpu-executor.js";
import { DeviceBuffers } from "./device-buffers.js";
import { GpuGraphExecutor } from "./gpu-graph-executor.js";
import {
  registerTensorConsumers,
  type ElementwiseFusionDecision,
} from "./elementwise-fusion.js";
import { preparePipelines } from "./device-pipelines.js";
import { isDevicePreference, selectRuntime } from "./device-runtime.js";
import { runLinearTraining } from "./linear-training-executor.js";
import type {
  BackendInfo,
  DeviceOptions,
  DevicePreference,
  ExecutionTier,
  LinearTrainingCommand,
} from "./device-types.js";

export type { LinearTrainingEntry } from "./kernel-registry.js";
export type {
  BackendInfo,
  DeviceOptions,
  DevicePreference,
  ExecutionTier,
} from "./device-types.js";

// TypeShade's direct call-layer configuration is process-global. A single active device keeps
// selection and resident ownership predictable; the GPU path itself uses an owned Runtime/Frame.
let activeDevice: Device | undefined;

export class Device {
  readonly backend: BackendInfo;
  private readonly buffers = new DeviceBuffers();
  private readonly pipelines: Map<Operation, unknown>;
  private readonly gpuExecutor?: GpuGraphExecutor;
  private readonly pendingMaterializations = new WeakMap<
    Tensor,
    Promise<Float32Array>
  >();
  private readonly pendingExecutions = new Set<Promise<Float32Array>>();
  private runtime?: Runtime;
  private disposed = false;
  private disposing = false;
  private _lastFusionDecision?: ElementwiseFusionDecision;
  private _lastDispatchCount?: number;

  private constructor(
    backend: BackendInfo,
    runtime?: Runtime,
    pipelines: ReadonlyMap<Operation, unknown> = new Map(),
  ) {
    this.backend = backend;
    this.runtime = runtime;
    this.pipelines = new Map(pipelines);
    if (runtime)
      this.gpuExecutor = new GpuGraphExecutor(
        runtime,
        this.pipelines,
        this.buffers,
        (decision, dispatches) => {
          this._lastFusionDecision = decision;
          this._lastDispatchCount = dispatches;
        },
      );
  }

  static async open(options: DeviceOptions = {}): Promise<Device> {
    const preference = options.device ?? "auto";
    if (!isDevicePreference(preference))
      throw new TypeError(
        `Unknown TypeNN device preference: ${String(preference)}.`,
      );
    if (activeDevice && !activeDevice.isDisposed())
      throw new Error(
        "TypeNN currently supports one active Device per JavaScript context.",
      );

    const { runtime, fallbackReason } = await selectRuntime(preference);

    if (runtime) {
      configure({ runtime });
      try {
        const pipelines = await preparePipelines(runtime);
        const device = new Device(
          {
            provider: "typeshade",
            name: runtime.tier,
            ...(fallbackReason ? { fallbackReason } : {}),
          },
          runtime,
          pipelines,
        );
        activeDevice = device;
        return device;
      } catch (error) {
        runtime.destroy();
        configure({ prefer: ["webgpu", "webgl2", "cpu"], runtime: null });
        throw error;
      }
    }
    if (!runtime) configure({ prefer: ["cpu"], runtime: null });
    const device = new Device({
      provider: "typeshade",
      name: "cpu",
      ...(fallbackReason ? { fallbackReason } : {}),
    });
    activeDevice = device;
    return device;
  }

  tensor(
    values: ArrayLike<number>,
    options: { readonly shape: Shape },
  ): Tensor {
    this.assertUsable();
    const shape = checkedShape(options.shape);
    const length = sizeOf(shape);
    if (values.length !== length)
      throw new RangeError(
        `Tensor data has ${values.length} values; shape requires ${length}.`,
      );
    const copy = Float32Array.from(values);
    return new Tensor(this, shape, { kind: "input", value: copy });
  }

  /** @internal Create a trainable leaf backed by the module's parameter array. */
  createParameter(
    values: Float32Array,
    options: { readonly shape: Shape },
  ): Tensor {
    this.assertUsable();
    const shape = checkedShape(options.shape);
    if (values.length !== sizeOf(shape))
      throw new RangeError("Parameter data length does not match its shape.");
    return new Tensor(this, shape, { kind: "input", value: values }, true);
  }

  /** @internal Allocate a host buffer for CPU or a TypeShade resident for device execution. */
  createKernelBuffer<T extends Float32Array | Uint32Array>(
    initial: T,
  ): T | Resident<T> {
    this.assertUsable();
    if (!this.runtime) return this.buffers.allocate(initial, false);
    return this.buffers.allocate(initial, true);
  }

  /** @internal Replace a reusable CPU or resident input buffer. */
  writeKernelBuffer<T extends Float32Array | Uint32Array>(
    target: T | Resident<T>,
    value: T,
  ): void {
    this.assertUsable();
    this.buffers.write(target, value);
  }

  /** @internal Read a kernel buffer to host memory. */
  async readKernelBuffer<T extends Float32Array | Uint32Array>(
    target: T | Resident<T>,
  ): Promise<T> {
    this.assertUsable();
    return this.buffers.read(target);
  }

  /** @internal Release a buffer allocated by createKernelBuffer. */
  releaseKernelBuffer<T extends Float32Array | Uint32Array>(
    target: T | Resident<T>,
  ): void {
    this.buffers.release(target);
  }

  /** @internal Run ordered training entries in one TypeShade frame, or on its CPU tier. */
  async runLinearTrainingFrame(
    commands: readonly LinearTrainingCommand[],
  ): Promise<void> {
    this.assertUsable();
    await runLinearTraining(this.runtime, this.pipelines, commands);
  }

  /** @internal */
  createOperation(shape: Shape, node: TensorNode): Tensor {
    this.assertUsable();
    const output = new Tensor(this, checkedShape(shape), node);
    registerTensorConsumers(output, node);
    return output;
  }

  /** @internal */
  materialize(tensor: Tensor): Promise<Float32Array> {
    this.assertUsable();
    const pending = this.pendingMaterializations.get(tensor);
    if (pending) return pending;
    const task = this.execute(tensor);
    const tracked = task.finally(() => {
      this.pendingMaterializations.delete(tensor);
      this.pendingExecutions.delete(tracked);
    });
    this.pendingMaterializations.set(tensor, tracked);
    this.pendingExecutions.add(tracked);
    return tracked;
  }

  private async execute(tensor: Tensor): Promise<Float32Array> {
    const existing = tensor.getMaterialized();
    if (existing instanceof Float32Array) return existing.slice();
    if (this.gpuExecutor) return this.gpuExecutor.execute(tensor);
    const value = await evaluateCpu(tensor);
    return value.slice();
  }

  /** @internal */
  async release(tensor: Tensor): Promise<void> {
    await this.pendingMaterializations.get(tensor)?.catch(() => undefined);
    const value = tensor.getMaterialized();
    if (value && !(value instanceof Float32Array))
      this.buffers.release(value as Resident<Float32Array>);
  }

  /** @internal Drop a cached resident value after host-side parameter updates. */
  async invalidate(tensor: Tensor): Promise<void> {
    await this.release(tensor);
    tensor.setMaterialized(undefined);
  }

  async dispose(): Promise<void> {
    if (this.disposed || this.disposing) return;
    this.disposing = true;
    await Promise.allSettled([...this.pendingExecutions]);
    this.disposed = true;
    this.buffers.dispose();
    this.runtime?.destroy();
    configure({ prefer: ["webgpu", "webgl2", "cpu"], runtime: null });
    if (activeDevice === this) activeDevice = undefined;
  }

  /** The fusion decision made for the most recent TypeShade graph execution. */
  get lastFusionDecision(): ElementwiseFusionDecision | undefined {
    return this._lastFusionDecision;
  }

  /** Number of TypeShade frame dispatches made by the most recent graph execution. */
  get lastDispatchCount(): number | undefined {
    return this._lastDispatchCount;
  }

  /** @internal */
  assertUsable(): void {
    if (this.disposed || this.disposing)
      throw new Error("TypeNN Device has been disposed.");
  }

  /** @internal */
  isDisposed(): boolean {
    return this.disposed;
  }
}

export async function device(
  options: DeviceOptions | DevicePreference = {},
): Promise<Device> {
  return Device.open(
    typeof options === "string" ? { device: options } : options,
  );
}
