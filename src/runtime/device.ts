import {
  createDevice,
  type Device as ShadeDevice,
  type Resident,
} from "typeshade/runtime";
// The planner. `device.fuse()` runs through the slot this import fills (TypeShade change 0071,
// Amendment 1): the runtime ships none of the planner, so the application imports it once. The
// binding is read in `Device.open()`, since the pinned compiler's `package.json` does not list
// `src/fuse.ts` in `sideEffects` yet, and Vite drops a bare import of it (typeshade PR, 2026-10-19).
import { fuse as planner, type FuseReport } from "typeshade/fuse";
import type { DevicePlan } from "./device-types.js";
import { checkedShape, sizeOf, type Shape } from "../tensor/shape.js";
import type { OpNode } from "../tensor/node.js";
import { Tensor } from "../tensor/tensor.js";
import { compute, execute, type Memory } from "./execute.js";
import {
  DeviceDataset,
  type DeviceDatasetOptions,
} from "../data/device-dataset.js";
import type {
  BackendInfo,
  DeviceOptions,
  DevicePreference,
  ExecutionTier,
} from "./device-types.js";

export type {
  BackendInfo,
  DeviceOptions,
  DevicePreference,
  ExecutionTier,
} from "./device-types.js";

/** How many reports of plans `Device.reports` keeps. */
const REPORT_HISTORY = 256;

/**
 * The most bytes of released arrays the device keeps for reuse, unless `poolBytes` says otherwise.
 * The pool is TypeShade's (`createDevice({ poolBytes })`, change 0071): a released array comes
 * back for the next array of the same element and length, and the oldest released array leaves
 * first when the pool is full.
 */
const DEFAULT_POOL_BYTES = 64 * 1024 * 1024;

/**
 * A TypeNN device: the owner of the tensors' arrays and the plans that compute them, over a
 * TypeShade device owner (`createDevice()` of `typeshade/runtime`, change 0071). TypeShade holds
 * the tier, the arrays and their pool, the limits and the plans; TypeNN holds the tensors, the
 * graph and the report history. Two devices in one JavaScript context are independent.
 */
export class Device {
  readonly backend: BackendInfo;
  /** The TypeShade device owner: the tier, the arrays and their pool, the plans. */
  private readonly shade: ShadeDevice;
  private readonly pending = new Set<Promise<unknown>>();
  private readonly history: FuseReport[] = [];
  private plan?: DevicePlan;
  private readonly memory: Memory = {
    keep: (values) => this.shade.array(values),
    output: (length) => this.shade.array(length, "f32"),
    // A handle the host destroys leaves the owner's books; a second destroy is a no-op.
    drop: (array) => array.destroy(),
    recycle: (array) => this.shade.release(array),
    fuse: (body) => this.shade.fuse(body),
    record: (report, nodes) => {
      this.history.push(report);
      if (this.history.length > REPORT_HISTORY) this.history.shift();
      this.plan = { report, nodes: nodes.slice() };
    },
  };
  private disposed = false;
  private disposing = false;

  private constructor(backend: BackendInfo, shade: ShadeDevice) {
    this.backend = backend;
    this.shade = shade;
  }

  static async open(options: DeviceOptions = {}): Promise<Device> {
    const poolBytes = options.poolBytes ?? DEFAULT_POOL_BYTES;
    if (!Number.isSafeInteger(poolBytes) || poolBytes < 0)
      throw new RangeError(
        "The poolBytes option is a whole number of bytes, 0 or more.",
      );
    const preference = options.device ?? "auto";
    if (!isDevicePreference(preference))
      throw new TypeError(
        `Unknown TypeNN device preference: ${String(preference)}.`,
      );
    if (
      options.gpuDevice !== undefined &&
      (preference === "cpu" || preference === "webgl2")
    )
      throw new TypeError(
        `The gpuDevice option is a GPUDevice, which the ${preference} device does not use.`,
      );
    // The kernel functions run on the first tier of `prefer` that opens, and a call that tier
    // cannot run falls to the tiers after it. An explicit WebGPU device is required to run them
    // all there; `auto` and `webgl2` keep the CPU tier for the calls they cannot run, such as the
    // ones that take an array of strides.
    const shade = await createDevice(
      options.gpuDevice !== undefined
        ? { device: options.gpuDevice, poolBytes }
        : {
            prefer:
              preference === "auto"
                ? ["webgpu", "cpu"]
                : preference === "webgl2"
                  ? ["webgl2", "cpu"]
                  : [preference],
            poolBytes,
          },
    );
    if (typeof planner !== "function")
      throw new Error(
        "typeshade/fuse is not loaded: device.fuse() needs its planner.",
      );
    const name: ExecutionTier = shade.tier;
    const fallbackReason =
      preference === "auto" && shade.skipped.length > 0
        ? shade.skipped.map((s) => `${s.tier}: ${s.reason}`).join("; ")
        : undefined;
    return new Device(
      {
        provider: "typeshade",
        name,
        ...(fallbackReason ? { fallbackReason } : {}),
      },
      shade,
    );
  }

  /** The report of the last plan a `data()` ran, or undefined before the first. */
  get lastReport(): FuseReport | undefined {
    return this.history[this.history.length - 1];
  }

  /**
   * The last plan the executor ran, with the graph id of each kernel call in its scope
   * (`nodes[k]` for the k-th call), or undefined before the first. A failed plan leaves
   * `lastPlan`, `lastReport` and `reports` as they were.
   */
  get lastPlan(): DevicePlan | undefined {
    return this.plan;
  }

  /** The reports of the last 256 plans, oldest first. */
  get reports(): readonly FuseReport[] {
    return this.history;
  }

  clearReports(): void {
    this.history.length = 0;
  }

  /** What the TypeShade device owner counts of the arrays: live, pooled, allocations, reuses. */
  get statistics(): ShadeDevice["statistics"] {
    return this.shade.statistics;
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

  /** @internal */
  createOperation(shape: Shape, node: OpNode): Tensor {
    this.assertUsable();
    return new Tensor(this, checkedShape(shape), node);
  }

  /** @internal */
  materialize(tensor: Tensor): Promise<Float32Array> {
    return this.materializeAll([tensor]).then(([values]) => values!);
  }

  /** @internal Compute and read several tensors in one plan. */
  materializeAll(tensors: readonly Tensor[]): Promise<Float32Array[]> {
    this.assertUsable();
    const task: Promise<Float32Array[]> = execute(tensors, this.memory).finally(
      () => this.pending.delete(task),
    );
    this.pending.add(task);
    return task;
  }

  /**
   * @internal Compute several tensors in one plan and keep their arrays. The tensors of `alsoRead`
   * are computed in that plan too, and their values are read back once, after it.
   */
  computeAll(
    tensors: readonly Tensor[],
    alsoRead: readonly Tensor[] = [],
  ): Promise<Float32Array[]> {
    this.assertUsable();
    const task: Promise<Float32Array[]> = execute(
      tensors,
      this.memory,
      alsoRead,
    ).finally(() => this.pending.delete(task));
    this.pending.add(task);
    return task;
  }

  /**
   * @internal Make the computed array of `source` the array of `target`, and free the array
   * `target` held. `source` keeps none. The host value of `target` is stale until it is read.
   */
  adopt(target: Tensor, source: Tensor): void {
    this.assertUsable();
    const array = source.storage;
    if (array === undefined)
      throw new Error("adopt() needs a tensor that a plan computed.");
    try {
      target.assertUsable();
    } catch (error) {
      // A disposed target takes no array: the computed one is released, not leaked.
      this.release(source);
      throw error;
    }
    this.release(target);
    target.storage = array;
    target.hostStale = true;
    source.storage = undefined;
  }

  /** @internal Give back the array of a tensor, which the device reuses or frees. */
  release(tensor: Tensor): void {
    const array = tensor.storage;
    if (array === undefined) return;
    tensor.storage = undefined;
    this.memory.recycle(array);
  }

  /**
   * @internal A resident array of `values` that this device owns, for what is not a Tensor: the
   * shards of a dataset. The caller destroys it.
   */
  keep<T extends Uint32Array | Float32Array>(values: T): Resident<T> {
    this.assertUsable();
    return this.shade.array(values);
  }

  async dispose(): Promise<void> {
    if (this.disposed || this.disposing) return;
    this.disposing = true;
    await Promise.allSettled([...this.pending]);
    this.disposed = true;
    // The owner destroys every array it holds, pooled or not, each once, and the WebGPU device it
    // requested; a GPUDevice the application handed in is left as it is.
    await this.shade.dispose();
  }

  /** @internal Create a dataset of device-resident sources; see `DeviceDataset`. */
  dataset(options: DeviceDatasetOptions): DeviceDataset {
    this.assertUsable();
    return new DeviceDataset(this, options);
  }

  /**
   * @internal The largest storage binding of the WebGPU device, from its limits, or undefined when
   * the device is not WebGPU or does not report it.
   */
  storageBindingLimit(): number | undefined {
    const binding = this.shade.limits["maxStorageBufferBindingSize"];
    if (typeof binding !== "number") return undefined;
    // A shard is one buffer, so it also fits maxBufferSize.
    const buffer = this.shade.limits["maxBufferSize"];
    return typeof buffer === "number" ? Math.min(binding, buffer) : binding;
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

function isDevicePreference(value: string): value is DevicePreference {
  return ["auto", "webgpu", "webgl2", "cpu"].includes(value);
}

export async function device(
  options: DeviceOptions | DevicePreference = {},
): Promise<Device> {
  return Device.open(
    typeof options === "string" ? { device: options } : options,
  );
}
