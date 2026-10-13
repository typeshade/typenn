import {
  configure,
  createRuntime,
  resident,
  type Resident,
  type Runtime,
} from "typeshade/runtime";
import type { FuseReport } from "typeshade/fuse";
import type { DevicePlan } from "./device-types.js";
import { checkedShape, sizeOf, type Shape } from "../tensor/shape.js";
import type { OpNode } from "../tensor/node.js";
import { Tensor } from "../tensor/tensor.js";
import { compute, execute, type Memory } from "./execute.js";
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

// TypeShade's direct call-layer configuration is process-global. A single active device keeps
// selection and resident ownership predictable.
let activeDevice: Device | undefined;

/** How many reports of plans `Device.reports` keeps. */
const REPORT_HISTORY = 256;

/**
 * The most bytes of released arrays the device keeps for reuse, unless `poolBytes` says otherwise.
 * Each array counts once, at its byte length: its host copy and its device buffer are one each.
 */
const DEFAULT_POOL_BYTES = 64 * 1024 * 1024;

export class Device {
  readonly backend: BackendInfo;
  /** Every array the device made and has not freed, with its number of elements. */
  private readonly live = new Map<Resident<Float32Array>, number>();
  /** The released arrays kept for reuse, by number of elements. Each is also in `live`. */
  private readonly free = new Map<number, Resident<Float32Array>[]>();
  private pooledBytes = 0;
  private readonly poolBytes: number;
  private readonly pending = new Set<Promise<unknown>>();
  private readonly history: FuseReport[] = [];
  private plan?: DevicePlan;
  private readonly memory: Memory = {
    keep: (values) => this.acquire(values.length, values),
    output: (length) => this.acquire(length),
    drop: (array) => {
      // An array is destroyed once: not again after the device freed it.
      if (!this.live.delete(array)) return;
      array.destroy();
    },
    recycle: (array) => this.recycle(array),
    record: (report, nodes) => {
      this.history.push(report);
      if (this.history.length > REPORT_HISTORY) this.history.shift();
      this.plan = { report, nodes: nodes.slice() };
    },
  };
  private runtime?: Runtime;
  private disposed = false;
  private disposing = false;

  private constructor(
    backend: BackendInfo,
    poolBytes: number,
    runtime?: Runtime,
  ) {
    this.backend = backend;
    this.poolBytes = poolBytes;
    this.runtime = runtime;
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
    if (activeDevice && !activeDevice.isDisposed())
      throw new Error(
        "TypeNN currently supports one active Device per JavaScript context.",
      );

    const { runtime, fallbackReason } = await selectRuntime(
      preference,
      options.gpuDevice,
    );
    const name: ExecutionTier = runtime?.tier ?? "cpu";
    try {
      // The kernel functions run on the first tier of `prefer` that can run them. An explicit
      // WebGPU device is required to run them all there; the other tiers list the CPU tier last
      // for the calls they cannot run, such as the ones that take an array of strides.
      configure({ runtime: name === "webgpu" ? runtime! : null });
      configure({
        prefer:
          name === "cpu"
            ? ["cpu"]
            : name === "webgl2"
              ? ["webgl2", "cpu"]
              : preference === "auto"
                ? ["webgpu", "cpu"]
                : ["webgpu"],
      });
    } catch (error) {
      runtime?.destroy();
      resetConfiguration();
      throw error;
    }
    // On WebGL2 the calls make a context of their own.
    if (name === "webgl2") runtime!.destroy();
    const device = new Device(
      {
        provider: "typeshade",
        name,
        ...(fallbackReason ? { fallbackReason } : {}),
      },
      poolBytes,
      name === "webgpu" ? runtime : undefined,
    );
    activeDevice = device;
    return device;
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

  /** The array for `length` values: a pooled one, with `values` written to it, or a new one. */
  private acquire(
    length: number,
    values?: Float32Array,
  ): Resident<Float32Array> {
    const pooled = this.takeFree(length);
    if (pooled !== undefined) {
      // A pooled array is written only where the caller gives values; an output is overwritten.
      if (values !== undefined) pooled.write(values);
      return pooled;
    }
    const array = resident(values ?? new Float32Array(length));
    this.live.set(array, length);
    return array;
  }

  private takeFree(length: number): Resident<Float32Array> | undefined {
    const list = this.free.get(length);
    const array = list?.pop();
    if (list !== undefined && list.length === 0) this.free.delete(length);
    if (array !== undefined)
      this.pooledBytes -= length * Float32Array.BYTES_PER_ELEMENT;
    return array;
  }

  /** Keep a released array for reuse while the pool has room, and free it otherwise. */
  private recycle(array: Resident<Float32Array>): void {
    const length = this.live.get(array);
    if (length === undefined) return;
    const bytes = length * Float32Array.BYTES_PER_ELEMENT;
    if (this.pooledBytes + bytes > this.poolBytes) {
      this.memory.drop(array);
      return;
    }
    const list = this.free.get(length);
    if (list === undefined) this.free.set(length, [array]);
    else list.push(array);
    this.pooledBytes += bytes;
  }

  async dispose(): Promise<void> {
    if (this.disposed || this.disposing) return;
    this.disposing = true;
    await Promise.allSettled([...this.pending]);
    this.disposed = true;
    // The pooled arrays are in `live` too, so each is destroyed once here.
    for (const array of this.live.keys()) array.destroy();
    this.live.clear();
    this.free.clear();
    this.pooledBytes = 0;
    this.runtime?.destroy();
    resetConfiguration();
    if (activeDevice === this) activeDevice = undefined;
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

/** The device and the order of tiers TypeShade's call layer has before a TypeNN device. */
function resetConfiguration(): void {
  configure({ runtime: null });
  configure({});
}

function isDevicePreference(value: string): value is DevicePreference {
  return ["auto", "webgpu", "webgl2", "cpu"].includes(value);
}

/** The program runtime that holds the device the calls run on, if the preference names a GPU. */
async function selectRuntime(
  preference: DevicePreference,
  gpuDevice: object | undefined,
): Promise<{ runtime?: Runtime; fallbackReason?: string }> {
  if (preference === "cpu") return {};
  try {
    return {
      runtime: await createRuntime(
        gpuDevice !== undefined
          ? { device: gpuDevice }
          : { prefer: [preference === "auto" ? "webgpu" : preference] },
      ),
    };
  } catch (error) {
    if (preference !== "auto") throw error;
    return {
      fallbackReason: error instanceof Error ? error.message : String(error),
    };
  }
}

export async function device(
  options: DeviceOptions | DevicePreference = {},
): Promise<Device> {
  return Device.open(
    typeof options === "string" ? { device: options } : options,
  );
}
