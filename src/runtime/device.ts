import {
  configure,
  createRuntime,
  resident,
  type Resident,
  type Runtime,
} from "typeshade/runtime";
import type { FuseReport } from "typeshade/fuse";
import { checkedShape, sizeOf, type Shape } from "../tensor/shape.js";
import type { OpNode } from "../tensor/node.js";
import { Tensor } from "../tensor/tensor.js";
import { execute, type Memory } from "./execute.js";
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

export class Device {
  readonly backend: BackendInfo;
  private readonly live = new Set<Resident<Float32Array>>();
  private readonly pending = new Set<Promise<unknown>>();
  private readonly history: FuseReport[] = [];
  private readonly memory: Memory = {
    keep: (values) => {
      const array = resident(values);
      this.live.add(array);
      return array;
    },
    drop: (array) => {
      this.live.delete(array);
      array.destroy();
    },
    record: (report) => {
      this.history.push(report);
      if (this.history.length > REPORT_HISTORY) this.history.shift();
    },
  };
  private runtime?: Runtime;
  private disposed = false;
  private disposing = false;

  private constructor(backend: BackendInfo, runtime?: Runtime) {
    this.backend = backend;
    this.runtime = runtime;
  }

  static async open(options: DeviceOptions = {}): Promise<Device> {
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
      name === "webgpu" ? runtime : undefined,
    );
    activeDevice = device;
    return device;
  }

  /** The report of the last plan a `data()` ran, or undefined before the first. */
  get lastReport(): FuseReport | undefined {
    return this.history[this.history.length - 1];
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
    this.assertUsable();
    const task: Promise<Float32Array> = execute(tensor, this.memory).finally(
      () => this.pending.delete(task),
    );
    this.pending.add(task);
    return task;
  }

  /** @internal Free the array of a tensor. */
  release(tensor: Tensor): void {
    const array = tensor.storage;
    if (array === undefined) return;
    tensor.storage = undefined;
    this.memory.drop(array);
  }

  async dispose(): Promise<void> {
    if (this.disposed || this.disposing) return;
    this.disposing = true;
    await Promise.allSettled([...this.pending]);
    this.disposed = true;
    for (const array of this.live) array.destroy();
    this.live.clear();
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
