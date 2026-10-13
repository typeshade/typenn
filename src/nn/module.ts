import type { Tensor } from "../tensor/tensor.js";
import type { Device } from "../runtime/device.js";

export abstract class Module {
  private readonly children = new Map<string, Module>();
  private training = true;
  private device?: Device;

  abstract forward(input: Tensor): Tensor;

  /** Register a child so Module subclasses can expose a stable model tree. */
  protected register<T extends Module>(name: string, child: T): T {
    if (!name || this.children.has(name))
      throw new Error(`Duplicate or empty module name: ${name}.`);
    this.children.set(name, child);
    return child;
  }

  modules(): ReadonlyMap<string, Module> {
    return this.children;
  }

  to(device: Device): this {
    device.assertUsable();
    this.device = device;
    for (const child of this.children.values()) child.to(device);
    return this;
  }

  parameters(): Tensor[] {
    if (!this.device)
      throw new Error("Call model.to(device) before requesting parameters.");
    return [
      ...this.ownParameters(this.device),
      ...[...this.children.values()].flatMap((child) => child.parameters()),
    ];
  }

  /**
   * Refresh the host arrays of every layer in this module. The optimizers update the parameters
   * on the device, so a layer's `weight` and `bias` arrays are host copies that lag behind
   * until `sync()` (or `Trainer.fit()`) reads the parameters back, in one plan. Does nothing
   * before `to(device)`, or when no parameter has changed on the device.
   */
  async sync(): Promise<void> {
    if (!this.device) return;
    const stale = this.parameters().filter((parameter) => parameter.hostStale);
    if (stale.length > 0) await this.device.materializeAll(stale);
  }

  protected ownParameters(_device: Device): readonly Tensor[] {
    return [];
  }

  protected assertInputDevice(device: Device): void {
    if (this.device && this.device !== device)
      throw new TypeError(
        "The input Tensor is on a different device than this module.",
      );
  }

  train(): this {
    this.training = true;
    for (const child of this.children.values()) child.train();
    return this;
  }

  eval(): this {
    this.training = false;
    for (const child of this.children.values()) child.eval();
    return this;
  }

  isTraining(): boolean {
    return this.training;
  }
}
