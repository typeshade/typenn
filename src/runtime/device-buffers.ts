import { resident, type Resident } from "typeshade/runtime";

type KernelArray = Float32Array | Uint32Array;

export class DeviceBuffers {
  private readonly residents = new Set<Resident<any>>();

  allocate<T extends KernelArray>(initial: T, useResident: false): T;
  allocate<T extends KernelArray>(initial: T, useResident: true): Resident<T>;
  allocate<T extends KernelArray>(
    initial: T,
    useResident: boolean,
  ): T | Resident<T> {
    return useResident ? this.track(resident(initial)) : (initial.slice() as T);
  }

  track<T extends KernelArray>(buffer: Resident<T>): Resident<T> {
    this.residents.add(buffer);
    return buffer;
  }

  write<T extends KernelArray>(target: T | Resident<T>, value: T): void {
    if (ArrayBuffer.isView(target)) {
      if (target.length !== value.length)
        throw new RangeError(
          "Kernel buffer writes must preserve buffer length.",
        );
      target.set(value as never);
      return;
    }
    target.write(value);
  }

  async read<T extends KernelArray>(target: T | Resident<T>): Promise<T> {
    return ArrayBuffer.isView(target) ? (target.slice() as T) : target.read();
  }

  release<T extends KernelArray>(target: T | Resident<T>): void {
    if (ArrayBuffer.isView(target)) return;
    target.destroy();
    this.residents.delete(target);
  }

  dispose(): void {
    for (const buffer of this.residents) buffer.destroy();
    this.residents.clear();
  }
}
