import { Tensor } from "../tensor/tensor.js";

export abstract class Module {
  private readonly children = new Map<string, Module>();

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
}

export class Sequential extends Module {
  readonly layers: readonly Module[];

  constructor(...layers: Module[]) {
    super();
    if (layers.length === 0)
      throw new RangeError("Sequential requires at least one layer.");
    this.layers = layers.map((layer, index) =>
      this.register(String(index), layer),
    );
  }

  forward(input: Tensor): Tensor {
    return this.layers.reduce((value, layer) => layer.forward(value), input);
  }
}
