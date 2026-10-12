import type { Resident } from "typeshade/runtime";
import type { Device } from "../runtime/device.js";
import type { Shape } from "./shape.js";
import { topologicalOrder } from "../autograd/graph.js";
import { backward } from "../autograd.js";
import type { TensorNode } from "./node.js";
import {
  binary,
  clipByValue,
  elu,
  leakyRelu,
  relu,
  reluGrad,
  special,
  unary,
} from "./operations/elementwise.js";
import { reshape } from "./operations/shape.js";
import type { Conv2DOptions, Pool2DOptions } from "./spatial.js";

export type { Conv2DOptions, Pool2DOptions } from "./spatial.js";

export interface MatmulOptions {
  readonly transposeA?: boolean;
  readonly transposeB?: boolean;
}

/** Until the operation has its kernel functions. */
function notYet(operation: string): never {
  throw new Error(
    `TypeNN operation ${operation} is not on kernel functions yet.`,
  );
}

export class Tensor {
  /**
   * @internal The resident array that holds the value. A plan makes it when it computes the
   * tensor, and the first plan that reads a tensor the caller made uploads it.
   */
  storage?: Resident<Float32Array>;
  private disposed = false;

  /** @internal */
  constructor(
    readonly device: Device,
    readonly shape: Shape,
    /** @internal */ readonly node: TensorNode,
    readonly requiresGrad = false,
  ) {}

  grad?: Tensor;

  relu(): Tensor {
    return relu(this);
  }

  reshape(shape: Shape): Tensor {
    return reshape(this, shape);
  }

  reluGrad(outputGradient: Tensor): Tensor {
    return reluGrad(this, outputGradient);
  }

  neg(): Tensor {
    return unary(this, "neg");
  }
  abs(): Tensor {
    return unary(this, "abs");
  }
  square(): Tensor {
    return unary(this, "square");
  }
  sqrt(): Tensor {
    return unary(this, "sqrt");
  }
  exp(): Tensor {
    return unary(this, "exp");
  }
  log(): Tensor {
    return unary(this, "log");
  }
  sigmoid(): Tensor {
    return unary(this, "sigmoid");
  }
  tanh(): Tensor {
    return unary(this, "tanh");
  }
  acos(): Tensor {
    return unary(this, "acos");
  }
  acosh(): Tensor {
    return unary(this, "acosh");
  }
  asin(): Tensor {
    return unary(this, "asin");
  }
  asinh(): Tensor {
    return unary(this, "asinh");
  }
  atan(): Tensor {
    return unary(this, "atan");
  }
  atanh(): Tensor {
    return unary(this, "atanh");
  }
  step(): Tensor {
    return unary(this, "step");
  }
  floor(): Tensor {
    return unary(this, "floor");
  }
  ceil(): Tensor {
    return unary(this, "ceil");
  }
  round(): Tensor {
    return unary(this, "round");
  }
  sign(): Tensor {
    return unary(this, "sign");
  }
  reciprocal(): Tensor {
    return unary(this, "reciprocal");
  }
  rsqrt(): Tensor {
    return unary(this, "rsqrt");
  }
  sin(): Tensor {
    return unary(this, "sin");
  }
  cos(): Tensor {
    return unary(this, "cos");
  }
  tan(): Tensor {
    return unary(this, "tan");
  }
  expm1(): Tensor {
    return unary(this, "expm1");
  }
  log1p(): Tensor {
    return unary(this, "log1p");
  }
  sinh(): Tensor {
    return unary(this, "sinh");
  }
  cosh(): Tensor {
    return unary(this, "cosh");
  }

  gelu(): Tensor {
    return special(this, "gelu");
  }
  relu6(): Tensor {
    return special(this, "relu6");
  }
  softplus(): Tensor {
    return special(this, "softplus");
  }
  leakyRelu(alpha = 0.2): Tensor {
    return leakyRelu(this, alpha);
  }
  elu(alpha = 1): Tensor {
    return elu(this, alpha);
  }
  clipByValue(minimum: number, maximum: number): Tensor {
    return clipByValue(this, minimum, maximum);
  }

  add(other: Tensor): Tensor {
    return binary(this, other, "add");
  }
  sub(other: Tensor): Tensor {
    return binary(this, other, "sub");
  }
  mul(other: Tensor): Tensor {
    return binary(this, other, "mul");
  }
  div(other: Tensor): Tensor {
    return binary(this, other, "div");
  }
  maximum(other: Tensor): Tensor {
    return binary(this, other, "maximum");
  }
  minimum(other: Tensor): Tensor {
    return binary(this, other, "minimum");
  }
  pow(other: Tensor): Tensor {
    return binary(this, other, "pow");
  }
  prelu(alpha: Tensor): Tensor {
    return binary(this, alpha, "prelu");
  }

  matmul(_other: Tensor, _options: MatmulOptions = {}): Tensor {
    return notYet("matmul");
  }

  conv2d(_filter: Tensor, _options: Conv2DOptions = {}): Tensor {
    return notYet("conv2d");
  }

  conv2dGradients(
    _filter: Tensor,
    _outputGradient: Tensor,
    _options: Conv2DOptions = {},
  ): readonly [Tensor, Tensor] {
    return notYet("conv2dGradients");
  }

  maxPool(_options: Pool2DOptions): Tensor {
    return notYet("maxPool");
  }
  avgPool(_options: Pool2DOptions): Tensor {
    return notYet("avgPool");
  }
  maxPoolGrad(_outputGradient: Tensor, _options: Pool2DOptions): Tensor {
    return notYet("maxPoolGrad");
  }
  avgPoolGrad(_outputGradient: Tensor, _options: Pool2DOptions): Tensor {
    return notYet("avgPoolGrad");
  }

  softmax(_axis = -1): Tensor {
    return notYet("softmax");
  }
  logSoftmax(_axis = -1): Tensor {
    return notYet("logSoftmax");
  }

  softmaxCrossEntropyWithLogits(
    _labels: Tensor,
    _options: { readonly reduction?: "none" | "sum" | "mean" } = {},
  ): Tensor {
    return notYet("softmaxCrossEntropyWithLogits");
  }

  sum(_axis?: number): Tensor {
    return notYet("sum");
  }
  mean(_axis?: number): Tensor {
    return notYet("mean");
  }
  prod(_axis?: number): Tensor {
    return notYet("prod");
  }
  max(_axis?: number): Tensor {
    return notYet("max");
  }
  min(_axis?: number): Tensor {
    return notYet("min");
  }
  argMax(_axis = 0): Tensor {
    return notYet("argMax");
  }
  argMin(_axis = 0): Tensor {
    return notYet("argMin");
  }

  async data(): Promise<Float32Array> {
    this.assertUsable();
    return this.device.materialize(this);
  }

  async backward(upstream?: Tensor): Promise<void> {
    await backward(this, upstream);
  }

  async disposeGraph(preserve: readonly Tensor[] = []): Promise<void> {
    const retained = new Set(preserve);
    await Promise.all(
      topologicalOrder(this)
        .filter((tensor) => !retained.has(tensor))
        .map((tensor) => tensor.dispose()),
    );
  }

  async zeroGrad(): Promise<void> {
    await this.grad?.dispose();
    this.grad = undefined;
  }

  /** @internal Update a parameter leaf, and the resident array that holds its value. */
  async assign(values: ArrayLike<number>): Promise<void> {
    this.assertUsable();
    if (!this.requiresGrad || this.node.kind !== "input")
      throw new TypeError(
        "assign() is only available on trainable parameter tensors.",
      );
    if (values.length !== this.node.value.length)
      throw new RangeError("Parameter assignment must preserve tensor size.");
    this.node.value.set(values);
    this.storage?.write(this.node.value);
  }

  /** @internal Host view used by the initial optimizer implementation. */
  parameterValues(): Float32Array {
    if (!this.requiresGrad || this.node.kind !== "input")
      throw new TypeError("Expected a trainable parameter tensor.");
    return this.node.value;
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    await this.device.release(this);
  }

  /** @internal */
  assertUsable(): void {
    if (this.disposed) throw new Error("Tensor has been disposed.");
    this.device.assertUsable();
  }
}
