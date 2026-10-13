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
import {
  conv2d,
  conv2dGradients,
  matmul,
  type MatmulOptions,
} from "./operations/linear-algebra.js";
import { softmaxCrossEntropyWithLogits } from "./operations/loss.js";
import { softmax } from "./operations/normalization.js";
import { reduce } from "./operations/reductions.js";
import { reshape } from "./operations/shape.js";
import { pool2d, pool2dGrad } from "./operations/spatial.js";
import type { Conv2DOptions, Pool2DOptions } from "./spatial.js";

export type { Conv2DOptions, Pool2DOptions } from "./spatial.js";

export class Tensor {
  /**
   * @internal The resident array that holds the value. A plan makes it when it computes the
   * tensor, and the first plan that reads a tensor the caller made uploads it.
   */
  storage?: Resident<Float32Array>;
  /**
   * @internal True when `storage` holds a newer value than the host copy of an input tensor. An
   * optimizer step sets it; `hostValues()` reads the array back once and clears it.
   */
  hostStale = false;
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

  matmul(other: Tensor, options: MatmulOptions = {}): Tensor {
    return matmul(this, other, options);
  }

  conv2d(filter: Tensor, options: Conv2DOptions = {}): Tensor {
    return conv2d(this, filter, options);
  }

  conv2dGradients(
    filter: Tensor,
    outputGradient: Tensor,
    options: Conv2DOptions = {},
  ): readonly [Tensor, Tensor] {
    return conv2dGradients(this, filter, outputGradient, options);
  }

  maxPool(options: Pool2DOptions): Tensor {
    return pool2d(this, "maxPool", options);
  }
  avgPool(options: Pool2DOptions): Tensor {
    return pool2d(this, "avgPool", options);
  }
  maxPoolGrad(outputGradient: Tensor, options: Pool2DOptions): Tensor {
    return pool2dGrad(this, "maxPoolGrad", outputGradient, options);
  }
  avgPoolGrad(outputGradient: Tensor, options: Pool2DOptions): Tensor {
    return pool2dGrad(this, "avgPoolGrad", outputGradient, options);
  }

  softmax(axis = -1): Tensor {
    return softmax(this, "softmax", axis);
  }
  logSoftmax(axis = -1): Tensor {
    return softmax(this, "logSoftmax", axis);
  }

  softmaxCrossEntropyWithLogits(
    labels: Tensor,
    options: { readonly reduction?: "none" | "sum" | "mean" } = {},
  ): Tensor {
    return softmaxCrossEntropyWithLogits(this, labels, options);
  }

  sum(axis?: number): Tensor {
    return reduce(this, "sum", axis);
  }
  mean(axis?: number): Tensor {
    return reduce(this, "mean", axis);
  }
  prod(axis?: number): Tensor {
    return reduce(this, "prod", axis);
  }
  max(axis?: number): Tensor {
    return reduce(this, "max", axis);
  }
  min(axis?: number): Tensor {
    return reduce(this, "min", axis);
  }
  argMax(axis = 0): Tensor {
    return reduce(this, "argMax", axis);
  }
  argMin(axis = 0): Tensor {
    return reduce(this, "argMin", axis);
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
    this.hostStale = false;
    this.storage?.write(this.node.value);
  }

  /**
   * The host value of a trainable parameter. After an optimizer step the value is on the device
   * only, and this reads it back once. The returned array is the parameter's own host array.
   */
  async parameterValues(): Promise<Float32Array> {
    this.assertUsable();
    if (!this.requiresGrad || this.node.kind !== "input")
      throw new TypeError("Expected a trainable parameter tensor.");
    return this.hostValues();
  }

  /**
   * @internal The host value of an input tensor. A stale one is read back from its array first.
   */
  async hostValues(): Promise<Float32Array> {
    const node = this.node;
    if (node.kind !== "input")
      throw new TypeError("Only an input tensor has a host value.");
    if (this.hostStale) {
      node.value.set(await this.storage!.read());
      this.hostStale = false;
    }
    return node.value;
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    // The value of a stale parameter goes back to the host before its array is freed.
    if (this.hostStale && !this.device.isDisposed()) await this.hostValues();
    await this.device.release(this);
  }

  /** @internal */
  assertUsable(): void {
    if (this.disposed) throw new Error("Tensor has been disposed.");
    this.device.assertUsable();
  }
}
