# TypeNN

TypeNN is a TypeScript neural-network library built on TypeShade. An operation on a Tensor adds a
node to a graph. Each node is one TypeShade kernel function. A read such as `data()` runs the graph
as one TypeShade `fuse` scope.

```ts
import * as nn from "typenn";

const device = await nn.device();
const a = device.tensor([1, 2, 3, 4], { shape: [2, 2] });
const b = device.tensor([5, 6, 7, 8], { shape: [2, 2] });
const y = a.matmul(b).relu();

console.log(await y.data()); // Float32Array [19, 22, 43, 50]
await device.dispose();
```

The function API follows TensorFlow.js-style names. The Tensor methods do the same work:

```ts
const logits = nn.add(nn.matMul(input, weight), bias);
const probabilities = nn.softmax(nn.relu(logits));
console.log(await probabilities.data());
```

The operation set is:

- rank-2 `matMul`, with `transposeA` and `transposeB`
- `add`, `sub`, `mul`, `div`, `maximum`, `minimum`, `pow` and `prelu`, with NumPy-style broadcasting
  up to rank 8
- elementwise arithmetic, rounding, trigonometric and activation functions
- `sum`, `mean`, `prod`, `max`, `min`, `argMax` and `argMin` over one axis or over all axes
- `softmax` and `logSoftmax` over any axis, and logits-based softmax cross entropy
- NHWC `conv2d`, `maxPool` and `avgPool`
- `reshape`

[TODO.md](./TODO.md) lists what is open. `Linear`, `Conv2d`, `MaxPool2d`, `AvgPool2d`, `ReLU`, `Tanh`, `Sigmoid` and `Sequential` compose
operations for inference:

```ts
const model = new nn.Sequential(new nn.Linear(784, 10));
const logits = model.forward(device.tensor(pixels, { shape: [1, 784] }));
const probabilities = await logits.softmax().data();
```

## How a read runs

An operation does not compute. It records a node. The node holds the kernel function, the input
Tensors, the scalars the kernel function takes by value, and the gradient rule. `data()` then runs
these steps:

1. Collect the nodes under the Tensor that have no array yet, in graph order.
2. Make one `Resident` array for each node, and upload each leaf that has no array yet.
3. Call `fuse` (`typeshade/fuse`) with a body that calls `kernel(out, ...inputs, ...values)` for each
   node.
4. Keep the `FuseReport` of the scope as `device.lastReport`.
5. Read the Tensor back, once.

The compiler plans the scope. It puts consecutive calls in one dispatch when the facts of TypeShade
Rule 11.13 hold. `device.lastReport` shows the plan:

```ts
await a.add(b).mul(c).relu().data();
console.log(device.lastReport);
// { tier: "webgpu", dispatches: 1, submissions: 1,
//   groups: [{ calls: ["add", "mul", "relu"], fused: true }] }
```

Three facts end a group in the plans that the tests assert:

- The loops differ in length (F2). A reduction or a layer that changes the element count starts a
  new dispatch.
- A call reads, at an index other than `i`, an array that the call before it wrote (F3). `matmul`
  and the strided forms of the binary operations do this.
- One dispatch would bind more storage buffers than the device allows, 8 by default (F5). Every
  operation output is a `Resident` array, so a long chain of distinct inputs splits.

`group.why` says why a group stands apart from the group before it, with the source line.
`device.reports` keeps the last 256 reports, oldest first, and `device.clearReports()` empties it.
The type `FuseReport` is exported.

The same code path runs on every tier. On the CPU tier the scope runs the calls one after the
other. The report then has `dispatches: 0`, `submissions: 0` and one group for each call.

A Tensor keeps its array after a read, so a later `data()` of it, or of a Tensor under it, adds no
dispatch for it. `dispose()` and `disposeGraph()` free the arrays. A call that takes one Tensor
twice, such as `a.add(a)`, reads the second one from a copy in a scratch array. One `Resident` must
not be two parameters of one call. If a plan fails, TypeNN frees the arrays of that plan and
rethrows the error. The next `data()` computes them again.

## Inspecting a graph

`nn.graph(tensor)` returns the graph under a Tensor. It lists each reachable Tensor once. An input
comes before each Tensor that uses it:

```ts
const snapshot = nn.graph(a.matmul(b).add(c).relu());
for (const node of snapshot.nodes)
  console.log(node.id, node.op, node.shape, node.inputs);
```

Each node has an `id`, which stays the same for the life of its Tensor. `computed` is true when the
Tensor already holds an array. `computed` is true for every input Tensor. A later `data()` does not
compute it.

After a read, `device.lastPlan` describes the plan. Its `report` is the `FuseReport`, and its `nodes`
holds the graph id of each kernel call, in call order. The calls of `report.groups` line up with
`nodes`. A copy that a repeated input needs takes the id of the operation that reads it.
`device.lastReport` and `device.reports` do not change.

If a plan fails, `device.lastPlan`, `device.lastReport` and `device.reports` keep the values they had
before the failed plan.

## Devices and tiers

`nn.device(preference)` selects where the kernel functions run.

| Preference         | Tiers, in order                              | Tier missing                                                |
| ------------------ | -------------------------------------------- | ----------------------------------------------------------- |
| `"auto"` (default) | WebGPU, then CPU                             | Falls back to CPU. `device.backend.fallbackReason` says why |
| `"webgpu"`         | WebGPU                                       | `nn.device()` rejects                                       |
| `"webgl2"`         | WebGL2, then CPU for the calls it cannot run | `nn.device()` rejects                                       |
| `"cpu"`            | CPU                                          | Not applicable                                              |

`device.backend.name` is the tier selected. The option `gpuDevice` hands TypeNN a `GPUDevice` that
the application owns. TypeNN runs its kernel functions on it and never destroys it. TypeShade's
call-layer configuration is process-global, so TypeNN permits one active device per JavaScript
context.

## Kernel functions

The kernel functions are in `src/kernels`. Each is an exported TypeShade function with at least one
`array<f32>` parameter and one loop that writes `out[i]` (TypeShade Rule 8.22). The host calls it as
`kernel(out, ...arrays, ...values)`. `out` and the arrays are `Resident` arrays. The scalars and the
fixed-size arrays follow them, by value. `vite.config.ts` builds them with `typeshade({ fuse: true })`.

| File               | Kernel functions                                                                                                                                              |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `map.shade.ts`     | `copy`, `relu`, `reluGrad`, and the maps `<operation>Map` such as `absMap`, `expMap` and `sigmoidMap`                                                         |
| `binary.shade.ts`  | `add`, `sub`, `mul`, `div`, `maximum`, `minimum`, `power` and `prelu` for equal shapes, their `Expand` and `Strided` forms for broadcasting, and `binaryGrad` |
| `reduce.shade.ts`  | `sumAxis`, `meanAxis`, `prodAxis`, `maxAxis`, `minAxis`, `argMaxAxis`, `argMinAxis` and `reduceGrad`                                                          |
| `matmul.shade.ts`  | `matmul`                                                                                                                                                      |
| `softmax.shade.ts` | `softmax`, `logSoftmax`, `softmaxGrad` and `logSoftmaxGrad`                                                                                                   |
| `loss.shade.ts`    | `softmaxCrossEntropy`, `crossEntropyLogitsGrad` and `crossEntropyLabelsGrad`                                                                                  |
| `conv.shade.ts`    | `conv2d`, `conv2dInputGrad` and `conv2dFilterGrad`                                                                                                            |
| `pool.shade.ts`    | `maxPool`, `avgPool`, `maxPoolGrad` and `avgPoolGrad`                                                                                                         |

`add(out, a, b)` reads `a[i]` and `b[i]` and writes `out[i]`. `addExpand` reads `b` through the
strides of the broadcast, and `addStrided` reads both operands that way.

Two naming rules come from the compiler:

- A kernel function cannot share the name of a TypeShade builtin that it calls (TS8099 and TS8031).
  A map is therefore `absMap`, and the power operation is `power`.
- WGSL reserves some words, such as `filter`. A WebGPU pipeline refuses a kernel function whose
  array parameter has such a name. The refusal comes at pipeline creation, and not at build time.
  `conv2d` therefore names its array `weights`.

To add an operation:

1. Write the kernel function in a file of `src/kernels`.
2. Name it so that it does not equal a builtin that it calls.
3. Write the operation in `src/tensor/operations`. It checks the shapes and returns
   `operation(shape, "name", kernel, inputs, values, backward)`.
4. Write `backward` as a closure that returns one gradient Tensor for each input, or `undefined`.
5. Add the Tensor method in `src/tensor/tensor.ts`.
6. Add the function in `src/nn/ops`.
7. Export the function from `src/index.ts`.
8. Add a case to `tests/_cases.mjs`.
9. Add the plain-JavaScript reference of the case to `tests/_reference.mjs`.
10. Run `npm test`.

## PyTorch-style model training

Models can subclass `Module`, register layers, and implement `forward()`. Parameters are
created on the device after `model.to(device)`. The training loop is explicit: select training
mode, clear gradients, compute logits and loss, call `backward()`, and step the optimizer.

```ts
class SimpleMLP extends nn.Module {
  private readonly flatten = this.register("flatten", new nn.Flatten());
  private readonly fc1 = this.register("fc1", new nn.Linear(784, 128));
  private readonly relu = this.register("relu", new nn.ReLU());
  private readonly dropout = this.register("dropout", new nn.Dropout(0.2));
  private readonly fc2 = this.register("fc2", new nn.Linear(128, 10));

  forward(input: nn.Tensor): nn.Tensor {
    return this.fc2.forward(
      this.dropout.forward(
        this.relu.forward(this.fc1.forward(this.flatten.forward(input))),
      ),
    );
  }
}

const model = new SimpleMLP().to(device);
const optimizer = nn.optim.Adam(model.parameters(), { lr: 0.001 });
const criterion = new nn.CrossEntropyLoss();
model.train();

for await (const batch of batches) {
  await optimizer.zeroGrad();
  const images = device.tensor(batch.images, {
    shape: [batch.size, 1, 28, 28],
  });
  const logits = model.forward(images);
  const loss = criterion.forward(logits, batch.labels); // integer class IDs
  await loss.backward();
  await optimizer.step();
  await loss.disposeGraph(model.parameters());
}

model.eval();
```

The complete IDX dataset loader, shuffled batch loop, and evaluation example are in
[`examples/mnist`](./examples/mnist). It keeps images as uint8 on the host, normalizes each
batch, uses logits directly with sparse-label cross entropy, and evaluates by argmax. The Tensor
forward and backward operations run as kernel functions on the selected tier. An optimizer step
computes the gradients and the parameter and moment updates in one plan, on the device. It reads no
gradient back and uploads no parameter. This example demonstrates the API. Its speed on SwiftShader
is in the Measurements section.

`createTrainer()` remains limited to a single `Linear` classifier and SGD. It runs the same Tensor
graph as the explicit loop: `matmul`, bias, softmax cross entropy, `backward()` and an `SGD` step.
The explicit loop supports a custom `Module` with multiple Linear layers, ReLU, Dropout, and the
current Tensor-graph gradient rules.

TypeShade owns the shader language and the GPU execution runtime. TypeNN's `.shade.ts` kernel
functions are compiled during the library build. Applications use TypeNN's Tensor and Module API
and do not author shaders. The build pins TypeShade to the source commit `5fb79c7`, which has
`typeshade/fuse`, because TypeShade's npm registry package is still a placeholder.

For custom functions written as pure TypeShade IR, TypeNN also re-exports TypeShade's
`grad` and `gradCheck` as `shadeGrad` and `shadeGradCheck`. They can generate and check
derivatives of those Shade functions. They do not differentiate TypeNN Tensor graphs. The
separate, limited Tensor-graph rules below cover selected TypeNN operations.

Conv2D and pooling also expose explicit gradient operations for lower-level use:

```ts
const [inputGradient, filterGradient] = nn.conv2dGradients(
  input,
  filter,
  outputGradient,
  {
    strides: 1,
    padding: "same",
  },
);
const poolInputGradient = nn.maxPoolGrad(poolInput, poolOutputGradient, {
  filterSize: 2,
  strides: 2,
});
```

These return deferred Tensors and run through TypeShade on the selected device. `maxPoolGrad`
routes tied maxima to the first element in scan order. They are primitive gradient operations.
The initial reverse-mode graph traversal uses them as well:

```ts
const [weightGradient] = nn.gradients(loss, [weight]);
console.log(await weightGradient.data());
```

`gradients(output, wrt, upstream?)` returns deferred gradient Tensors. Without an explicit
`upstream`, `output` must contain one value. The first graph rules cover ReLU, broadcast
add/subtract/multiply/divide, unary arithmetic and trigonometric ops, `sigmoid` and `tanh`, rank-2 MatMul including
transpose options, Conv2D, 2D pooling, `sum`/`mean` reductions, and logits-based softmax cross
entropy. `prod`/`max`/`min` reduction gradients and special activation gradients remain
unsupported, and ask for the missing rule by name when called.

## Current scope

- `device()` selects a TypeShade tier: WebGPU, WebGL2 or the CPU.
- Tensor operation calls construct a lazy dependency graph. The first result read runs it.
- Pure numerical operations are TypeShade kernel functions on every tier. TypeNN keeps no separate
  JavaScript math implementation. The tests keep one, as a reference.
- Intermediate tensors stay resident until disposed or the device is disposed. Use `disposeGraph()`
  to release an operation graph while preserving parameters.
- `Module.to()`, `parameters()`, `train()`, and `eval()` support user-defined models and
  registered child layers. `Flatten`, `Linear`/`Dense`, `Conv2d`, `MaxPool2d`, `AvgPool2d`, `ReLU`, `Tanh`, `Sigmoid` and
  `Dropout` are the layers. `MaxPool2d` and `AvgPool2d` default `strides` to `filterSize`
  (the `maxPool` and `avgPool` operations default it to 1). The example uses
  `Flatten`, `Linear`, `ReLU` and `Dropout`. The losses are `CrossEntropyLoss`, with sparse integer
  labels, and `MSELoss`, the mean of squared differences. They are `nn.loss.crossEntropy` and
  `nn.loss.mse`.
- `Tensor.backward()` accumulates gradients on registered parameter leaves. SGD and Adam update
  the parameters, and the Adam moments, on the device, in one plan (`src/kernels/optim.shade.ts`).
  A step reads no gradient back and uploads no parameter. The moments start at zero on the first
  step. A parameter keeps its Tensor object, and its host copy is read back on demand:
  `parameterValues()`, `data()` and `dispose()` do it. A layer's `weight` and `bias` arrays are the
  host copy of its parameters. They lag behind the device, and `await model.sync()` refreshes the
  arrays of every layer in one plan. The `fit()` of a Trainer does it at its end, also when the fit
  fails.
- `parameterValues()` is `async` and returns `Promise<Float32Array>`. It was synchronous, and it
  returned the host array without a read. Add an `await` to each call. `Module.sync()` and
  `Optimizer.dispose()` are new. `dispose()` frees the Adam moments on the device (the moments
  also go with the device). `Device.dispose()` does not read the parameters back: call `sync()`
  first when the host arrays must hold the trained values.
- Tensor-graph autograd remains limited. Model serialization and WebNN are not implemented.
- `shadeGrad`/`shadeGradCheck` operate on pure TypeShade functions only.

## Measurements

Before is `main` at commit `77dbb19`, which pins TypeShade `946b0ea`. After is this branch, which
pins TypeShade `5fb79c7`. Both were measured on one machine.

### Calls the device is given

The counts come from `node scripts/bench-step.mjs --scenario counts`. They are the calls of the page
to `GPUComputePassEncoder.dispatchWorkgroups` and `GPUQueue.prototype.submit`, on WebGPU in
Chromium 141 with SwiftShader. A count of submissions includes the submission that reads the result
back. The plan alone is one submission (`device.lastReport.submissions`).

| One `data()` of                                   | Dispatches before | Dispatches after | Submissions before | Submissions after |
| ------------------------------------------------- | ----------------: | ---------------: | -----------------: | ----------------: |
| the chain `a.add(b).mul(c).relu()`                |                 3 |                1 |                  2 |                 2 |
| a dense layer `x.matmul(w).add(bias).relu()`      |                 3 |                1 |                  2 |                 2 |
| MNIST forward, `loss.data()`                      |                 9 |                5 |                  2 |                 2 |
| MNIST backward, four `grad.data()` calls          |                11 |               10 |                  8 |                 8 |
| MNIST backward, `optimizer.step()` reads all four |                11 |               10 |                  8 |                 1 |

The graphs are the chain of 1,024 elements, a dense layer of 784 inputs and 128 outputs on 64 rows,
and the model of `examples/mnist` on a batch of 64. Before, an optimizer read each gradient with a
`data()` of its own. In the last row the step of this branch computes the gradients inside its
update plan, so it is one submission. The dispatches stay 10, because the update calls join the
dispatches of the backward pass.

### Size of the build

The size of the index chunk of `npm run build` (Vite 6.4.4 on Node 22). Raw sizes are before gzip.

| Build                                              |         Raw |      gzip |
| -------------------------------------------------- | ----------: | --------: |
| `77dbb19`, TypeShade `946b0ea`                     | 2,222.56 kB | 198.27 kB |
| TypeShade `5fb79c7` with `fuse: true`, old kernels | 2,223.09 kB | 199.48 kB |
| this branch, built without `fuse` (not shipped)    | 1,026.89 kB | 210.99 kB |
| this branch                                        | 1,485.90 kB | 230.42 kB |

The raw size falls by a third, and the gzip size rises by 32.15 kB (16 percent). The build without
`fuse` shows where the rise comes from. The option `fuse: true` accounts for 19.43 kB gzip of it, and
the rewrite itself for the other 12.72 kB. The WebGL2 chunk grows from 13.25 kB to 19.55 kB gzip.
Inference, not measured: the 73 old entry files shared much text, which gzip compresses well, so the
raw size falls more than the gzip size.

### Time on SwiftShader

SwiftShader is a software device. The times below say what a step costs in this harness, and not how
fast a hardware GPU trains. The median of each round is in milliseconds. The rounds of the two builds
alternate. The time is host wall time.

| Scenario                                   | Build  | Rounds                     | Median of the rounds |
| ------------------------------------------ | ------ | -------------------------- | -------------------: |
| MNIST step, batch 64, Adam (20 steps each) | before | 206.2, 183.7, 181.5, 168.7 |                182.6 |
| MNIST step, batch 64, Adam (20 steps each) | after  | 235.6, 188.0, 166.8, 196.6 |                192.3 |
| chain, 32,768 elements                     | before | 14.8, 15.3, 14.9           |                 14.9 |
| chain, 32,768 elements                     | after  | 7.5, 11.5, 8.5             |                  8.5 |
| chain, 1,048,576 elements                  | before | 513.4, 493.9, 467.8        |                493.9 |
| chain, 1,048,576 elements                  | after  | 262.0, 291.6, 279.4        |                279.4 |

The rounds of the MNIST step overlap, so the two builds do not differ outside the noise. The device
is given 20 dispatches in 10 submissions per step before, and 15 dispatches in 7 submissions after.
The chain is faster after, in each round. Run `node scripts/bench-step.mjs --scenario step` or
`--scenario chain` to repeat a row. `--dist <directory>` runs the script on the `dist` of another
build.

### Optimizer on the device

The baseline here is `main` at commit `31e5387`, which the optimizer updated on the host. Both builds
run the same `scripts/bench-step.mjs` on Chromium 141 with SwiftShader, and the rounds alternate.

| Measure, one `step()` or one training step | Build  | Result                                   |
| ------------------------------------------ | ------ | ---------------------------------------- |
| SGD, `counts`: the step on the graph       | before | 10 dispatches in 5 submissions           |
| SGD, `counts`: the step on the graph       | after  | 10 dispatches in 1 submission            |
| Adam, `step`: the whole step, batch 64     | before | 15 dispatches in 7 submissions (one run) |
| Adam, `step`: the whole step, batch 64     | after  | 19 dispatches in 3 submissions (one run) |

The times are medians of 20 steps after 5 warm-up steps, in milliseconds. The rounds are in the order
they ran.

| Scenario                                                      | Build  | Rounds              | Median |
| ------------------------------------------------------------- | ------ | ------------------- | -----: |
| `optimizer.step()` alone, the queue finished inside the clock | before | 126.3, 111.9, 107.8 |  111.9 |
| `optimizer.step()` alone, the queue finished inside the clock | after  | 126.4, 139.3, 141.2 |  139.3 |
| whole MNIST step, batch 64, Adam                              | before | 198.5, 234.1, 203.5 |  203.5 |
| whole MNIST step, batch 64, Adam                              | after  | 217.8, 252.0, 215.2 |  217.8 |

On SwiftShader the device-side step is not faster. A software device runs the update kernels on the
CPU, so each kernel costs about what the host loop it replaces cost, and the removed readbacks are
short here. A hardware GPU is not measured in this table. An earlier measurement of `step()` without
the flush read 30 to 40 ms, because the device had not finished the work when the clock stopped. The
flushed rows are the ones above. The per-step traffic that this change removes is measured in the
test `a training step reads back only the loss and uploads no parameter values`, on a fake device.

## Tests

```sh
npm run lint      # Prettier over src, scripts, examples, tests and the documents
npm test          # builds, then runs tests/*.test.mjs on Node
TYPESHADE_CHROMIUM=/path/to/chrome npm run test:gpu   # builds, then runs tests/gpu/*.test.mjs
```

- `numerics.test.mjs` runs every case of `tests/_cases.mjs` on the CPU tier against a plain
  JavaScript f32 model (`tests/_reference.mjs`).
- `plan.test.mjs` asserts the report of a `data()` on a recording stand-in for a WebGPU device
  (`tests/_fake-gpu.mjs`, passed to `nn.device` as `gpuDevice`): the tier, the dispatches, the
  groups and the reason for each boundary.
- `resources.test.mjs` asserts the arrays that a read, a failed plan and a disposal leave behind.
- `graph-cpu.test.mjs`, `api.test.mjs`, `trainer.test.mjs` and `smoke.test.mjs` hold the graph and
  gradient rules, the refusals and their messages, the Trainer, and a first run.
- `mnist.test.mjs` runs the MNIST example on the CPU tier, on synthetic data of the same shape.
- `instrument.test.mjs` plants a fault in a copy of the sources and requires the right test to fail.
- `gpu/gpu.test.mjs` runs on WebGPU in headless Chromium with SwiftShader. It holds the numerics
  cases to the CPU tier and to the reference, the plans to the plans of the stand-in, the report
  counts to the calls the page sees, and the MNIST example to its run on the CPU tier.

The case of 2^24 + 1 elements makes `numerics.test.mjs` use about 2.4 GB of memory. The CPU tier
copies each array argument to a JavaScript array on each call. `TYPENN_FAST=1` leaves that case out.

On SwiftShader, `sin`, `cos`, `tan`, `acos` and the gradients through them differ from the f32
reference by more than 1e-5 relative. Seven cases are listed as expected failures in
`SWIFTSHADER_TRIG`, and the test fails when any other case differs, or when one of the seven agrees.

CI (`.github/workflows/ci.yml`) has two jobs. `cpu` runs `npm run lint` and `npm test` on Node 22.
`gpu` installs Chromium and runs `npm run test:gpu`.

## Publishing

Publishing a GitHub release runs the npm publishing workflow. A manual workflow run defaults to a
dry run; clear `dry_run` to publish. Actual publication requires the `NPM_ACCESS_TOKEN` repository
or organization secret with npm publish access. The workflow validates the packed package before
publishing it publicly with provenance.

## License

Apache-2.0. See [LICENSE](./LICENSE).
