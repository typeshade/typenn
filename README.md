# TypeNN

TypeNN is a TypeScript neural-network library built on TypeShade. The first implementation
establishes a deferred Tensor graph and executes its primitive operations through TypeShade
compute kernels and the TypeShade program Runtime.

```ts
import * as nn from "typenn";

const device = await nn.device();
const a = device.tensor([1, 2, 3, 4], { shape: [2, 2] });
const b = device.tensor([5, 6, 7, 8], { shape: [2, 2] });
const y = a.matmul(b).relu();

console.log(await y.data()); // Float32Array [19, 22, 43, 50]
await device.dispose();
```

Operations add graph nodes; `data()` materializes the requested result and reads it back. For a
GPU device, TypeNN records the reachable graph as `Frame.dispatch()` calls and submits one frame,
so a chained graph does not read intermediate values back to the CPU. The
function API follows TensorFlow.js-style names, while Tensor methods are also available:

```ts
const logits = nn.add(nn.matMul(input, weight), bias);
const probabilities = nn.softmax(nn.relu(logits));
console.log(await probabilities.data());
```

The current operation set includes rank-2 `matMul`; same-shape and row-bias binary operations
(`add`, `sub`, `mul`, `div`, `maximum`, `minimum`, `pow`); elementwise arithmetic and activations;
axis and full reductions (`sum`, `mean`, `max`, `min`, `argMax`); and stable rank-2 row-wise
`softmax`, `logSoftmax`, and logits-based softmax cross entropy. Shapes and broadcasting are
intentionally limited and listed in [TODO.md](./TODO.md). `Linear`, `ReLU`, and `Sequential`
compose operations for inference:

```ts
const model = new nn.Sequential(new nn.Linear(784, 10));
const logits = model.forward(device.tensor(pixels, { shape: [1, 784] }));
const probabilities = await logits.softmax().data();
```

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
batch, uses logits directly with sparse-label cross entropy, and evaluates by argmax. Tensor
forward and backward operations use the selected TypeShade device. Adam moment calculation and
parameter updates currently use host arrays, so this example demonstrates the API rather than
optimized training throughput.

The earlier `createTrainer()` convenience path remains limited to a single `Linear` classifier
and SGD. The explicit PyTorch-style loop supports a custom `Module` with multiple Linear layers,
ReLU, Dropout, and the current Tensor-graph gradient rules.

TypeShade owns the shader language and GPU execution runtime. TypeNN's internal `.shade.ts`
kernels are compiled during the library build; applications use TypeNN's Tensor and Module API
and do not author shaders. `device()` reports the selected TypeShade tier through
`device.backend.name`; `auto` tries WebGPU and falls back to the TypeShade CPU tier. WebGL2 is
available as an explicit choice because lowered compute passes can cost more than native GPU
compute. The current build pins TypeShade to a source commit because its npm registry package is
still a placeholder.

For custom functions written as pure TypeShade IR, TypeNN also re-exports TypeShade's
`grad` and `gradCheck` as `shadeGrad` and `shadeGradCheck`. They can generate and check
derivatives of those Shade functions. They do not differentiate TypeNN Tensor graphs; the
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
routes tied maxima to the first element in scan order. They are primitive gradient operations;
they can also be used by the initial reverse-mode graph traversal:

```ts
const [weightGradient] = nn.gradients(loss, [weight]);
console.log(await weightGradient.data());
```

`gradients(output, wrt, upstream?)` returns deferred gradient Tensors. Without an explicit
`upstream`, `output` must contain one value. The first graph rules cover ReLU, broadcast
add/subtract/multiply/divide, unary arithmetic and trigonometric ops, rank-2 MatMul including
transpose options, Conv2D, 2D pooling, `sum`/`mean` reductions, and logits-based softmax cross
entropy. `prod`/`max`/`min` reduction gradients and special activation gradients remain
unsupported.

## Current scope

- `device()` creates a TypeShade Runtime for WebGPU/WebGL2 or selects TypeShade's CPU tier.
- Tensor operation calls construct a lazy dependency graph; the first result read executes it.
- Pure numerical operations use TypeShade kernels for GPU and CPU execution; no separate JS
  production math implementation is maintained.
- Intermediate tensors remain resident until disposed or the device is disposed; use
  `disposeGraph()` to release an operation graph while preserving parameters.
- `Module.to()`, `parameters()`, `train()`, and `eval()` support user-defined models and
  registered child layers. `Flatten`, `Linear`/`Dense`, `ReLU`, and `Dropout` cover the example.
- `Tensor.backward()` accumulates gradients on registered parameter leaves. SGD and Adam
  optimizer objects update host parameter arrays; Adam is not yet a device kernel.
- Tensor-graph autograd remains limited; model serialization and WebNN are not implemented.
  The standalone softmax-cross-entropy operation returns one loss per row; the Trainer's
  explicit backward kernel computes its gradient separately.
- `shadeGrad`/`shadeGradCheck` operate on pure TypeShade functions only; Conv2D and pooling
  Tensor graph gradients remain unimplemented.

TypeShade's call-layer configuration is currently process-global, so TypeNN permits one active
device per JavaScript context. Trainer support is limited to a single `Linear` classifier.

### Elementwise graph fusion

For an eligible same-shape `a.add(b).mul(c).relu()` graph, TypeNN walks the actual
Tensor nodes and their operand order into a TypeShade `ModuleDecl`. That shared
IR produces the CPU implementation through `compileModuleJs` and the GPU pack
through `packModule`; it is not a separately authored fused shader. Structural
program caches reuse code while binding the current graph's input values.

The initial scope is deliberately limited to this three-operation chain with
three distinct input tensors. Broadcasting, computed or aliased operands,
gradient-requiring tensors, materialized intermediates, and externally consumed
intermediates use the existing nodewise execution path. Fusion preserves the
original Tensor DAG for later intermediate reads and autograd. A wrapped chain
is not partially fused: selection applies to the requested output.

The CPU path normally executes emitted JavaScript. If a content security policy
blocks dynamic code generation, it uses the TypeShade interpreter for the same
IR. Other compilation errors are propagated.

`npm test` checks graph-derived IR against independent nodewise TypeNN execution
and the TypeShade interpreter. The production browser harness additionally
checks the packed GPU program and a consumer of the built library; run
`npm run build && npm run build:elementwise-fusion:browser`.
GPU timing includes dispatch, submission, and readback, so one dispatch does not
by itself guarantee a proportional speedup. Dynamic IR compilation also has a
first-use cost and requires TypeShade compiler code in the application bundle.

### Kernel source naming

Each hand-authored TypeShade compute entrypoint has its own `<operation>.shade.ts` file, such as
`add.shade.ts` or `relu.shade.ts`. The runtime registry imports these kernel packs individually.
The TypeShade Vite plugin generates sibling `*.shade.typeshade.ts` files containing executable CPU
implementations and program manifests for TypeScript tooling and the runtime build. These are
generated source views, not declaration files, so they must not be renamed to `.d.ts` or edited
directly. The build removes stale generated views before recompiling current kernels.

## Publishing

Publishing a GitHub release runs the npm publishing workflow. A manual workflow run defaults to a
dry run; clear `dry_run` to publish. Actual publication requires the `NPM_ACCESS_TOKEN` repository
or organization secret with npm publish access. The workflow validates the packed package before
publishing it publicly with provenance.

## License

Apache-2.0. See [LICENSE](./LICENSE).
