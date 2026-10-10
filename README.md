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

The first training path follows TypeShade's MNIST computation example: one `Linear` classifier,
integer labels, stable softmax cross entropy, an explicit TypeShade backward kernel, and SGD.
The dataset stays in host memory and only the active batch is uploaded. CPU training calls the
CPU functions generated from the same TypeShade kernels.

```ts
const model = new nn.Linear(784, 10, { seed: 123, weightScale: 0.01 });
const trainer = nn.createTrainer(model, {
  device,
  optimizer: nn.optim.SGD({ learningRate: 0.1 }),
});

await trainer.fit(
  { inputs: trainPixels, labels: trainLabels },
  {
    epochs: 5,
    batchSize: 32,
    onEpoch: ({ epoch, loss, accuracy }) => console.log(epoch, loss, accuracy),
  },
);

await trainer.dispose();
```

This initial Trainer only accepts a single `Linear` module and SGD. It uses an explicit
backward kernel like the TypeShade MNIST example; it does not yet build gradients from the
Tensor graph. Model weights are copied back after training so the same `Linear` can be used
for inference.

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
- Intermediate tensors remain resident until disposed or the device is disposed.
- `Module`, `Linear`, and `Sequential` support inference graph composition; `Linear` also
  supports the initial single-layer SGD Trainer.
- Tensor-graph autograd has an initial limited reverse-mode path; Adam, dataset loaders, model
  files, and WebNN are not implemented yet.
  The standalone softmax-cross-entropy operation returns one loss per row; the Trainer's
  explicit backward kernel computes its gradient separately.
- `shadeGrad`/`shadeGradCheck` operate on pure TypeShade functions only; Conv2D and pooling
  Tensor graph gradients remain unimplemented.

TypeShade's call-layer configuration is currently process-global, so TypeNN permits one active
device per JavaScript context. Trainer support is limited to a single `Linear` classifier.

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
