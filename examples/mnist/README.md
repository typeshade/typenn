# MNIST MLP

This example follows the PyTorch training shape: a user-defined `Module`, explicit
`train()`/`eval()` modes, an optimizer, `zeroGrad()`, `backward()`, and `step()`.
The network is `Flatten → Linear(784, 128) → ReLU → Dropout(0.2) → Linear(128, 10)`.
Cross entropy consumes integer class labels and logits, so the model does not add a
softmax layer during training.

The IDX gzip files are downloaded from the Google-hosted MNIST mirror on first use.
Pixels are normalized per batch while the dataset remains in its compact uint8 form.
Only each active batch is copied into a TypeNN Tensor.

`model.ts` is the network, `loop.ts` is the training loop as a function, `train()`, and
`train.ts` is the script that downloads the data and calls it. Run the script in a runtime
that supports `fetch` and `DecompressionStream("gzip")` after building TypeNN (`npm run build`).
It is TypeScript, so `tsx` runs it:

```sh
npx tsx examples/mnist/train.ts
MNIST_DEVICE=cpu MNIST_STEPS=40 MNIST_TEST_SAMPLES=1000 npx tsx examples/mnist/train.ts
```

`MNIST_DEVICE` is `auto` (WebGPU, else the CPU tier), `webgpu`, `webgl2` or `cpu`.
`MNIST_STEPS` stops after that many training steps, and `MNIST_TEST_SAMPLES` evaluates on
that many test images. The tests run `train()` on the CPU tier in Node (`tests/mnist.test.mjs`)
and on a WebGPU device in Chromium (`tests/gpu/gpu.test.mjs`), on synthetic data of the same
shape.

Every `data()` of a step runs as one TypeShade `fuse` scope, so `device.lastReport` says what the
plan of the last one was: the dispatches, the submissions and which kernel functions share a
dispatch. The optimizer keeps Adam moments on the CPU and applies parameter updates through
Tensor's host parameter storage, and reads the gradients of all parameters in one plan.
This path demonstrates the API and is not a throughput-tuned training loop.
