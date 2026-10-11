# MNIST MLP

This example follows the PyTorch training shape: a user-defined `Module`, explicit
`train()`/`eval()` modes, an optimizer, `zeroGrad()`, `backward()`, and `step()`.
The network is `Flatten → Linear(784, 128) → ReLU → Dropout(0.2) → Linear(128, 10)`.
Cross entropy consumes integer class labels and logits, so the model does not add a
softmax layer during training.

The IDX gzip files are downloaded from the Google-hosted MNIST mirror on first use.
Pixels are normalized per batch while the dataset remains in its compact uint8 form.
Only each active batch is copied into a TypeNN Tensor.

Run this TypeScript example in a runtime that supports `fetch` and
`DecompressionStream("gzip")` after building TypeNN:

```sh
npx tsx examples/mnist/train.ts
```

The optimizer currently keeps Adam moments on the CPU and applies parameter updates
through Tensor's host parameter storage. Forward and backward Tensor operations use
the selected TypeShade device. This path demonstrates the API and is not yet a
throughput-tuned training loop.
