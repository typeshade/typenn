// The plans the tests hold TypeNN to, as data: the MNIST model's step, and the groups the fuse
// scope makes of its forward and backward passes. `plan.test.mjs` asserts them on a recording fake
// device, and `gpu/gpu.test.mjs` on a real WebGPU device, so the two cannot part. Pure ES module.

/** The groups of a report with the line numbers left out of the reasons. */
export const outline = (report) =>
  report.groups.map(({ calls, fused, why }) => ({
    calls,
    fused,
    ...(why === undefined
      ? {}
      : { why: why.replace(/\(line \d+\)/, "(line N)") }),
  }));

/** The model of `examples/mnist` on a batch of 64: Flatten, Linear 784 to 128, ReLU, Dropout 0.2,
 *  Linear 128 to 10, and the mean cross entropy. */
export function mnistStep(nn, device) {
  const model = new nn.Sequential(
    new nn.Flatten(),
    new nn.Linear(784, 128, { seed: 101 }),
    new nn.ReLU(),
    new nn.Dropout(0.2),
    new nn.Linear(128, 10, { seed: 202 }),
  ).to(device);
  const images = device.tensor(new Float32Array(64 * 784), {
    shape: [64, 1, 28, 28],
  });
  const labels = Array.from({ length: 64 }, (_, row) => row % 10);
  const loss = new nn.CrossEntropyLoss().forward(model.forward(images), labels);
  return { loss, parameters: model.parameters() };
}

/**
 * The forward pass: five dispatches in one submission. Where the number of elements changes the
 * trip counts differ and the next call stands apart (F2): 50,176 for the flatten (a copy), 8,192
 * for the first layer and the dropout, 640 for the second layer, 64 for the loss of each row, 1 for
 * their mean. The layers themselves fuse: `matmul`, the bias, `relu` and the dropout mask are one
 * dispatch.
 */
export const MNIST_FORWARD = {
  dispatches: 5,
  submissions: 1,
  groups: [
    { calls: ["copy"], fused: false },
    {
      calls: ["matmul", "addExpand", "relu", "mul"],
      fused: true,
      why: "copy and matmul run as two dispatches because the trip counts differ (50176 and 8192)",
    },
    {
      calls: ["matmul", "addExpand"],
      fused: true,
      why: "mul and matmul run as two dispatches because the trip counts differ (8192 and 640)",
    },
    {
      calls: ["softmaxCrossEntropy"],
      fused: false,
      why: "addExpand and softmaxCrossEntropy run as two dispatches because the trip counts differ (640 and 64)",
    },
    {
      calls: ["meanAxis"],
      fused: false,
      why: "softmaxCrossEntropy and meanAxis run as two dispatches because the trip counts differ (64 and 1)",
    },
  ],
};

/**
 * The backward pass: an optimizer reads the gradient of every parameter in one `data()` of its
 * own, so it is one plan. A call after a `binaryGrad` or a `matmul` stands apart: `binaryGrad`
 * reads `dy` through an index (F3) and the counts change from layer to layer (F2). `reluGrad`
 * joins the bias gradient of the layer it follows.
 */
export const MNIST_BACKWARD = {
  dispatches: 10,
  submissions: 1,
  groups: [
    { calls: ["reduceGrad"], fused: false },
    {
      calls: ["crossEntropyLogitsGrad"],
      fused: false,
      why: "reduceGrad and crossEntropyLogitsGrad run as two dispatches because the trip counts differ (64 and 640)",
    },
    {
      calls: ["binaryGrad"],
      fused: false,
      why: "crossEntropyLogitsGrad and binaryGrad run as two dispatches because binaryGrad reads `dy[at]` (line N)",
    },
    {
      calls: ["matmul"],
      fused: false,
      why: "binaryGrad and matmul run as two dispatches because the trip counts differ (640 and 8192)",
    },
    {
      calls: ["binaryGrad", "reluGrad"],
      fused: true,
      why: "matmul and binaryGrad run as two dispatches because binaryGrad reads `dy[at]` (line N)",
    },
    {
      calls: ["binaryGrad"],
      fused: false,
      why: "reluGrad and binaryGrad run as two dispatches because binaryGrad reads `dy[at]` (line N)",
    },
    {
      calls: ["matmul"],
      fused: false,
      why: "binaryGrad and matmul run as two dispatches because the trip counts differ (8192 and 100352)",
    },
    {
      calls: ["binaryGrad"],
      fused: false,
      why: "matmul and binaryGrad run as two dispatches because the trip counts differ (100352 and 128)",
    },
    {
      calls: ["matmul"],
      fused: false,
      why: "binaryGrad and matmul run as two dispatches because the trip counts differ (128 and 1280)",
    },
    {
      calls: ["binaryGrad"],
      fused: false,
      why: "matmul and binaryGrad run as two dispatches because the trip counts differ (1280 and 10)",
    },
  ],
};

/** Run the forward pass of `mnistStep` on `device`, and then the step of an optimizer, which
 *  reads the four gradients. Returns the report of the forward plan and of the plans the step ran. */
export async function mnistPlans(nn, device) {
  const { loss, parameters } = mnistStep(nn, device);
  await loss.data();
  const forward = device.lastReport;
  await loss.backward();
  const before = device.reports.length;
  await new nn.SGD(parameters, { lr: 0.1 }).step();
  return { forward, backward: device.reports.slice(before) };
}

/** A report as `MNIST_FORWARD` and `MNIST_BACKWARD` state a plan. */
export const stated = (report) => ({
  dispatches: report.dispatches,
  submissions: report.submissions,
  groups: outline(report),
});
