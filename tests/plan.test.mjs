import test, { after } from "node:test";
import assert from "node:assert/strict";
import { fakeGpu } from "./_fake-gpu.mjs";
import { closeNn, loadNn } from "./_nn.mjs";

// The plan of a `data()`: what TypeNN gives the compiler's fuse scope, and what the scope makes of
// it, on a recording stand-in for a WebGPU device (`_fake-gpu.mjs`). The device reads every buffer
// back as zeros, so these tests hold the report and the commands the device is given, never a
// value. `numerics.test.mjs` holds the values; `gpu/` holds both on a real device.
//
// A boundary between two groups is decided by a fact of the compiler (TypeShade Rule 11.13), and
// each test that expects one names it: F3, a call reads at another index than `i` what the call
// before it wrote; F5, one dispatch binds at most the storage buffers the device allows (8).

const nn = await loadNn();
after(closeNn);

/** Run `body` on a TypeNN device over a fake WebGPU device, and dispose of both. */
async function onFakeGpu(body, options = {}) {
  const gpu = fakeGpu(options);
  const device = await nn.device({ device: "webgpu", gpuDevice: gpu.device });
  try {
    await body(device, gpu);
  } finally {
    await device.dispose();
  }
}

const vector = (device, values) =>
  device.tensor(values, { shape: [values.length] });

test("a.add(b).mul(c).relu() is one dispatch, one submission, one fused group", async () => {
  await onFakeGpu(async (device, gpu) => {
    assert.deepEqual(device.backend, { provider: "typeshade", name: "webgpu" });
    const a = vector(device, [1, 2, -3, 4]);
    const b = vector(device, [5, -3, 2, -8]);
    const c = vector(device, [2, 1, -2, 0.5]);
    const y = a.add(b).mul(c).relu();
    gpu.mark();
    await y.data();
    assert.deepEqual(device.lastReport, {
      tier: "webgpu",
      dispatches: 1,
      submissions: 1,
      groups: [{ calls: ["add", "mul", "relu"], fused: true }],
    });
    // The device is given what the report counts: one dispatch and the plan's one submission,
    // and then the submission of the read-back of `y`.
    assert.deepEqual(gpu.flow, ["dispatch 1 1 1", "submit", "submit"]);
  });
});

test("a broadcast operand that is read at the loop's index keeps the chain in one dispatch", async () => {
  await onFakeGpu(async (device) => {
    const x = device.tensor([1, 2, 3, 4, 5, 6], { shape: [2, 3] });
    const bias = vector(device, [10, 20, 30]);
    await x.mul(x).add(bias).relu().data();
    // `x.mul(x)` reads one tensor twice, so its second read is a copy that joins the dispatch.
    // `addExpand` reads the product at `i` and the bias at `i % 3`: only the written array
    // decides (F3).
    assert.deepEqual(device.lastReport, {
      tier: "webgpu",
      dispatches: 1,
      submissions: 1,
      groups: [{ calls: ["copy", "mul", "addExpand", "relu"], fused: true }],
    });
  });
});

test("a broadcast operand that holds the other operand's place breaks the chain, and the report says where", async () => {
  await onFakeGpu(async (device) => {
    const x = device.tensor([1, 2, 3, 4, 5, 6], { shape: [2, 3] });
    const bias = vector(device, [10, 20, 30]);
    // The bias is the first operand: `addStrided` reads the product `x * x` through strides, and
    // not at `i` (F3).
    await bias.add(x.mul(x)).data();
    const report = device.lastReport;
    assert.equal(report.dispatches, 2);
    assert.deepEqual(
      report.groups.map((group) => group.calls),
      [["copy", "mul"], ["addStrided"]],
    );
    assert.match(
      report.groups[1].why,
      /^mul and addStrided run as two dispatches because addStrided reads `b\[operandIndex\(i, rank, dims, bStrides\)\]` \(line \d+\)$/,
    );
  });
});

test("a dense layer, matmul then bias then relu, is one dispatch", async () => {
  await onFakeGpu(async (device) => {
    const x = device.tensor(new Float32Array(64 * 784), { shape: [64, 784] });
    const w = device.tensor(new Float32Array(784 * 128), { shape: [784, 128] });
    const bias = vector(device, new Float32Array(128));
    await x.matmul(w).add(bias).relu().data();
    // `matmul` reads its inputs at any index, which is no fact against a fuse: only an array
    // that one call wrote and a later call reads must be read at `i` (F3). The output of `matmul`
    // is written at `i` and read at `i` by `addExpand`, and the bias is only read.
    assert.deepEqual(device.lastReport, {
      tier: "webgpu",
      dispatches: 1,
      submissions: 1,
      groups: [{ calls: ["matmul", "addExpand", "relu"], fused: true }],
    });
  });
});

test("a reduction runs over fewer elements than the call before it, which is a dispatch of its own (F2)", async () => {
  await onFakeGpu(async (device) => {
    const x = device.tensor(new Float32Array(6 * 4), { shape: [6, 4] });
    await x.relu().sum(1).data();
    assert.equal(device.lastReport.dispatches, 2);
    assert.deepEqual(device.lastReport.groups, [
      { calls: ["relu"], fused: false },
      {
        calls: ["sumAxis"],
        fused: false,
        why: "relu and sumAxis run as two dispatches because the trip counts differ (24 and 6)",
      },
    ]);
  });
});

test("a matmul reads the array before it through an index, so it stands apart from the call that wrote it (F3)", async () => {
  await onFakeGpu(async (device) => {
    const x = device.tensor(new Float32Array(6 * 4), { shape: [6, 4] });
    const w = device.tensor(new Float32Array(4 * 4), { shape: [4, 4] });
    // The same trip count, 24: the output of `relu` is read as `a[ai]`, and not as `a[i]`.
    await x.relu().matmul(w).data();
    const report = device.lastReport;
    assert.equal(report.dispatches, 2);
    assert.deepEqual(
      report.groups.map((group) => group.calls),
      [["relu"], ["matmul"]],
    );
    assert.match(
      report.groups[1].why,
      /^relu and matmul run as two dispatches because matmul reads `a\[ai\]` \(line \d+\)$/,
    );
  });
});

/** The groups of a report with the line numbers left out of the reasons. */
const outline = (report) =>
  report.groups.map(({ calls, fused, why }) => ({
    calls,
    fused,
    ...(why === undefined
      ? {}
      : { why: why.replace(/\(line \d+\)/, "(line N)") }),
  }));

/** The model of `examples/mnist`: Flatten, Linear 784 to 128, ReLU, Dropout 0.2, Linear 128 to 10. */
function mnistStep(device) {
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

test("the forward pass of the MNIST model is five dispatches in one submission", async () => {
  await onFakeGpu(async (device) => {
    const { loss } = mnistStep(device);
    await loss.data();
    // Where the number of elements changes, the trip counts differ and the next call stands apart
    // (F2): 50,176 for the flatten (a copy), 8,192 for the first layer and the dropout, 640 for the
    // second layer, 64 for the loss of each row, 1 for their mean. The layers themselves fuse:
    // `matmul`, the bias, `relu` and the dropout mask are one dispatch.
    assert.equal(device.lastReport.dispatches, 5);
    assert.equal(device.lastReport.submissions, 1);
    assert.deepEqual(outline(device.lastReport), [
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
    ]);
  });
});

test("the backward pass of the MNIST model is ten dispatches in four plans, one for each gradient", async () => {
  await onFakeGpu(async (device) => {
    const { loss, parameters } = mnistStep(device);
    await loss.data();
    await loss.backward();
    const before = device.reports.length;
    // The optimizers read the gradient of each parameter with its own `data()`.
    for (const parameter of parameters) await parameter.grad.data();
    const plans = device.reports.slice(before);
    assert.deepEqual(
      plans.map((plan) => [plan.dispatches, plan.submissions]),
      [
        [7, 1],
        [1, 1],
        [1, 1],
        [1, 1],
      ],
    );
    // The first gradient (of the first layer's weights) computes the whole graph. A call after a
    // `binaryGrad` or a `matmul` stands apart: `binaryGrad` reads `dy` through an index (F3) and
    // the counts change from layer to layer (F2). `reluGrad` joins the bias gradient of the layer
    // it follows. The three gradients that follow find their inputs computed.
    assert.deepEqual(outline(plans[0]), [
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
    ]);
    assert.deepEqual(
      plans.slice(1).map((plan) => plan.groups.map((group) => group.calls)),
      [[["binaryGrad"]], [["matmul"]], [["binaryGrad"]]],
    );
  });
});

test("a diamond computes its shared node once, in one dispatch", async () => {
  await onFakeGpu(async (device) => {
    const a = vector(device, [2, 1, 3]);
    const b = vector(device, [3, -4, 2]);
    const shared = a.add(b);
    const out = shared.mul(a).add(shared.mul(b)).relu();
    await out.data();
    // Five calls and not seven: `shared` is one call. Seven arrays are bound, under the device's
    // eight (F5): the two leaves, the four intermediates and the result.
    assert.deepEqual(device.lastReport.groups, [
      { calls: ["add", "mul", "mul", "add", "relu"], fused: true },
    ]);
    assert.equal(device.lastReport.dispatches, 1);
  });
});

test("a tensor that reads one tensor twice reads a copy of it", async () => {
  await onFakeGpu(async (device) => {
    const a = vector(device, [2, -3, 0, 4]);
    await a.add(a).data();
    // One array must not be two parameters of one call, so the second read is a copy, and the
    // copy and the add are one dispatch.
    assert.deepEqual(device.lastReport.groups, [
      { calls: ["copy", "add"], fused: true },
    ]);
    assert.equal(device.lastReport.dispatches, 1);
  });
});

test("a later data() of an intermediate dispatches nothing, and reads its array", async () => {
  await onFakeGpu(async (device, gpu) => {
    const a = vector(device, [1, 2, 3]);
    const b = vector(device, [4, 5, 6]);
    const sum = a.add(b);
    await sum.mul(a).relu().data();
    const plans = device.reports.length;
    gpu.mark();
    await sum.data();
    assert.equal(device.reports.length, plans);
    // The one submission is the read-back.
    assert.deepEqual(gpu.flow, ["submit"]);
    // A new graph over `sum` is a plan of its own that takes `sum` as a leaf.
    gpu.mark();
    await sum.relu().data();
    assert.equal(device.reports.length, plans + 1);
    assert.deepEqual(device.lastReport.groups, [
      { calls: ["relu"], fused: false },
    ]);
  });
});

test("a chain longer than the device's storage buffers is cut where it stops fitting", async () => {
  await onFakeGpu(
    async (device) => {
      const [a, b, c, d, e, f] = [1, 2, 3, 4, 5, 6].map((x) =>
        vector(device, [x, x, x]),
      );
      await a.add(b).add(c).add(d).add(e).add(f).relu().data();
      // Every tensor TypeNN computes has an array of its own, which a later `data()` and the
      // gradient graph read, so none is a local of the entry (F5 counts each as a buffer). Three
      // calls bind a, b, c, d and their three results: 7 arrays, and the fourth would make 9.
      assert.deepEqual(device.lastReport.groups, [
        { calls: ["add", "add", "add"], fused: true },
        {
          calls: ["add", "add", "relu"],
          fused: true,
          why: "add and add run as two dispatches because one dispatch would bind 9 storage buffers, and the device allows 8",
        },
      ]);
      assert.equal(device.lastReport.dispatches, 2);
      assert.equal(device.lastReport.submissions, 1);
    },
    { storageBuffers: 8 },
  );
});

test("the reports keep the last 256 plans, oldest first", async () => {
  await onFakeGpu(async (device) => {
    const a = vector(device, [1, 2]);
    for (let i = 0; i < 300; i++) await a.relu().data();
    assert.equal(device.reports.length, 256);
    assert.equal(device.reports.at(-1), device.lastReport);
    device.clearReports();
    assert.deepEqual(device.reports, []);
    assert.equal(device.lastReport, undefined);
  });
});

test("on the CPU tier the plan has no dispatch and no submission", async () => {
  const device = await nn.device({ device: "cpu" });
  try {
    const a = vector(device, [1, 2, -3, 4]);
    const b = vector(device, [5, -3, 2, -8]);
    await a.add(b).relu().data();
    // The CPU tier never fuses, and says nothing of why (TypeShade Rule 11.13).
    assert.deepEqual(device.lastReport, {
      tier: "cpu",
      dispatches: 0,
      submissions: 0,
      groups: [
        { calls: ["add"], fused: false },
        { calls: ["relu"], fused: false },
      ],
    });
  } finally {
    await device.dispose();
  }
});
