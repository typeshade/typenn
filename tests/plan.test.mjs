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
