import test, { after } from "node:test";
import assert from "node:assert/strict";
import { fakeGpu } from "./_fake-gpu.mjs";
import { closeNn, loadNn } from "./_nn.mjs";

// The buffer pool of a device: the arrays a plan outputs are reused from the arrays a released
// Tensor gave back, and a training step reads its loss in the plan of its update. The counters
// come from the fake WebGPU device, which proves the calls and never a value. The values come from
// the CPU tier, where the arrays are real.

const nn = await loadNn();
after(closeNn);

/** The loss of 20 steps of the two models below, as `origin/main` 3a6ba27 computed them on the CPU. */
const RECORDED_MLP = [
  2.3285720348358154, 2.2661049365997314, 2.215897798538208, 2.172762870788574,
  2.0978927612304688, 2.0505495071411133, 1.9738157987594604,
  1.9186973571777344, 1.8722507953643799, 1.811076283454895, 1.7379155158996582,
  1.7072731256484985, 1.6429568529129028, 1.574596881866455, 1.494470477104187,
  1.4696836471557617, 1.407051920890808, 1.3253984451293945, 1.2298675775527954,
  1.1899969577789307,
];
const RECORDED_CNN = [
  2.3246567249298096, 2.264599561691284, 2.2140796184539795, 2.155900716781616,
  2.097506284713745, 2.041372299194336, 1.9855618476867676, 1.928479552268982,
  1.8704766035079956, 1.8126192092895508, 1.755347490310669, 1.6983989477157593,
  1.6415375471115112, 1.5848430395126343, 1.5285425186157227,
  1.4727919101715088, 1.4176115989685059, 1.3630743026733398,
  1.3093175888061523, 1.2564538717269897,
];

const BATCH = 64;

/** The sequence of the test models: pixels and labels from a fixed generator. */
function samples() {
  let state = 7;
  const next = () =>
    (state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 2 ** 32;
  return {
    images: Float32Array.from({ length: BATCH * 784 }, next),
    labels: Array.from({ length: BATCH }, (_, row) => row % 10),
  };
}

/** The model of examples/mnist, with the seeds of the recorded losses. */
function mlp(device) {
  return new nn.Sequential(
    new nn.Flatten(),
    new nn.Linear(784, 128, { seed: 101, weightScale: 1 / Math.sqrt(784) }),
    new nn.ReLU(),
    new nn.Dropout(0.2),
    new nn.Linear(128, 10, { seed: 202, weightScale: 1 / Math.sqrt(128) }),
  ).to(device);
}

/** A small convolutional model: one Conv2d, one max pool, one Linear. */
function cnn(device) {
  return new nn.Sequential(
    new nn.Conv2d(1, 8, { kernelSize: 3, seed: 5 }),
    new nn.ReLU(),
    new nn.MaxPool2d({ filterSize: 2 }),
    new nn.Flatten(),
    new nn.Linear(13 * 13 * 8, 10, { seed: 6 }),
  ).to(device);
}

const MODELS = {
  mlp: { make: mlp, shape: [BATCH, 1, 28, 28], recorded: RECORDED_MLP },
  cnn: { make: cnn, shape: [BATCH, 28, 28, 1], recorded: RECORDED_CNN },
};

/**
 * One model, its Adam optimizer and its samples, stepped one at a time. `step(alsoRead)` is the
 * loop of the README when `alsoRead` is true: the loss is read in the update's plan. Otherwise the
 * loss is read before `backward()`, as the lab did before. Returns the loss of the step.
 */
function session(device, name) {
  const { make, shape } = MODELS[name];
  const { images, labels } = samples();
  const model = make(device);
  const parameters = model.parameters();
  const optimizer = nn.optim.Adam(parameters, { lr: 0.001 });
  const criterion = new nn.CrossEntropyLoss();
  return {
    model,
    parameters,
    optimizer,
    async step(alsoRead) {
      await optimizer.zeroGrad();
      const loss = criterion.forward(
        model.forward(device.tensor(images, { shape })),
        labels,
      );
      let value;
      if (alsoRead) {
        await loss.backward();
        [value] = await optimizer.step({ alsoRead: [loss] });
      } else {
        value = await loss.data();
        await loss.backward();
        await optimizer.step();
      }
      await loss.disposeGraph(parameters);
      return value[0];
    },
  };
}

/** The loss of each of `steps` steps. */
async function train(device, name, { steps, alsoRead }) {
  const run = session(device, name);
  const losses = [];
  for (let step = 0; step < steps; step++)
    losses.push(await run.step(alsoRead));
  await run.optimizer.zeroGrad();
  return losses;
}

/** The counters of the fake device for `run`, and the storage buffers it made. */
async function counted(gpu, run) {
  const before = gpu.stats;
  const made = gpu.made.length;
  const result = await run();
  const after = gpu.stats;
  const delta = Object.fromEntries(
    Object.keys(after).map((key) => [key, after[key] - before[key]]),
  );
  // WebGPU's STORAGE usage, which the specification fixes.
  delta.storageBuffers = gpu.made
    .slice(made)
    .filter((buffer) => (buffer.usage & 128) !== 0).length;
  return { delta, result };
}

async function onFakeGpu(body, options = {}) {
  const { poolBytes, ...gpuOptions } = options;
  const gpu = fakeGpu(gpuOptions);
  const device = await nn.device({
    device: "webgpu",
    gpuDevice: gpu.device,
    ...(poolBytes === undefined ? {} : { poolBytes }),
  });
  try {
    await body(device, gpu);
  } finally {
    await device.dispose();
  }
}

const storageLive = (gpu) =>
  gpu.made.filter((buffer) => !buffer.destroyed && (buffer.usage & 128) !== 0)
    .length;

test("the values of 20 steps are those origin/main computed, bit for bit, with and without alsoRead", async () => {
  for (const name of ["mlp", "cnn"]) {
    for (const alsoRead of [false, true]) {
      const device = await nn.device("cpu");
      try {
        const losses = await train(device, name, { steps: 20, alsoRead });
        assert.deepEqual(
          losses,
          MODELS[name].recorded,
          `${name} alsoRead=${alsoRead}`,
        );
      } finally {
        await device.dispose();
      }
    }
  }
});

for (const name of ["mlp", "cnn"]) {
  for (const alsoRead of [false, true]) {
    const loop = alsoRead ? "alsoRead" : "read before backward";
    test(`${name}, ${loop}: a steady step makes no storage buffer and uploads no zeros`, () =>
      onFakeGpu(async (device, gpu) => {
        const run = session(device, name);
        for (let warm = 0; warm < 3; warm++) await run.step(alsoRead);
        const { delta } = await counted(gpu, () => run.step(alsoRead));
        assert.equal(
          delta.storageBuffers,
          0,
          "storage buffers made by one step",
        );
        // The loss read back in a step of `alsoRead` is a 4-byte array that was read before, so the
        // runtime uploads its host copy again when the pool reuses it. The fake reads zeros back.
        assert.ok(
          delta.zeroUploadBytes <= 4,
          `zero bytes uploaded: ${delta.zeroUploadBytes}`,
        );
        // Every other buffer is the runtime's own: one uniform buffer per dispatch, and one staging
        // buffer per readback.
        assert.equal(
          delta.buffers - delta.storageBuffers,
          delta.dispatches + delta.readbacks,
        );
        assert.equal(delta.dispatches, 19);
        assert.equal(delta.readbacks, 1);
        assert.equal(delta.readBytes, 4);
        // alsoRead: one plan, then the readback. Before: the forward plan, the update plan and the
        // readback.
        assert.equal(delta.submissions, alsoRead ? 2 : 3);
      }));
  }
}

test("the loss of step({ alsoRead }) is computed in the update's plan: a later data() adds no dispatch", () =>
  onFakeGpu(async (device, gpu) => {
    const { images, labels } = samples();
    const model = mlp(device);
    const parameters = model.parameters();
    const optimizer = nn.optim.Adam(parameters, { lr: 0.001 });
    const criterion = new nn.CrossEntropyLoss();
    const once = async () => {
      await optimizer.zeroGrad();
      const loss = criterion.forward(
        model.forward(device.tensor(images, { shape: [BATCH, 1, 28, 28] })),
        labels,
      );
      await loss.backward();
      const {
        delta: step,
        result: [value],
      } = await counted(gpu, () => optimizer.step({ alsoRead: [loss] }));
      const { delta: read, result: again } = await counted(gpu, () =>
        loss.data(),
      );
      await loss.disposeGraph(parameters);
      return { step, read, value: value[0], again: again[0] };
    };
    await once();
    const { step, read, value, again } = await once();
    assert.equal(
      again,
      value,
      "the value read in the plan is the value data() returns",
    );
    // The read computes nothing, and the host copy the step read is current: no dispatch, no
    // submission and no readback.
    assert.equal(read.dispatches, 0);
    assert.equal(read.submissions, 0);
    assert.equal(read.readbacks, 0);
    assert.equal(step.readbacks, 1);
    assert.equal(step.submissions, 2);
    await optimizer.zeroGrad();
  }));

test("without alsoRead, a loss read after step() computes the loss in a plan of its own", () =>
  onFakeGpu(async (device, gpu) => {
    const { images, labels } = samples();
    const model = mlp(device);
    const parameters = model.parameters();
    const optimizer = nn.optim.Adam(parameters, { lr: 0.001 });
    const loss = new nn.CrossEntropyLoss().forward(
      model.forward(device.tensor(images, { shape: [BATCH, 1, 28, 28] })),
      labels,
    );
    await loss.backward();
    await optimizer.step();
    const { delta } = await counted(gpu, () => loss.data());
    // The update's plan does not compute the loss: its gradients need the logits, not the loss.
    assert.equal(delta.dispatches, 2);
    assert.equal(delta.submissions, 2);
    await loss.disposeGraph(parameters);
    await optimizer.zeroGrad();
  }));

test("step() without alsoRead reads nothing back, and returns no values", () =>
  onFakeGpu(async (device, gpu) => {
    const run = session(device, "mlp");
    await run.step(false);
    const { delta, result } = await counted(gpu, async () => {
      await run.optimizer.zeroGrad();
      const loss = new nn.CrossEntropyLoss().forward(
        run.model.forward(
          device.tensor(samples().images, { shape: [BATCH, 1, 28, 28] }),
        ),
        samples().labels,
      );
      await loss.backward();
      const values = await run.optimizer.step();
      await loss.disposeGraph(run.parameters);
      return values;
    });
    assert.deepEqual(result, []);
    assert.equal(delta.readbacks, 0);
    assert.equal(delta.readBytes, 0);
  }));

test("alsoRead returns the values before the update, and refuses a disposed or foreign tensor", async () => {
  const device = await nn.device("cpu");
  try {
    const run = session(device, "mlp");
    await run.step(true);
    const weight = run.parameters[0];
    const before = Array.from(await weight.data());
    const { images, labels } = samples();
    const loss = new nn.CrossEntropyLoss().forward(
      run.model.forward(device.tensor(images, { shape: [BATCH, 1, 28, 28] })),
      labels,
    );
    await loss.backward();
    const [weightValues] = await run.optimizer.step({ alsoRead: [weight] });
    assert.deepEqual(
      Array.from(weightValues),
      before,
      "the parameter before the update",
    );
    assert.notDeepEqual(
      Array.from(await weight.data()),
      before,
      "the update did change it",
    );
    await loss.disposeGraph(run.parameters);

    const gone = device.tensor([1, 2], { shape: [2] });
    await gone.dispose();
    const iteration = run.optimizer.iteration;
    await assert.rejects(
      () => run.optimizer.step({ alsoRead: [gone] }),
      /disposed/,
    );
    assert.equal(
      run.optimizer.iteration,
      iteration,
      "a refused step is not counted",
    );
    await run.optimizer.zeroGrad();
  } finally {
    await device.dispose();
  }
});

test("a failed plan with alsoRead leaves the parameters as they were", () =>
  onFakeGpu(async (device, gpu) => {
    // The next error scope that is popped reports a validation error: the step's plan fails.
    let failNext = false;
    const pop = gpu.device.popErrorScope;
    gpu.device.popErrorScope = async () => {
      if (!failNext) return pop();
      failNext = false;
      return { message: "planted validation error" };
    };
    const run = session(device, "mlp");
    await run.step(false);
    const { images, labels } = samples();
    const loss = new nn.CrossEntropyLoss().forward(
      run.model.forward(device.tensor(images, { shape: [BATCH, 1, 28, 28] })),
      labels,
    );
    await loss.backward();
    const arrays = run.parameters.map((parameter) => parameter.storage);
    const iteration = run.optimizer.iteration;
    failNext = true;
    await assert.rejects(
      () => run.optimizer.step({ alsoRead: [loss] }),
      /planted validation error/,
    );
    run.parameters.forEach((parameter, index) =>
      assert.equal(parameter.storage, arrays[index], "parameter array"),
    );
    assert.equal(run.optimizer.iteration, iteration);
    await loss.disposeGraph(run.parameters);
    await run.optimizer.zeroGrad();
    // The same step runs again.
    assert.ok(Number.isFinite(await run.step(true)));
  }));

test("a tensor kept across steps is never overwritten by a reused array", async () => {
  const device = await nn.device("cpu");
  try {
    const a = device.tensor([1, 2, 3, 4], { shape: [4] });
    const b = device.tensor([10, 20, 30, 40], { shape: [4] });
    const kept = a.mul(b).add(a);
    await kept.data();
    for (let round = 0; round < 12; round++) {
      const scratch = a.add(b).mul(a).relu().sub(b);
      await scratch.data();
      await scratch.disposeGraph([a, b]);
      const churn = device
        .tensor([round, round, round, round], { shape: [4] })
        .neg();
      await churn.data();
      await churn.disposeGraph();
    }
    assert.deepEqual(Array.from(await kept.data()), [11, 42, 93, 164]);
    assert.deepEqual(Array.from(await a.data()), [1, 2, 3, 4]);
    assert.deepEqual(Array.from(await b.data()), [10, 20, 30, 40]);
  } finally {
    await device.dispose();
  }
});

test("a scratch copy (a.add(a)) reads the array it copies, after the pool has reused arrays", async () => {
  const device = await nn.device("cpu");
  try {
    for (let round = 0; round < 4; round++) {
      const churn = device
        .tensor([round, 1, 2, 3, 4, 5, 6, 7], { shape: [8] })
        .relu();
      await churn.data();
      await churn.disposeGraph();
    }
    const a = device.tensor([1, 2, 3, 4, 5, 6, 7, 8], { shape: [8] });
    assert.deepEqual(
      Array.from(await a.add(a).data()),
      [2, 4, 6, 8, 10, 12, 14, 16],
    );
    assert.deepEqual(
      Array.from(await a.mul(a).data()),
      [1, 4, 9, 16, 25, 36, 49, 64],
    );
  } finally {
    await device.dispose();
  }
});

test("adopt() moves a computed array into a parameter, and the array it replaces is not handed out again", async () => {
  const device = await nn.device("cpu");
  try {
    const target = device.createParameter(new Float32Array([1, 2, 3, 4]), {
      shape: [4],
    });
    await target.data();
    const source = device
      .tensor([5, 6, 7, 8], { shape: [4] })
      .mul(device.tensor([1, 1, 1, 1], { shape: [4] }));
    await source.data();
    device.adopt(target, source);
    assert.equal(source.storage, undefined);
    assert.deepEqual(Array.from(await target.data()), [5, 6, 7, 8]);
    for (let round = 0; round < 3; round++) {
      const other = device
        .tensor([10, 20, 30, 40], { shape: [4] })
        .add(device.tensor([round, 0, 0, 0], { shape: [4] }));
      await other.data();
      await other.disposeGraph();
    }
    assert.deepEqual(Array.from(await target.data()), [5, 6, 7, 8]);
  } finally {
    await device.dispose();
  }
});

test("a read of a tensor disposed before the read fails, and does not return another tensor's values", async () => {
  const device = await nn.device("cpu");
  try {
    const y = device.tensor([1, 2, 3, 4], { shape: [4] }).relu();
    const pending = y.data();
    await y.dispose();
    const z = device
      .tensor([9, 9, 9, 9], { shape: [4] })
      .mul(device.tensor([2, 2, 2, 2], { shape: [4] }));
    assert.deepEqual(Array.from(await z.data()), [18, 18, 18, 18]);
    await assert.rejects(pending, TypeError);
  } finally {
    await device.dispose();
  }
});

test("the pool holds the bytes it is given, and frees the rest at once", async () => {
  for (const [poolBytes, kept] of [
    [4096, 1],
    [0, 0],
    [undefined, 20],
  ]) {
    await onFakeGpu(
      async (device, gpu) => {
        const inputs = Array.from({ length: 10 }, () =>
          device.tensor(new Float32Array(1024), { shape: [1024] }),
        );
        const outputs = inputs.map((input) => input.relu());
        await device.computeAll(outputs);
        assert.equal(storageLive(gpu), 20);
        await Promise.all(outputs.map((tensor) => tensor.dispose()));
        await Promise.all(inputs.map((tensor) => tensor.dispose()));
        assert.equal(storageLive(gpu), kept, `poolBytes ${poolBytes}`);
      },
      poolBytes === undefined ? {} : { poolBytes },
    );
  }
});

test("device.dispose() frees the pooled arrays, each once", async () => {
  const gpu = fakeGpu();
  const device = await nn.device({ device: "webgpu", gpuDevice: gpu.device });
  const run = session(device, "cnn");
  for (let step = 0; step < 4; step++) await run.step(true);
  assert.ok(storageLive(gpu) > 0, "the pool holds arrays before the dispose");
  await device.dispose();
  assert.equal(storageLive(gpu), 0);
  for (const buffer of gpu.made)
    assert.equal(buffer.destroys, 1, "destroy calls");
});

test("poolBytes is a whole number of bytes, 0 or more", async () => {
  for (const poolBytes of [-1, 1.5, Number.NaN]) {
    await assert.rejects(
      () => nn.device({ device: "cpu", poolBytes }),
      RangeError,
    );
  }
  const device = await nn.device({ device: "cpu", poolBytes: 0 });
  await device.dispose();
});
