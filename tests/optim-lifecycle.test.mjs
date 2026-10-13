import test, { after } from "node:test";
import assert from "node:assert/strict";
import { fakeGpu } from "./_fake-gpu.mjs";
import { closeNn, loadNn } from "./_nn.mjs";

// The life of the arrays an optimizer step moves: the array a parameter gives up, the moments, the
// failure of a plan, a disposal, and the host copy that a read refreshes. The values come from a
// CPU device. The buffer counts and the plans come from a fake WebGPU device, which reads every
// buffer back as zeros and so proves identity and counts, never a value.

const nn = await loadNn();
after(closeNn);

const count = (shape) => shape.reduce((a, b) => a * b, 1);
const inputs = (samples = 6, features = 4) =>
  Float32Array.from({ length: samples * features }, (_, i) => Math.sin(i + 1));
const labels = [0, 1, 2, 0, 1, 2];

function mlp() {
  return new nn.Sequential(
    new nn.Linear(4, 6, { seed: 3 }),
    new nn.ReLU(),
    new nn.Linear(6, 3, { seed: 4 }),
  );
}

/** Zero the gradients, forward, loss, backward. Returns the loss; the caller steps and disposes. */
async function backward(device, model, optimizer, wait = true) {
  await optimizer.zeroGrad();
  const loss = new nn.CrossEntropyLoss({ reduction: "mean" }).forward(
    model.forward(device.tensor(inputs(), { shape: [6, 4] })),
    labels,
  );
  if (wait) await loss.data();
  await loss.backward();
  return loss;
}

async function train(device, model, optimizer, steps) {
  const parameters = model.parameters();
  for (let step = 0; step < steps; step++) {
    const loss = await backward(device, model, optimizer);
    await optimizer.step();
    await loss.disposeGraph(parameters);
  }
  await optimizer.zeroGrad();
}

const onCpu = (body) => async () => {
  const device = await nn.device({ device: "cpu" });
  try {
    await body(device);
  } finally {
    await device.dispose();
  }
};

/**
 * Run `body` on a fake WebGPU device. `poolBytes` is the device's pool bound.
 */
/**
 * The tests that count the arrays a step frees run with `poolBytes: 0`, which frees each array at
 * once, as before the pool. The pool has its own tests in buffer-pool.test.mjs.
 */
const onFakeGpuUnpooled = (body, options = {}) =>
  onFakeGpu(body, { ...options, poolBytes: 0 });

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

const live = (gpu) => gpu.made.filter((buffer) => !buffer.destroyed).length;
/** The storage buffers alone: the arrays of tensors. The runtime keeps its uniform buffers. */
const liveArrays = (gpu) =>
  gpu.made.filter((buffer) => !buffer.destroyed && (buffer.usage & 128) !== 0)
    .length;

/** Count the `destroy()` calls of every buffer the fake device makes. */
function countDestroys(gpu) {
  const destroys = new Map();
  const create = gpu.device.createBuffer;
  gpu.device.createBuffer = (descriptor) => {
    const buffer = create(descriptor);
    const destroy = buffer.destroy;
    buffer.destroy = () => {
      destroys.set(buffer, (destroys.get(buffer) ?? 0) + 1);
      destroy();
    };
    return buffer;
  };
  return destroys;
}

test("every array an optimizer step replaces is destroyed exactly once", async () => {
  for (const make of [
    (p) => nn.optim.Adam(p, { lr: 0.01 }),
    (p) => nn.optim.SGD(p, { lr: 0.1 }),
  ])
    await onFakeGpuUnpooled(async (device, gpu) => {
      const destroys = countDestroys(gpu);
      const model = mlp().to(device);
      const parameters = model.parameters();
      const optimizer = make(parameters);
      const counts = [];
      for (let step = 0; step < 4; step++) {
        const loss = await backward(device, model, optimizer);
        await optimizer.step();
        await loss.disposeGraph(parameters);
        await optimizer.zeroGrad();
        counts.push(live(gpu));
      }
      assert.equal(new Set(counts).size, 1, `live after each step: ${counts}`);
      await optimizer.dispose();
      await Promise.all(parameters.map((parameter) => parameter.dispose()));
      assert.equal(live(gpu), 0);
      await device.dispose();
      assert.equal(live(gpu), 0);
      assert.ok(destroys.size > 0);
      for (const times of destroys.values()) assert.equal(times, 1);
    });
});

test("a parameter used twice in one graph gets one summed gradient and one update", async () => {
  await onCpu(async (device) => {
    const weight = device.createParameter(
      Float32Array.from({ length: 12 }, (_, i) => Math.cos(i) / 3),
      { shape: [4, 3] },
    );
    const x = device.tensor(inputs(), { shape: [6, 4] });
    const logits = x.matmul(weight).add(x.matmul(weight));
    const loss = new nn.CrossEntropyLoss({ reduction: "mean" }).forward(
      logits,
      labels,
    );
    await loss.backward();
    const before = Float32Array.from(await weight.parameterValues());
    const gradient = Float32Array.from(await weight.grad.data());
    assert.ok(gradient.some((value) => value !== 0));
    const optimizer = nn.optim.SGD([weight], { lr: 0.1 });
    await optimizer.step();
    const after = await weight.parameterValues();
    for (let i = 0; i < 12; i++)
      assert.ok(
        Math.abs(after[i] - (before[i] - 0.1 * gradient[i])) <= 1e-6,
        `element ${i}`,
      );
    // Two entries of one parameter in one optimizer are refused, so one plan never updates it twice.
    assert.throws(
      () => nn.optim.Adam([weight, weight], { lr: 0.1 }),
      RangeError,
    );
  })();
});

test("a step on a disposed parameter is refused and leaves no array behind", async () => {
  await onFakeGpuUnpooled(async (device, gpu) => {
    const model = mlp().to(device);
    const parameters = model.parameters();
    const optimizer = nn.optim.Adam(parameters, { lr: 0.01 });
    const loss = await backward(device, model, optimizer);
    await optimizer.step();
    await loss.disposeGraph(parameters);
    const loss2 = await backward(device, model, optimizer);
    const grads = parameters.map((parameter) => parameter.grad);
    await parameters[1].dispose();
    const before = live(gpu);
    await assert.rejects(() => optimizer.step(), /disposed/);
    assert.equal(live(gpu), before, "the refused step made no array");
    // adopt() on a disposed target frees the array it was given instead of keeping it.
    const target = parameters[1];
    const source = parameters[0].add(parameters[0]);
    await device.computeAll([source]);
    const withSource = live(gpu);
    assert.throws(() => device.adopt(target, source), /disposed/);
    assert.equal(source.storage, undefined);
    assert.equal(live(gpu), withSource - 1);
    assert.equal(target.storage, undefined);
    void loss2;
    void grads;
  });
});

test("device.dispose() with parameters not yet read back frees everything once", async () => {
  await onFakeGpu(async (device, gpu) => {
    const destroys = countDestroys(gpu);
    const model = mlp().to(device);
    const parameters = model.parameters();
    const optimizer = nn.optim.Adam(parameters, { lr: 0.01 });
    await train(device, model, optimizer, 2);
    assert.ok(parameters.every((parameter) => parameter.hostStale));
    await device.dispose();
    assert.equal(live(gpu), 0);
    // The tensors outlive the device: disposing them neither reads nor frees again.
    await Promise.all(parameters.map((parameter) => parameter.dispose()));
    await optimizer.dispose();
    for (const times of destroys.values()) assert.equal(times, 1);
    await assert.rejects(() => parameters[0].parameterValues(), /disposed/);
  });
});

test("a device disposed while a step runs ends the step, not the process", async () => {
  await onFakeGpu(async (device, gpu) => {
    const destroys = countDestroys(gpu);
    const model = mlp().to(device);
    const parameters = model.parameters();
    const optimizer = nn.optim.Adam(parameters, { lr: 0.01 });
    await backward(device, model, optimizer);
    const step = optimizer.step();
    const disposal = device.dispose();
    const [stepped] = await Promise.allSettled([step, disposal]);
    // The step either finished first or was refused as disposed; nothing else.
    if (stepped.status === "rejected")
      assert.match(String(stepped.reason), /disposed/);
    assert.equal(live(gpu), 0);
    for (const times of destroys.values()) assert.equal(times, 1);
  });
});

test("the host copy refreshed after several steps is a fresh read-back of the device array", async () => {
  await onCpu(async (device) => {
    const model = mlp().to(device);
    const [first] = model.modules().values();
    const initial = Float32Array.from(first.weight);
    const optimizer = nn.optim.Adam(model.parameters(), { lr: 0.01 });
    await train(device, model, optimizer, 5);
    // The layer's array lags: the update is on the device only.
    assert.deepEqual(Array.from(first.weight), Array.from(initial));
    const parameters = model.parameters();
    assert.ok(parameters.every((parameter) => parameter.hostStale));
    const fresh = await Promise.all(
      parameters.map((parameter) => parameter.storage.read()),
    );
    await model.sync();
    assert.ok(parameters.every((parameter) => !parameter.hostStale));
    assert.notDeepEqual(Array.from(first.weight), Array.from(initial));
    const layers = [...model.modules().values()].filter(
      (layer) => layer instanceof nn.Linear,
    );
    const hosts = layers.flatMap((layer) => [layer.weight, layer.bias]);
    hosts.forEach((host, index) =>
      assert.deepEqual(Array.from(host), Array.from(fresh[index])),
    );
    // A second sync has nothing to read; a read of a refreshed parameter does not touch the device.
    const before = device.reports.length;
    await model.sync();
    await parameters[0].parameterValues();
    assert.equal(device.reports.length, before);
  })();
});

test("Module.sync() reads every stale parameter of a model in one pass, and Conv2d is covered", async () => {
  await onCpu(async (device) => {
    const model = new nn.Sequential(
      new nn.Conv2d(1, 2, { kernelSize: 3, seed: 5 }),
      new nn.Flatten(),
      new nn.Linear(18, 3, { seed: 6 }),
    ).to(device);
    const [conv] = model.modules().values();
    const initial = Float32Array.from(conv.weight);
    const optimizer = nn.optim.SGD(model.parameters(), { lr: 0.1 });
    const parameters = model.parameters();
    for (let step = 0; step < 2; step++) {
      await optimizer.zeroGrad();
      const loss = new nn.CrossEntropyLoss({ reduction: "mean" }).forward(
        model.forward(device.tensor(inputs(6, 25), { shape: [6, 5, 5, 1] })),
        labels,
      );
      await loss.backward();
      await optimizer.step();
      await loss.disposeGraph(parameters);
    }
    assert.deepEqual(Array.from(conv.weight), Array.from(initial));
    await model.sync();
    assert.notDeepEqual(Array.from(conv.weight), Array.from(initial));
    assert.deepEqual(
      Array.from(conv.weight),
      Array.from(await parameters[0].storage.read()),
    );
    // A model that was never moved to a device has nothing to sync.
    await new nn.Sequential(new nn.Linear(2, 2)).sync();
  })();
});

test("assign() on a stale parameter replaces the device value, and the next step starts from it", async () => {
  await onCpu(async (device) => {
    const model = mlp().to(device);
    const parameters = model.parameters();
    const optimizer = nn.optim.SGD(parameters, { lr: 0.1 });
    await train(device, model, optimizer, 3);
    const target = parameters[0];
    assert.ok(target.hostStale);
    const replacement = Float32Array.from(
      { length: count(target.shape) },
      (_, i) => (i % 7) / 10 - 0.3,
    );
    await target.assign(replacement);
    assert.ok(!target.hostStale);
    assert.deepEqual(
      Array.from(await target.storage.read()),
      Array.from(replacement),
    );
    const loss = await backward(device, model, optimizer);
    const gradient = Float32Array.from(await target.grad.data());
    await optimizer.step();
    const after = await target.parameterValues();
    for (let i = 0; i < replacement.length; i++)
      assert.ok(
        Math.abs(after[i] - (replacement[i] - 0.1 * gradient[i])) <= 1e-6,
        `element ${i}`,
      );
    await loss.disposeGraph(parameters);
  })();
});

test("a plan that fails leaves the parameters and the moments as they were, and the step runs again", async () => {
  // Find which popErrorScope call is the plan of the second step: one dry run counts them.
  let stepPop = 0;
  await onFakeGpuUnpooled(async (device, gpu) => {
    let pops = 0;
    const pop = gpu.device.popErrorScope;
    gpu.device.popErrorScope = async () => (pops++, await pop());
    const model = mlp().to(device);
    const parameters = model.parameters();
    const optimizer = nn.optim.Adam(parameters, { lr: 0.01 });
    const loss = await backward(device, model, optimizer);
    await optimizer.step();
    await loss.disposeGraph(parameters);
    const second = await backward(device, model, optimizer);
    const base = pops;
    await optimizer.step();
    assert.ok(pops > base);
    stepPop = base + 1;
    await second.disposeGraph(parameters);
  });
  await onFakeGpuUnpooled(
    async (device, gpu) => {
      const model = mlp().to(device);
      const parameters = model.parameters();
      const optimizer = nn.optim.Adam(parameters, { lr: 0.01 });
      const clean = async (loss) => {
        await loss.disposeGraph(parameters);
        await optimizer.zeroGrad();
      };
      await clean(
        await backward(device, model, optimizer).then(async (loss) => {
          await optimizer.step();
          return loss;
        }),
      );
      const settled = liveArrays(gpu);
      const second = await backward(device, model, optimizer);
      const arrays = parameters.map((parameter) => parameter.storage);
      const state = (parameter) => {
        const { first, second } = optimizer.moments.get(parameter);
        return [first, first.storage, second, second.storage];
      };
      const moments = parameters.map(state);
      const iteration = optimizer.iteration;
      const stale = parameters.map((parameter) => parameter.hostStale);
      await assert.rejects(() => optimizer.step(), /planted validation error/);
      parameters.forEach((parameter, index) => {
        assert.equal(parameter.storage, arrays[index], "parameter array");
        assert.equal(parameter.hostStale, stale[index]);
        assert.deepEqual(state(parameter), moments[index]);
      });
      assert.equal(optimizer.iteration, iteration, "the step count");
      // The failed plan keeps no array: once its graph is disposed, the count is the settled one.
      // (The runtime keeps the uniform buffers of a failed plan until the device is disposed, as it
      // does for a failed gradient read; they are not arrays of tensors.)
      await clean(second);
      assert.equal(liveArrays(gpu), settled, "arrays after the failed step");
      // The same step runs again on a new backward pass.
      const third = await backward(device, model, optimizer);
      await optimizer.step();
      assert.equal(optimizer.iteration, iteration + 1);
      await clean(third);
      assert.equal(liveArrays(gpu), settled, "arrays after the repeated step");
    },
    { validation: "planted validation error", validationAt: stepPop },
  );
});

test("a failed Trainer.fit() still leaves the steps it made in the layer", async () => {
  await onCpu(async (device) => {
    const model = new nn.Linear(4, 3, { seed: 2 });
    const initial = Float32Array.from(model.weight);
    const trainer = nn.createTrainer(model, {
      device,
      optimizer: { kind: "sgd", learningRate: 0.5 },
    });
    const dataset = { inputs: inputs(6, 4), labels: Uint32Array.from(labels) };
    let epochs = 0;
    await assert.rejects(
      () =>
        trainer.fit(dataset, {
          epochs: 3,
          batchSize: 3,
          onEpoch: () => {
            if (++epochs === 2) throw new Error("stop after two epochs");
          },
        }),
      /stop after two epochs/,
    );
    assert.notDeepEqual(Array.from(model.weight), Array.from(initial));
    await trainer.dispose();
  })();
});

test("the first Adam step starts the moments at zero", async () => {
  await onCpu(async (device) => {
    const model = mlp().to(device);
    const parameters = model.parameters();
    const optimizer = nn.optim.Adam(parameters, { lr: 0.01 });
    const loss = await backward(device, model, optimizer);
    const gradients = await Promise.all(parameters.map((p) => p.grad.data()));
    const before = parameters.map((p) => Float32Array.from(p.node.value));
    // No moment exists before the first step.
    assert.equal(optimizer.moments.size, 0);
    await optimizer.step();
    for (const [index, parameter] of parameters.entries()) {
      const state = optimizer.moments.get(parameter);
      const first = await state.first.storage.read();
      const second = await state.second.storage.read();
      const after = await parameter.parameterValues();
      for (let i = 0; i < first.length; i++) {
        const g = gradients[index][i];
        assert.ok(Math.abs(first[i] - 0.1 * g) <= 1e-6 * (1 + Math.abs(g)));
        assert.ok(Math.abs(second[i] - 0.001 * g * g) <= 1e-6 * (1 + g * g));
        // m/c1 = g and sqrt(v/c2) = |g|, so the update is lr * g / (|g| + eps).
        if (Math.abs(g) > 1e-4)
          assert.ok(
            Math.abs(
              after[i] - (before[index][i] - (0.01 * g) / (Math.abs(g) + 1e-8)),
            ) <= 1e-6,
            `parameter ${index} element ${i}`,
          );
      }
    }
    await loss.disposeGraph(parameters);
  })();
});

test("the moments are freed by optimizer.dispose(), and two optimizers keep their own", async () => {
  await onFakeGpuUnpooled(async (device, gpu) => {
    const model = mlp().to(device);
    const [layerA, , layerB] = model.modules().values();
    const parameters = model.parameters();
    const setA = parameters.slice(0, 2);
    const setB = parameters.slice(2);
    const adamA = nn.optim.Adam(setA, { lr: 0.01 });
    const adamB = nn.optim.Adam(setB, { lr: 0.01 });
    const stepBoth = async () => {
      const loss = await backward(device, model, adamA);
      await adamA.step();
      await adamB.step();
      await loss.disposeGraph(parameters);
      // Not adamA.zeroGrad(): it retains only its own set, and the gradient graph of that set
      // reaches the parameters of the other (see the finding on Optimizer.zeroGrad()).
      for (const parameter of parameters) {
        await parameter.grad?.disposeGraph(parameters);
        parameter.grad = undefined;
      }
    };
    await stepBoth();
    const both = liveArrays(gpu);
    assert.equal(adamA.moments.size, 2);
    assert.equal(adamB.moments.size, 2);
    await stepBoth();
    assert.equal(liveArrays(gpu), both, "no growth over steps");
    // The first optimizer frees its own moments and no one else's.
    await adamA.dispose();
    assert.equal(liveArrays(gpu), both - 2 * setA.length);
    assert.equal(adamB.moments.size, 2);
    assert.ok(setA.every((parameter) => parameter.storage !== undefined));
    await adamA.dispose();
    assert.equal(
      liveArrays(gpu),
      both - 2 * setA.length,
      "dispose twice is one dispose",
    );
    await adamB.dispose();
    assert.equal(liveArrays(gpu), both - 2 * parameters.length);
    // Disposing the parameters afterwards frees theirs; no moment is left to leak.
    await Promise.all(parameters.map((parameter) => parameter.dispose()));
    assert.equal(live(gpu), 0);
    void layerA;
    void layerB;
  });
});

test("two optimizers over different parameter sets update only their own", async () => {
  await onCpu(async (device) => {
    const model = mlp().to(device);
    const parameters = model.parameters();
    const setA = parameters.slice(0, 2);
    const setB = parameters.slice(2);
    const adam = nn.optim.Adam(setA, { lr: 0.01 });
    const sgd = nn.optim.SGD(setB, { lr: 0.1 });
    const loss = await backward(device, model, adam);
    const read = async (set) =>
      Promise.all(
        set.map(async (p) => Float32Array.from(await p.parameterValues())),
      );
    const [beforeA, beforeB] = [await read(setA), await read(setB)];
    const gradientsB = await Promise.all(setB.map((p) => p.grad.data()));
    await adam.step();
    assert.deepEqual(
      await read(setB),
      beforeB,
      "SGD's set is untouched by Adam",
    );
    const afterA = await read(setA);
    assert.ok(
      afterA.some((values, index) =>
        values.some((v, i) => v !== beforeA[index][i]),
      ),
    );
    await sgd.step();
    assert.deepEqual(
      await read(setA),
      afterA,
      "Adam's set is untouched by SGD",
    );
    const afterB = await read(setB);
    afterB.forEach((values, index) =>
      values.forEach((value, i) =>
        assert.ok(
          Math.abs(value - (beforeB[index][i] - 0.1 * gradientsB[index][i])) <=
            1e-6,
        ),
      ),
    );
    await loss.disposeGraph(parameters);
  })();
});

test("an Adam step binds no more storage buffers than the device allows", async () => {
  for (const limit of [8, 5, 4]) {
    await onFakeGpu(
      async (device, gpu) => {
        const model = mlp().to(device);
        const parameters = model.parameters();
        const optimizer = nn.optim.Adam(parameters, { lr: 0.01 });
        const loss = await backward(device, model, optimizer);
        const first = gpu.shaders.length;
        await optimizer.step();
        const report = device.lastReport;
        const bound = gpu.shaders
          .slice(first)
          .map((code) => (code.match(/var<storage/g) ?? []).length);
        assert.ok(bound.length > 0);
        assert.ok(
          Math.max(...bound) <= limit,
          `limit ${limit}: shaders bind ${bound}`,
        );
        // adamUpdate alone binds out, param, m and v: four.
        const update = report.groups.filter((group) =>
          group.calls.includes("adamUpdate"),
        );
        assert.ok(update.length > 0);
        // The whole step is one submission, at any limit.
        assert.equal(report.submissions, 1);
        console.log(
          `limit ${limit}: groups ${report.groups.length}, dispatches ${report.dispatches}, submissions ${report.submissions}, storage buffers per shader ${bound}`,
        );
        await loss.disposeGraph(parameters);
      },
      { storageBuffers: limit },
    );
  }
});
