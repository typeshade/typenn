import test, { after } from "node:test";
import assert from "node:assert/strict";
import { fakeGpu } from "./_fake-gpu.mjs";
import { closeNn, loadNn } from "./_nn.mjs";

// The optimizers update the parameters and their moments on the device, in one plan. These tests
// hold that to the host algorithm the optimizers used before (copied below, as it was), count the
// traffic of a step on a fake WebGPU device, and check the identity of the parameters.

const nn = await loadNn();
after(closeNn);

/** A deterministic sequence of numbers in [-1, 1). */
function sequence(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 31 - 1;
  };
}

const count = (shape) => shape.reduce((a, b) => a * b, 1);

/** Models of the same shape and initial values: an MLP, and a Conv2d with a classifier. */
function models(kind) {
  if (kind === "mlp")
    return new nn.Sequential(
      new nn.Linear(4, 8, { seed: 7 }),
      new nn.ReLU(),
      new nn.Linear(8, 3, { seed: 8 }),
    );
  return new nn.Sequential(
    new nn.Conv2d(1, 2, { kernelSize: 3, seed: 9 }),
    new nn.ReLU(),
    new nn.Flatten(),
    new nn.Linear(18, 3, { seed: 10 }),
  );
}

function batch(kind) {
  const next = sequence(kind === "mlp" ? 11 : 12);
  const samples = 6;
  const inputShape = kind === "mlp" ? [samples, 4] : [samples, 5, 5, 1];
  const values = Float32Array.from({ length: count(inputShape) }, next);
  const labels = Array.from({ length: samples }, (_, i) => i % 3);
  return { values, inputShape, labels };
}

/** The host Adam update the optimizer used before device kernels, verbatim: Float32 moments. */
function hostAdam(state, { lr, beta1, beta2, epsilon }) {
  state.iteration++;
  const firstCorrection = 1 - beta1 ** state.iteration;
  const secondCorrection = 1 - beta2 ** state.iteration;
  state.values.forEach((updated, index) => {
    const gradient = state.gradients[index];
    const moment = state.moments[index];
    for (let element = 0; element < updated.length; element++) {
      const value = gradient[element];
      moment.first[element] =
        beta1 * moment.first[element] + (1 - beta1) * value;
      moment.second[element] =
        beta2 * moment.second[element] + (1 - beta2) * value * value;
      const first = moment.first[element] / firstCorrection;
      const second = moment.second[element] / secondCorrection;
      updated[element] -= (lr * first) / (Math.sqrt(second) + epsilon);
    }
  });
}

/** The host SGD update the optimizer used before device kernels, verbatim. */
function hostSgd(state, { lr }) {
  state.values.forEach((updated, index) => {
    const gradient = state.gradients[index];
    for (let element = 0; element < updated.length; element++)
      updated[element] -= lr * gradient[element];
  });
}

/** Zero the gradients, then forward, loss and backward. */
async function forwardBackward(device, model, data, zero) {
  await zero();
  const input = device.tensor(data.values, { shape: data.inputShape });
  const loss = new nn.CrossEntropyLoss({ reduction: "mean" }).forward(
    model.forward(input),
    data.labels,
  );
  await loss.backward();
  return loss;
}

/**
 * Train a model with the device optimizer for `steps` steps. A second copy of the model runs the
 * host algorithm: its parameters are written from the host values each step, and its gradients are
 * computed there. Both see the same inputs and the same initial values.
 */
async function parity(device, kind, makeOptimizer, hostStep, options, steps) {
  const data = batch(kind);
  const trained = models(kind).to(device);
  const copy = models(kind).to(device);
  const optimizer = makeOptimizer(trained.parameters());
  // `step()` must not read the gradients back: the optimizer's own read is made to fail.
  optimizer.gradients = async () => {
    throw new Error("step() must not read gradients back");
  };
  const copyParameters = copy.parameters();
  const copyZero = nn.optim.SGD(copyParameters, { lr: 1 });
  const host = { values: [], gradients: [], moments: [], iteration: 0 };
  for (const parameter of copyParameters) {
    const values = Float32Array.from(await parameter.parameterValues());
    host.values.push(values);
    host.moments.push({
      first: new Float32Array(values.length),
      second: new Float32Array(values.length),
    });
  }
  for (let step = 0; step < steps; step++) {
    await Promise.all(
      copyParameters.map((parameter, index) =>
        parameter.assign(host.values[index]),
      ),
    );
    const copyLoss = await forwardBackward(device, copy, data, () =>
      copyZero.zeroGrad(),
    );
    host.gradients = await Promise.all(
      copyParameters.map((parameter) => parameter.grad.data()),
    );
    hostStep(host, options);
    await copyLoss.disposeGraph(copyParameters);

    const loss = await forwardBackward(device, trained, data, () =>
      optimizer.zeroGrad(),
    );
    await optimizer.step();
    await loss.disposeGraph(trained.parameters());
  }
  // Two measures. `scaled` is the largest absolute difference over the largest magnitude of its
  // parameter, which is what f32 rounding bounds. `element` is the largest relative difference of
  // the elements at least 1% of that magnitude: a relative measure that a near-zero element does
  // not dominate.
  const measure = { scaled: 0, element: 0, maxAbsolute: 0 };
  for (const [index, parameter] of trained.parameters().entries()) {
    const actual = await parameter.parameterValues();
    const expected = host.values[index];
    const magnitude = Math.max(...expected.map(Math.abs));
    let maxAbsolute = 0;
    for (let element = 0; element < actual.length; element++) {
      const absolute = Math.abs(actual[element] - expected[element]);
      maxAbsolute = Math.max(maxAbsolute, absolute);
      if (Math.abs(expected[element]) >= 0.01 * magnitude)
        measure.element = Math.max(
          measure.element,
          absolute / Math.abs(expected[element]),
        );
    }
    measure.maxAbsolute = Math.max(measure.maxAbsolute, maxAbsolute);
    measure.scaled = Math.max(measure.scaled, maxAbsolute / magnitude);
  }
  return measure;
}

/**
 * The bounds of the parity tests. The device update is f32 and the host algorithm is f64 until
 * it rounds to f32, so the two differ by rounding. Accepted (decision of the owner's
 * orchestrator): 2e-5 relative per element at least 1% of the parameter's magnitude, and 1e-5
 * relative to the largest element of the parameter. Measured over 50 steps: 1.21e-5 per
 * element and 5.7e-7 scaled for Adam, 6e-7 for SGD.
 */
function assertParity(measure) {
  assert.ok(measure.scaled <= 1e-5, `scaled difference ${measure.scaled}`);
  assert.ok(measure.element <= 2e-5, `element difference ${measure.element}`);
}

const onCpu = (body) => async () => {
  const device = await nn.device({ device: "cpu" });
  try {
    await body(device);
  } finally {
    await device.dispose();
  }
};

test("Adam on the device matches the host algorithm over 50 steps", async (t) => {
  const options = { lr: 0.01, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 };
  for (const kind of ["mlp", "conv"]) {
    await t.test(
      kind,
      onCpu(async (device) => {
        const measure = await parity(
          device,
          kind,
          (parameters) => nn.optim.Adam(parameters, options),
          hostAdam,
          options,
          50,
        );
        console.log(`Adam ${kind} parity: ${JSON.stringify(measure)}`);
        assertParity(measure);
      }),
    );
  }
});

test("SGD on the device matches the host algorithm over 50 steps", async (t) => {
  for (const kind of ["mlp", "conv"]) {
    await t.test(
      kind,
      onCpu(async (device) => {
        const measure = await parity(
          device,
          kind,
          (parameters) => nn.optim.SGD(parameters, { lr: 0.1 }),
          hostSgd,
          { lr: 0.1 },
          50,
        );
        console.log(`SGD ${kind} parity: ${JSON.stringify(measure)}`);
        assertParity(measure);
      }),
    );
  }
});

/**
 * Records every upload and readback of a fake WebGPU device. An upload is the bytes written to a
 * buffer, and whether they are all zero. A readback is the bytes copied out of a buffer.
 */
function traffic(gpu) {
  const log = { uploads: [], read: 0 };
  const queue = gpu.device.queue;
  const write = queue.writeBuffer;
  queue.writeBuffer = (target, offset, data, ...rest) => {
    const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    log.uploads.push({
      bytes: data.byteLength,
      zero: bytes.every((byte) => byte === 0),
      storage: ((target.record?.usage ?? 0) & 128) !== 0,
    });
    return write(target, offset, data, ...rest);
  };
  // The fake device returns one encoder for every call, so each encoder is wrapped once.
  const wrapped = new WeakSet();
  const createEncoder = gpu.device.createCommandEncoder;
  gpu.device.createCommandEncoder = () => {
    const encoder = createEncoder();
    if (!wrapped.has(encoder)) {
      wrapped.add(encoder);
      const copy = encoder.copyBufferToBuffer;
      encoder.copyBufferToBuffer = (
        source,
        sourceOffset,
        target,
        targetOffset,
        size,
      ) => {
        log.read += size;
        return copy(source, sourceOffset, target, targetOffset, size);
      };
    }
    return encoder;
  };
  return log;
}

test("a training step reads back only the loss and uploads no parameter values", async () => {
  const gpu = fakeGpu();
  const device = await nn.device({ device: "webgpu", gpuDevice: gpu.device });
  try {
    const data = batch("mlp");
    const model = models("mlp").to(device);
    const parameters = model.parameters();
    const optimizer = nn.optim.Adam(parameters, { lr: 0.01 });
    const log = traffic(gpu);
    const parameterBytes = parameters.reduce(
      (total, parameter) => total + count(parameter.shape) * 4,
      0,
    );
    const report = [];
    for (let step = 1; step <= 3; step++) {
      log.uploads.length = 0;
      log.read = 0;
      await optimizer.zeroGrad();
      const loss = new nn.CrossEntropyLoss({ reduction: "mean" }).forward(
        model.forward(device.tensor(data.values, { shape: data.inputShape })),
        data.labels,
      );
      await loss.data();
      await loss.backward();
      const lossBytes = 4; // a scalar mean loss
      const iterationRead = log.read;
      log.uploads.length = 0;
      log.read = 0;
      await optimizer.step();
      const stepUploads = log.uploads;
      const stepRead = log.read;
      report.push({
        step,
        iterationRead,
        stepRead,
        stepUploads: stepUploads.length,
        stepUploadBytes: stepUploads.reduce((t, u) => t + u.bytes, 0),
        zeroFillBytes: stepUploads
          .filter((u) => u.storage && u.zero)
          .reduce((t, u) => t + u.bytes, 0),
        valueBytes: stepUploads
          .filter((u) => u.storage && !u.zero)
          .reduce((t, u) => t + u.bytes, 0),
      });
      assert.equal(
        iterationRead,
        lossBytes,
        `step ${step} iteration reads back ${iterationRead} bytes`,
      );
      assert.equal(
        stepRead,
        0,
        `step ${step} reads back ${stepRead} bytes in step()`,
      );
      assert.equal(
        report.at(-1).valueBytes,
        0,
        `step ${step} uploads ${report.at(-1).valueBytes} bytes of values in step()`,
      );
      await loss.disposeGraph(parameters);
    }
    console.log(
      `parameter bytes ${parameterBytes}; traffic ${JSON.stringify(report)}`,
    );
  } finally {
    await device.dispose();
  }
});

test("parameters keep their identity, and their values are read back on demand", async () => {
  const device = await nn.device({ device: "cpu" });
  try {
    const data = batch("mlp");
    const model = models("mlp").to(device);
    const parameters = model.parameters();
    const optimizer = nn.optim.Adam(parameters, { lr: 0.01 });
    const train = async () => {
      await optimizer.zeroGrad();
      const loss = new nn.CrossEntropyLoss({ reduction: "mean" }).forward(
        model.forward(device.tensor(data.values, { shape: data.inputShape })),
        data.labels,
      );
      await loss.backward();
      await optimizer.step();
      await loss.disposeGraph(parameters);
    };
    for (let step = 0; step < 3; step++) await train();

    // The same objects, after the steps.
    assert.deepEqual(model.parameters(), parameters);
    assert.ok(
      model
        .parameters()
        .every((parameter, index) => parameter === parameters[index]),
    );
    // The host value is the value a computation on the device reads.
    for (const parameter of parameters) {
      const host = Float32Array.from(await parameter.parameterValues());
      const onDevice = await parameter.mul(onesLike(device, parameter)).data();
      assert.deepEqual(Array.from(host), Array.from(onDevice));
    }
    // assign() after steps replaces the values, and the next plan reads them.
    const first = parameters[0];
    const replacement = Float32Array.from(
      { length: count(first.shape) },
      (_, i) => i / 10,
    );
    await first.assign(replacement);
    assert.deepEqual(
      Array.from(await first.parameterValues()),
      Array.from(replacement),
    );
    assert.deepEqual(
      Array.from(await first.mul(onesLike(device, first)).data()),
      Array.from(replacement),
    );
    // A step after assign() starts from the assigned values.
    const before = Float32Array.from(replacement);
    await train();
    const after = await first.parameterValues();
    assert.ok(after.some((value, index) => value !== before[index]));
  } finally {
    await device.dispose();
  }
});

function onesLike(device, parameter) {
  return device.tensor(new Float32Array(count(parameter.shape)).fill(1), {
    shape: parameter.shape,
  });
}
