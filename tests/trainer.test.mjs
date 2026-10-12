import test, { after } from "node:test";
import assert from "node:assert/strict";
import { closeNn, loadNn } from "./_nn.mjs";
import { difference, f32, random, values } from "./_reference.mjs";

const nn = await loadNn();
after(closeNn);

/** A classification set of `classes` clusters in `features` dimensions, in a seeded order. */
function clusters(seed, size, features, classes) {
  const next = random(seed);
  const inputs = new Float32Array(size * features);
  const labels = new Uint32Array(size);
  for (let row = 0; row < size; row++) {
    const label = Math.floor(next() * classes);
    labels[row] = label;
    for (let feature = 0; feature < features; feature++)
      inputs[row * features + feature] = f32(
        (feature % classes === label ? 1 : 0) + (next() - 0.5) * 0.4,
      );
  }
  return { inputs, labels };
}

/** One SGD step on the mean softmax cross entropy of `inputs w + bias`, in plain JavaScript. */
function sgdStep(inputs, labels, weight, bias, rate, features, classes) {
  const rows = labels.length;
  const gradientW = new Float64Array(weight.length);
  const gradientB = new Float64Array(bias.length);
  for (let row = 0; row < rows; row++) {
    const logits = Array.from({ length: classes }, (_, c) => {
      let total = bias[c];
      for (let k = 0; k < features; k++)
        total += inputs[row * features + k] * weight[k * classes + c];
      return total;
    });
    const peak = Math.max(...logits);
    const exps = logits.map((v) => Math.exp(v - peak));
    const sum = exps.reduce((p, q) => p + q, 0);
    for (let c = 0; c < classes; c++) {
      const delta = (exps[c] / sum - (c === labels[row] ? 1 : 0)) / rows;
      gradientB[c] += delta;
      for (let k = 0; k < features; k++)
        gradientW[k * classes + c] += inputs[row * features + k] * delta;
    }
  }
  return [
    Float32Array.from(weight, (w, i) => w - rate * gradientW[i]),
    Float32Array.from(bias, (b, i) => b - rate * gradientB[i]),
  ];
}

test("one epoch of one batch is one SGD step of plain JavaScript", async () => {
  const device = await nn.device("cpu");
  try {
    const [features, classes] = [6, 3];
    const { inputs, labels } = clusters(11, 24, features, classes);
    const model = new nn.Linear(features, classes, {
      seed: 5,
      weightScale: 0.3,
    });
    const [weight, bias] = [model.weight.slice(), model.bias.slice()];
    const trainer = nn.createTrainer(model, {
      device,
      optimizer: nn.optim.SGD({ learningRate: 0.5 }),
    });
    await trainer.fit({ inputs, labels }, { epochs: 1, batchSize: 24 });
    const [expectedWeight, expectedBias] = sgdStep(
      inputs,
      labels,
      weight,
      bias,
      0.5,
      features,
      classes,
    );
    assert.equal(difference(model.weight, expectedWeight, 1e-5), undefined);
    assert.equal(difference(model.bias, expectedBias, 1e-5), undefined);
  } finally {
    await device.dispose();
  }
});

test("the loss falls and the accuracy rises over the epochs", async () => {
  const device = await nn.device("cpu");
  try {
    const [features, classes] = [8, 4];
    const { inputs, labels } = clusters(12, 96, features, classes);
    const model = new nn.Linear(features, classes, {
      seed: 6,
      weightScale: 0.05,
    });
    const trainer = nn.createTrainer(model, {
      device,
      optimizer: nn.optim.SGD({ learningRate: 0.5 }),
    });
    const seen = [];
    const history = await trainer.fit(
      { inputs, labels },
      { epochs: 12, batchSize: 32, onEpoch: (m) => seen.push(m.epoch) },
    );
    assert.deepEqual(
      seen,
      history.map((m) => m.epoch),
    );
    assert.equal(history.length, 12);
    assert.ok(
      history[11].loss < history[0].loss * 0.5,
      `loss ${history[0].loss} to ${history[11].loss}`,
    );
    assert.ok(history[11].accuracy > 0.95, `accuracy ${history[11].accuracy}`);
    assert.ok(history[0].accuracy <= history[11].accuracy);
    // The trained values are in the model, and the parameter tensors a caller holds see them.
    const [weightTensor] = model.to(device).parameters();
    assert.equal(
      difference(await weightTensor.data(), model.weight),
      undefined,
    );
  } finally {
    await device.dispose();
  }
});

test("the trainer refuses a dataset or options that do not fit", async () => {
  const device = await nn.device("cpu");
  try {
    const model = new nn.Linear(2, 2);
    const trainer = nn.createTrainer(model, {
      device,
      optimizer: nn.optim.SGD({ learningRate: 0.1 }),
    });
    const good = {
      inputs: values(1, 4, 0, 1),
      labels: Uint32Array.from([0, 1]),
    };
    await assert.rejects(
      () =>
        trainer.fit(
          { inputs: values(1, 3, 0, 1), labels: good.labels },
          { epochs: 1, batchSize: 1 },
        ),
      RangeError,
    );
    await assert.rejects(
      () =>
        trainer.fit(
          { ...good, labels: Uint32Array.from([0, 2]) },
          { epochs: 1, batchSize: 1 },
        ),
      /label exceeds/,
    );
    await assert.rejects(
      () => trainer.fit(good, { epochs: 0, batchSize: 1 }),
      /epochs/,
    );
    assert.throws(
      () => nn.createTrainer(model, { device, optimizer: { kind: "adam" } }),
      /SGD only/,
    );
  } finally {
    await device.dispose();
  }
});
