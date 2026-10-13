import test, { after } from "node:test";
import assert from "node:assert/strict";
import { closeNn, loadNn } from "./_nn.mjs";

// The API the training lab and the kernel-graph view read: graph(), device.lastPlan, the
// activation layers with their gradients, MSELoss, and two small training runs on the CPU tier.

const nn = await loadNn();
after(closeNn);

/** Run `body` on a CPU device, and dispose of it. */
async function onCpu(body) {
  const device = await nn.device({ device: "cpu" });
  try {
    await body(device);
  } finally {
    await device.dispose();
  }
}

const close = (actual, expected, tolerance = 1e-5) => {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++) {
    const absolute = Math.abs(actual[i] - expected[i]);
    const relative = absolute / Math.max(Math.abs(expected[i]), 1e-8);
    assert.ok(
      absolute <= tolerance || relative <= tolerance,
      `index ${i}: got ${actual[i]}, expected ${expected[i]}`,
    );
  }
};

test("graph() lists a chain in topological order with stable ids", async () => {
  await onCpu(async (device) => {
    const a = device.tensor(new Float32Array(6).fill(1), { shape: [2, 3] });
    const b = device.tensor(new Float32Array(12).fill(1), { shape: [3, 4] });
    const c = device.tensor(new Float32Array(8).fill(1), { shape: [2, 4] });
    const y = a.matmul(b).add(c).relu();
    const first = nn.graph(y);
    const second = nn.graph(y);
    assert.deepEqual(
      first.nodes.map((node) => node.id),
      second.nodes.map((node) => node.id),
    );
    assert.equal(first.output, first.nodes.at(-1).id);
    assert.equal(first.nodes.length, 6);
    const byId = new Map(first.nodes.map((node) => [node.id, node]));
    const position = new Map(
      first.nodes.map((node, index) => [node.id, index]),
    );
    for (const node of first.nodes)
      for (const input of node.inputs)
        assert.ok(position.get(input) < position.get(node.id), "inputs first");
    assert.equal(first.nodes.at(-1).id, first.output);
    assert.deepEqual(
      first.nodes.map((node) => [node.kind, node.op, node.shape]),
      [
        ["input", "input", [2, 3]],
        ["input", "input", [3, 4]],
        ["op", "matmul", [2, 4]],
        ["input", "input", [2, 4]],
        ["op", "add", [2, 4]],
        ["op", "relu", [2, 4]],
      ],
    );
    const add = first.nodes[4];
    assert.equal(add.inputs.length, 2);
    assert.equal(byId.get(add.inputs[0]).op, "matmul");
    assert.equal(byId.get(add.inputs[1]).op, "input");
    assert.ok(first.nodes.every((node) => node.parameter === false));
    assert.equal(first.nodes[0].computed, true);
    assert.equal(first.nodes[2].computed, false);
    await y.data();
    const after = nn.graph(y);
    assert.ok(
      after.nodes
        .filter((node) => node.kind === "op")
        .every((node) => node.computed),
    );
  });
});

test("device.lastPlan lines up with the calls of the plan", async () => {
  await onCpu(async (device) => {
    assert.equal(device.lastPlan, undefined);
    const a = device.tensor([1, -2, 3, -4], { shape: [4] });
    const b = device.tensor([0.5, 0.5, 0.5, 0.5], { shape: [4] });
    const y = a.add(b).mul(b).relu().sigmoid();
    await y.data();
    const { report, nodes } = device.lastPlan;
    const calls = report.groups.flatMap((group) => group.calls);
    assert.equal(nodes.length, calls.length);
    assert.equal(nodes.length, 4);
    const snapshot = nn.graph(y);
    const ops = new Map(snapshot.nodes.map((node) => [node.id, node.op]));
    // A call is named by its kernel function (`sigmoidMap`), so only the ids are compared.
    for (const id of nodes)
      assert.ok(ops.get(id) !== undefined && ops.get(id) !== "input");
  });
});

test("device.lastPlan records a repeated input as a copy of the call that reads it", async () => {
  await onCpu(async (device) => {
    const a = device.tensor([1, 2, 3], { shape: [3] });
    const y = a.add(a);
    await y.data();
    const { report, nodes } = device.lastPlan;
    const calls = report.groups.flatMap((group) => group.calls);
    assert.equal(nodes.length, calls.length);
    assert.equal(new Set(nodes).size, 1);
    assert.equal(nodes[0], nn.graph(y).output);
  });
});

test("sigmoid and tanh gradients match finite differences", async () => {
  await onCpu(async (device) => {
    const values = new Float32Array([0.4, -0.7, 1.3]);
    const reference = {
      sigmoid: (x) => 1 / (1 + Math.exp(-x)),
      tanh: (x) => Math.tanh(x),
    };
    for (const name of ["sigmoid", "tanh"]) {
      const parameter = device.createParameter(values.slice(), { shape: [3] });
      const objective = parameter[name]().sum();
      await objective.backward();
      const analytic = await parameter.grad.data();
      const h = 1e-3;
      const numeric = [...values].map((x, i) => {
        const plus = values.map((v, j) => (j === i ? v + h : v));
        const minus = values.map((v, j) => (j === i ? v - h : v));
        const f = (array) =>
          array.reduce((sum, v) => sum + reference[name](v), 0);
        return (f(plus) - f(minus)) / (2 * h);
      });
      close(analytic, numeric, 1e-3);
      await objective.disposeGraph([parameter, parameter.grad]);
    }
  });
});

test("Tanh and Sigmoid layers apply their activation", async () => {
  await onCpu(async (device) => {
    const x = device.tensor([0, 1, -1], { shape: [1, 3] });
    const tanh = new nn.Tanh().to(device);
    const sigmoid = new nn.Sigmoid().to(device);
    close(await tanh.forward(x).data(), [0, Math.tanh(1), Math.tanh(-1)], 1e-6);
    close(
      await sigmoid.forward(x).data(),
      [0.5, 1 / (1 + Math.exp(-1)), 1 / (1 + Math.exp(1))],
      1e-6,
    );
  });
});

test("MSELoss is the mean of squared differences and backpropagates", async () => {
  await onCpu(async (device) => {
    const prediction = device.createParameter(new Float32Array([1, 2, 3, 4]), {
      shape: [2, 2],
    });
    const loss = new nn.MSELoss().forward(
      prediction,
      new Float32Array([0, 2, 5, 4]),
    );
    close(await loss.data(), [(1 + 0 + 4 + 0) / 4]);
    await loss.backward();
    close(await prediction.grad.data(), [0.5, 0, -1, 0]);
    const viaTensor = nn.loss
      .mse()
      .forward(prediction, device.tensor([0, 2, 5, 4], { shape: [2, 2] }));
    close(await viaTensor.data(), [1.25]);
    await loss.disposeGraph([prediction, prediction.grad]);
    await viaTensor.disposeGraph([prediction, prediction.grad]);
  });
});

test("a 2-16-1 tanh MLP fits y = sin(x) with Adam in 200 steps", async () => {
  await onCpu(async (device) => {
    const count = 32;
    const xs = Array.from(
      { length: count },
      (_, i) => -Math.PI + (2 * Math.PI * i) / (count - 1),
    );
    const features = new Float32Array(count * 2);
    const targets = new Float32Array(count);
    xs.forEach((x, i) => {
      features[2 * i] = x;
      features[2 * i + 1] = 1;
      targets[i] = Math.sin(x);
    });
    const model = new nn.Sequential(
      new nn.Linear(2, 16, { seed: 7, weightScale: 0.5 }),
      new nn.Tanh(),
      new nn.Linear(16, 1, { seed: 11, weightScale: 0.5 }),
    ).to(device);
    const optimizer = nn.optim.Adam(model.parameters(), { lr: 0.02 });
    const criterion = new nn.MSELoss();
    // zeroGrad() frees the gradient graph, which reads the input, so each step makes its own.
    const input = () => device.tensor(features, { shape: [count, 2] });
    for (let step = 0; step < 200; step++) {
      await optimizer.zeroGrad();
      const loss = criterion.forward(model.forward(input()), targets);
      await loss.backward();
      await optimizer.step();
      await loss.disposeGraph(model.parameters());
    }
    const final = criterion.forward(model.forward(input()), targets);
    const value = (await final.data())[0];
    assert.ok(value < 0.05, `MSE after 200 steps is ${value}`);
    await final.disposeGraph(model.parameters());
  });
});

test("a 2-16-2 ReLU MLP with cross entropy solves XOR in 300 steps", async () => {
  await onCpu(async (device) => {
    // zeroGrad() frees the gradient graph, which reads the input, so each use makes its own.
    const features = () =>
      device.tensor([0, 0, 0, 1, 1, 0, 1, 1], { shape: [4, 2] });
    const labels = [0, 1, 1, 0];
    const model = new nn.Sequential(
      new nn.Linear(2, 16, { seed: 3, weightScale: 1 }),
      new nn.ReLU(),
      new nn.Linear(16, 2, { seed: 5, weightScale: 1 }),
    ).to(device);
    const optimizer = nn.optim.Adam(model.parameters(), { lr: 0.05 });
    const criterion = new nn.CrossEntropyLoss();
    const accuracy = async () => {
      const output = model.forward(features());
      const logits = await output.data();
      await output.disposeGraph(model.parameters());
      let correct = 0;
      for (let row = 0; row < 4; row++) {
        const predicted = logits[2 * row + 1] > logits[2 * row] ? 1 : 0;
        if (predicted === labels[row]) correct++;
      }
      return correct / 4;
    };
    for (let step = 0; step < 300; step++) {
      await optimizer.zeroGrad();
      const loss = criterion.forward(model.forward(features()), labels);
      await loss.backward();
      await optimizer.step();
      await loss.disposeGraph(model.parameters());
      if ((await accuracy()) === 1) return;
    }
    assert.equal(await accuracy(), 1, "XOR accuracy after 300 steps");
  });
});

test("Conv2d output shape follows the padding and the strides", async () => {
  await onCpu(async (device) => {
    const x = device.tensor(new Float32Array(2 * 5 * 5 * 3).fill(1), {
      shape: [2, 5, 5, 3],
    });
    const cases = [
      [{ kernelSize: 3, padding: "valid" }, [2, 3, 3, 4]],
      [{ kernelSize: 3, padding: "same" }, [2, 5, 5, 4]],
      [{ kernelSize: 3, padding: "same", strides: 2 }, [2, 3, 3, 4]],
      [{ kernelSize: [3, 3], strides: 2 }, [2, 2, 2, 4]],
    ];
    for (const [options, shape] of cases) {
      const layer = new nn.Conv2d(3, 4, { ...options, seed: 1 }).to(device);
      const output = layer.forward(x);
      assert.deepEqual(output.shape, shape, JSON.stringify(options));
      assert.equal(
        (await output.data()).length,
        shape.reduce((a, b) => a * b),
      );
    }
  });
});

test("Conv2d adds its bias to every position of each output channel", async () => {
  await onCpu(async (device) => {
    const layer = new nn.Conv2d(1, 3, { kernelSize: 2 }).to(device);
    layer.weight.fill(0);
    layer.bias.set([1, -2, 0.5]);
    const x = device.tensor(new Float32Array(1 * 3 * 3).fill(7), {
      shape: [1, 3, 3, 1],
    });
    const output = await layer.forward(x).data();
    const expected = Array.from({ length: 2 * 2 }, () => [1, -2, 0.5]).flat();
    close(output, expected, 1e-6);
    assert.equal(layer.parameters().length, 2);
  });
});

test("Conv2d with bias false has one parameter", async () => {
  await onCpu(async (device) => {
    const layer = new nn.Conv2d(1, 2, { kernelSize: 2, bias: false }).to(
      device,
    );
    assert.equal(layer.parameters().length, 1);
    assert.equal(layer.parameters()[0].shape.join(","), "2,2,1,2");
  });
});

test("Conv2d filter and bias gradients match central differences", async () => {
  await onCpu(async (device) => {
    const input = [0.5, -1, 0.25, 1.5, 0.75, -0.5, 0.3, -0.2, 1.1];
    const coefficients = [1, -0.5, 0.25, 2, -1.5, 0.75, 0.5, -0.25];
    const weights = [0.2, -0.3, 0.4, 0.1, -0.5, 0.6, 0.7, -0.8];
    const biases = [0.05, -0.1];
    // The objective is sum(c * tanh(conv(x, w) + b)), so it is not linear in the parameters.
    const build = (w, b) => {
      const layer = new nn.Conv2d(1, 2, { kernelSize: 2 }).to(device);
      layer.weight.set(w);
      layer.bias.set(b);
      const x = device.tensor(input, { shape: [1, 3, 3, 1] });
      const c = device.tensor(coefficients, { shape: [1, 2, 2, 2] });
      const objective = layer.forward(x).tanh().mul(c).sum();
      return { layer, objective };
    };
    const { layer, objective } = build(weights, biases);
    await objective.backward();
    const [weight, bias] = layer.parameters();
    const analytic = [
      ...(await weight.grad.data()),
      ...(await bias.grad.data()),
    ];
    const value = async (w, b) => {
      const { objective: f } = build(w, b);
      return (await f.data())[0];
    };
    const h = 1e-3;
    const numeric = [];
    for (let i = 0; i < weights.length; i++) {
      const plus = weights.map((v, j) => (j === i ? v + h : v));
      const minus = weights.map((v, j) => (j === i ? v - h : v));
      numeric.push(
        ((await value(plus, biases)) - (await value(minus, biases))) / (2 * h),
      );
    }
    for (let i = 0; i < biases.length; i++) {
      const plus = biases.map((v, j) => (j === i ? v + h : v));
      const minus = biases.map((v, j) => (j === i ? v - h : v));
      numeric.push(
        ((await value(weights, plus)) - (await value(weights, minus))) /
          (2 * h),
      );
    }
    close(analytic, numeric, 1e-3);
  });
});

test("a Conv2d, ReLU, MaxPool2d, Flatten and Linear CNN learns bar orientation in 150 Adam steps", async () => {
  await onCpu(async (device) => {
    // A deterministic generator: each image has one full horizontal or vertical bar, at a random
    // row or column, with a little noise. Class 0 is horizontal, class 1 is vertical.
    const bars = (count, seed) => {
      let state = seed >>> 0;
      const next = () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return state / 0x100000000;
      };
      const pixels = new Float32Array(count * 64);
      const labels = [];
      for (let n = 0; n < count; n++) {
        const label = next() < 0.5 ? 0 : 1;
        const offset = Math.floor(next() * 8);
        labels.push(label);
        for (let k = 0; k < 8; k++) {
          const row = label === 0 ? offset : k;
          const column = label === 0 ? k : offset;
          pixels[n * 64 + row * 8 + column] = 1;
        }
        for (let i = 0; i < 64; i++) pixels[n * 64 + i] += (next() - 0.5) * 0.2;
      }
      return { pixels, labels };
    };
    const train = bars(64, 1);
    const test = bars(128, 2);
    const model = new nn.Sequential(
      new nn.Conv2d(1, 4, { kernelSize: 3, seed: 21 }),
      new nn.ReLU(),
      new nn.MaxPool2d({ filterSize: 2, strides: 2 }),
      new nn.Flatten(),
      new nn.Linear(36, 2, { seed: 22, weightScale: 0.2 }),
    ).to(device);
    const optimizer = nn.optim.Adam(model.parameters(), { lr: 0.01 });
    const criterion = new nn.CrossEntropyLoss();
    const accuracy = async ({ pixels, labels }) => {
      const count = labels.length;
      const output = model.forward(
        device.tensor(pixels, { shape: [count, 8, 8, 1] }),
      );
      const logits = await output.data();
      await output.disposeGraph(model.parameters());
      let correct = 0;
      for (let row = 0; row < count; row++) {
        const predicted = logits[2 * row + 1] > logits[2 * row] ? 1 : 0;
        if (predicted === labels[row]) correct++;
      }
      return correct / count;
    };
    for (let step = 0; step < 150; step++) {
      await optimizer.zeroGrad();
      const input = device.tensor(train.pixels, { shape: [64, 8, 8, 1] });
      const loss = criterion.forward(model.forward(input), train.labels);
      await loss.backward();
      await optimizer.step();
      await loss.disposeGraph(model.parameters());
    }
    const held = await accuracy(test);
    assert.ok(held >= 0.95, `held-out accuracy after 150 steps is ${held}`);
  });
});
