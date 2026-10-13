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
