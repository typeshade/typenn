// The numerics cases of TypeNN: a tensor graph built with the public API on a device, and the
// reference the plain-JavaScript model (`_reference.mjs`) gives for it. Each case carries its
// tolerance on the CPU tier, and on a GPU. The Node tests run every case on the CPU tier; the
// browser test runs the cases that are not `cpuOnly` on WebGPU and holds them to the CPU's values.
// Pure ES module.

import {
  BINARY,
  BROADCASTS,
  DERIVATIVES,
  EDGES,
  MAPS,
  REDUCTIONS,
  SCALED_MAPS,
  axisLayout,
  binaryGradReference,
  binaryReference,
  broadcastShape,
  conv2dGradReference,
  conv2dReference,
  crossEntropyReference,
  f32,
  matmulGradReference,
  matmulReference,
  matmulShape,
  poolGradReference,
  poolReference,
  reduceGradReference,
  reduceReference,
  sizeOf,
  softmaxGradReference,
  softmaxReference,
  values,
} from "./_reference.mjs";

/** @typedef {{ name: string, tolerance: number, gpuTolerance: number, cpuOnly?: boolean,
 *   run: (nn: any, device: any) => Promise<{ actual: ArrayLike<number>, expected: ArrayLike<number> }> }} Case */

let seed = 1000;
/** A fresh seed for each call, so that the cases differ; the list restarts the count, so that
 *  every call that builds it gets the same cases. */
const nextSeed = () => seed++;

const within = (range) => EDGES.filter((x) => x >= range[0] && x <= range[1]);

function mapInput(range) {
  return Float32Array.from([
    ...values(nextSeed(), 67, range[0], range[1]),
    ...within(range),
  ]);
}

/** @returns {Case[]} */
function mapCases() {
  const cases = [];
  for (const [name, spec] of Object.entries(MAPS)) {
    const x = mapInput(spec.range);
    cases.push({
      name: `map ${name}`,
      tolerance: spec.tolerance,
      gpuTolerance: spec.gpu ?? spec.tolerance,
      run: async (_nn, device) => ({
        actual: await device
          .tensor(x, { shape: [x.length] })
          [name]()
          .data(),
        expected: x.map(spec.f),
      }),
    });
  }
  for (const [name, spec] of Object.entries(SCALED_MAPS)) {
    const x = mapInput(spec.range);
    cases.push({
      name: `map ${name}`,
      tolerance: spec.tolerance,
      gpuTolerance: spec.tolerance,
      run: async (_nn, device) => ({
        actual: await spec.call(device.tensor(x, { shape: [x.length] })).data(),
        expected: x.map(spec.f),
      }),
    });
  }
  return cases;
}

/** @returns {Case[]} */
function binaryCases() {
  const cases = [];
  for (const [op, spec] of Object.entries(BINARY)) {
    for (const [kind, [aShape, bShape]] of Object.entries(BROADCASTS)) {
      const a = values(nextSeed(), sizeOf(aShape), ...spec.a);
      const b = values(nextSeed(), sizeOf(bShape), ...spec.b);
      cases.push({
        name: `binary ${op}, ${kind}`,
        tolerance: spec.tolerance,
        gpuTolerance: spec.gpu ?? spec.tolerance,
        run: async (_nn, device) => {
          const out = device
            .tensor(a, { shape: aShape })
            [op](device.tensor(b, { shape: bShape }));
          return {
            actual: await out.data(),
            expected: binaryReference(op, aShape, a, bShape, b),
          };
        },
      });
    }
  }
  return cases;
}

/** The add chain `relu((a + b) * c)` over the lengths that straddle a workgroup of 64, and over
 *  2^24 + 1 elements, where an f32 index would stop counting. */
function chainCases() {
  const cases = [];
  for (const length of [1, 63, 64, 65, 67, 4097, 16777216 + 1]) {
    const big = length > 1 << 24;
    // The seeds are drawn here, and the arrays when the case runs: the case of 2^24 + 1 elements
    // makes three arrays of 64 MB, and the browser leaves it out.
    const [seedA, seedB, seedC] = [nextSeed(), nextSeed(), nextSeed()];
    cases.push({
      name: `chain relu((a + b) * c), ${length} elements`,
      tolerance: 0,
      gpuTolerance: 0,
      ...(big ? { cpuOnly: true } : {}),
      run: async (_nn, device) => {
        const a = values(seedA, length, -4, 4);
        const b = values(seedB, length, -4, 4);
        const c = values(seedC, length, -2, 2);
        const shape = [length];
        const out = device
          .tensor(a, { shape })
          .add(device.tensor(b, { shape }))
          .mul(device.tensor(c, { shape }))
          .relu();
        const expected = new Float32Array(length);
        for (let i = 0; i < length; i++)
          expected[i] = Math.max(f32(f32(a[i] + b[i]) * c[i]), 0);
        return { actual: await out.data(), expected };
      },
    });
  }
  return cases;
}

/** Graphs of add, mul and relu from PR #6 of TypeNN: each node is `[op, ...inputs]`, an input a
 *  name of a leaf and `$n` the n-th node. `repeated leaf` and `diamond` read one tensor twice. */
const FIXTURES = [
  [
    "add-relu",
    { a: [2, -4, 0.5], b: [3, 1, -0.25] },
    [
      ["add", "a", "b"],
      ["relu", "$0"],
    ],
  ],
  [
    "add-mul-relu",
    { a: [1, 2, -3, 4], b: [5, -3, 2, -8], c: [2, 1, -2, 0.5] },
    [
      ["add", "a", "b"],
      ["mul", "$0", "c"],
      ["relu", "$1"],
    ],
  ],
  [
    "mul-add-relu",
    { a: [2, -3, 4, -2], b: [3, 2, -1, -4], c: [-1, 4, 5, -9] },
    [
      ["mul", "a", "b"],
      ["add", "$0", "c"],
      ["relu", "$1"],
    ],
  ],
  [
    "reordered four ops",
    { a: [-2, 3, -1, 4], b: [9, 2, 5, -3], c: [1, -1, 2, 0.5] },
    [
      ["relu", "a"],
      ["mul", "$0", "b"],
      ["add", "$1", "c"],
      ["relu", "$2"],
    ],
  ],
  [
    "repeated add and mul",
    { a: [1, -2, 3, 4], b: [2, 5, -1, 3], c: [1, -2, 4, -5] },
    [
      ["add", "a", "b"],
      ["mul", "$0", "a"],
      ["add", "$1", "c"],
      ["mul", "$2", "b"],
      ["relu", "$3"],
    ],
  ],
  [
    "repeated leaf",
    { a: [2, -3, 0, 4] },
    [
      ["add", "a", "a"],
      ["mul", "$0", "a"],
      ["relu", "$1"],
    ],
  ],
  [
    "diamond with a shared node",
    { a: [2, 1, 3], b: [3, -4, 2] },
    [
      ["add", "a", "b"],
      ["mul", "$0", "a"],
      ["mul", "$0", "b"],
      ["add", "$1", "$2"],
      ["relu", "$3"],
    ],
  ],
];

/** @returns {Case[]} */
function fixtureCases() {
  return FIXTURES.map(([name, inputs, nodes]) => ({
    name: `graph ${name}`,
    tolerance: 0,
    gpuTolerance: 0,
    run: async (_nn, device) => {
      const length = Object.values(inputs)[0].length;
      const read = (nodesSoFar, leaves, reference) =>
        reference.startsWith("$")
          ? nodesSoFar[Number(reference.slice(1))]
          : leaves[reference];
      // The tensors, built with the public API.
      const leaves = Object.fromEntries(
        Object.entries(inputs).map(([key, v]) => [
          key,
          device.tensor(v, { shape: [length] }),
        ]),
      );
      const tensors = [];
      for (const [op, ...args] of nodes) {
        const [first, second] = args.map((r) => read(tensors, leaves, r));
        tensors.push(op === "relu" ? first.relu() : first[op](second));
      }
      // The same graph in f32, one element at a time.
      const expected = new Float32Array(length);
      for (let i = 0; i < length; i++) {
        const numbers = [];
        const get = (r) =>
          r.startsWith("$") ? numbers[Number(r.slice(1))] : f32(inputs[r][i]);
        for (const [op, ...args] of nodes) {
          const [x, y] = args.map(get);
          numbers.push(
            op === "relu"
              ? Math.max(0, x)
              : op === "add"
                ? f32(x + y)
                : f32(x * y),
          );
        }
        expected[i] = numbers[numbers.length - 1];
      }
      return { actual: await tensors[tensors.length - 1].data(), expected };
    },
  }));
}

/** The gradient of each map, from an upstream gradient, against its derivative in double
 *  precision. */
function mapGradientCases() {
  const cases = [];
  for (const name of Object.keys(DERIVATIVES)) {
    const spec = MAPS[name];
    // The derivative of `rsqrt`, `acos` and the like is large at the ends of the range.
    const [low, high] = spec.range;
    const margin = (high - low) * 0.02;
    const x = values(nextSeed(), 67, low + margin, high - margin);
    const upstream = values(nextSeed(), 67, -2, 2);
    cases.push({
      name: `gradient of ${name}`,
      tolerance: 1e-5,
      gpuTolerance: 1e-5,
      run: async (nn, device) => {
        const input = device.tensor(x, { shape: [67] });
        const [gradient] = nn.gradients(
          input[name](),
          [input],
          device.tensor(upstream, { shape: [67] }),
        );
        return {
          actual: await gradient.data(),
          expected: x.map((v, i) => f32(upstream[i] * DERIVATIVES[name](v))),
        };
      },
    });
  }
  return cases;
}

/** The gradient of each binary operation with respect to both operands, for each way they meet. */
function binaryGradientCases() {
  const cases = [];
  for (const [op, spec] of Object.entries(BINARY)) {
    for (const [kind, [aShape, bShape]] of Object.entries(BROADCASTS)) {
      const a = values(nextSeed(), sizeOf(aShape), ...spec.a);
      const b = values(nextSeed(), sizeOf(bShape), ...spec.b);
      const shape = broadcastShape(aShape, bShape);
      const dy = values(nextSeed(), sizeOf(shape), -2, 2);
      const [expectedA, expectedB] = binaryGradReference(
        op,
        aShape,
        a,
        bShape,
        b,
        dy,
      );
      for (const [target, expected] of [
        ["a", expectedA],
        ["b", expectedB],
      ]) {
        cases.push({
          name: `gradient of binary ${op} with respect to ${target}, ${kind}`,
          tolerance: 1e-5,
          gpuTolerance: 1e-5,
          run: async (nn, device) => {
            const x = device.tensor(a, { shape: aShape });
            const y = device.tensor(b, { shape: bShape });
            const out = x[op](y);
            const [gradient] = nn.gradients(
              out,
              [target === "a" ? x : y],
              device.tensor(dy, { shape }),
            );
            return { actual: await gradient.data(), expected };
          },
        });
      }
    }
  }
  return cases;
}

/** Graphs that read one tensor twice, and the gradients that flow back through both reads. */
function repeatedOperandCases() {
  const x = values(nextSeed(), 9, -3, 3);
  const upstream = values(nextSeed(), 9, -2, 2);
  const through = (name, build, expected) => ({
    name,
    tolerance: 1e-5,
    gpuTolerance: 1e-5,
    run: async (nn, device) => {
      const input = device.tensor(x, { shape: [9] });
      const [gradient] = nn.gradients(
        build(input),
        [input],
        device.tensor(upstream, { shape: [9] }),
      );
      return {
        actual: await gradient.data(),
        expected: x.map((v, i) => expected(v, upstream[i])),
      };
    },
  });
  return [
    through(
      "gradient of x + x",
      (t) => t.add(t),
      (v, d) => f32(2 * d),
    ),
    through(
      "gradient of x * x",
      (t) => t.mul(t),
      (v, d) => f32(2 * v * d),
    ),
    through(
      "gradient of relu(x) * x",
      (t) => t.relu().mul(t),
      (v, d) => (v > 0 ? f32(2 * v * d) : 0),
    ),
    through(
      "gradient of reshape",
      (t) => t.reshape([3, 3]).square().reshape([9]),
      (v, d) => f32(2 * v * d),
    ),
  ];
}

/** @returns {Case[]} */
export function elementwiseCases() {
  seed = 1000;
  return [
    ...mapCases(),
    ...binaryCases(),
    ...chainCases(),
    ...fixtureCases(),
    ...mapGradientCases(),
    ...binaryGradientCases(),
    ...repeatedOperandCases(),
  ];
}

// ─── reductions and matmul ────────────────────────────────────────────────────────────────────

/** Shapes and axes a reduction runs over: every axis, and each axis of rank 1, 2 and 3. */
const REDUCE_AXES = [
  [[5], undefined],
  [[5], 0],
  [[3, 4], undefined],
  [[3, 4], 0],
  [[3, 4], 1],
  [[3, 4], -1],
  [[2, 3, 4], 0],
  [[2, 3, 4], 1],
  [[2, 3, 4], 2],
  [[1, 1, 5], 1],
];

/** @returns {Case[]} */
function reductionCases() {
  const cases = [];
  for (const [op, spec] of Object.entries(REDUCTIONS)) {
    for (const [shape, axis] of REDUCE_AXES) {
      // `argMax()` and `argMin()` take axis 0 by default: they reduce no more than one axis.
      if (axis === undefined && op.startsWith("arg")) continue;
      const x = values(nextSeed(), sizeOf(shape), ...spec.range);
      cases.push({
        name: `reduction ${op}, shape [${shape}], axis ${axis}`,
        tolerance: spec.tolerance,
        gpuTolerance: spec.gpu ?? spec.tolerance,
        run: async (_nn, device) => ({
          actual: await device.tensor(x, { shape })[op](axis).data(),
          expected: reduceReference(op, shape, x, axis),
        }),
      });
    }
  }
  // The first of equal elements wins an arg-max and an arg-min.
  const ties = Float32Array.from([1, 3, 3, 2, 3, 0, 0, 5, 5, 5]);
  for (const [op, expected] of [
    ["argMax", [1, 2]],
    ["argMin", [0, 0]],
  ])
    cases.push({
      name: `reduction ${op} with ties`,
      tolerance: 0,
      gpuTolerance: 0,
      run: async (_nn, device) => ({
        actual: await device
          .tensor(ties, { shape: [2, 5] })
          [op](1)
          .data(),
        expected,
      }),
    });
  return cases;
}

/** @returns {Case[]} */
function reductionGradientCases() {
  const cases = [];
  for (const op of ["sum", "mean"]) {
    for (const [shape, axis] of REDUCE_AXES) {
      const x = values(nextSeed(), sizeOf(shape), -3, 3);
      const outShape = axisLayout(shape, axis).outShape;
      const dy = values(nextSeed(), sizeOf(outShape), -2, 2);
      cases.push({
        name: `gradient of ${op}, shape [${shape}], axis ${axis}`,
        tolerance: 0,
        gpuTolerance: 1e-5,
        run: async (nn, device) => {
          const input = device.tensor(x, { shape });
          const [gradient] = nn.gradients(
            input[op](axis),
            [input],
            device.tensor(dy, { shape: outShape }),
          );
          return {
            actual: await gradient.data(),
            expected: reduceGradReference(op, shape, dy, axis),
          };
        },
      });
    }
  }
  return cases;
}

/** [a shape, b shape, transposeA, transposeB]: a matrix is stored transposed where it is flagged. */
const MATMULS = [
  [[2, 3], [3, 4], false, false],
  [[3, 2], [3, 4], true, false],
  [[2, 3], [4, 3], false, true],
  [[3, 2], [4, 3], true, true],
  [[1, 1], [1, 1], false, false],
  [[1, 5], [5, 1], false, false],
  [[5, 1], [1, 5], false, false],
  [[17, 33], [33, 9], false, false],
  [[33, 17], [9, 33], true, true],
  [[64, 100], [100, 32], false, false],
];

/** @returns {Case[]} */
function matmulCases() {
  const cases = [];
  for (const [aShape, bShape, transposeA, transposeB] of MATMULS) {
    const label = `[${aShape}] x [${bShape}]${transposeA ? ", transposeA" : ""}${transposeB ? ", transposeB" : ""}`;
    const a = values(nextSeed(), sizeOf(aShape), -2, 2);
    const b = values(nextSeed(), sizeOf(bShape), -2, 2);
    cases.push({
      name: `matmul ${label}`,
      tolerance: 0,
      gpuTolerance: 1e-5,
      run: async (_nn, device) => ({
        actual: await device
          .tensor(a, { shape: aShape })
          .matmul(device.tensor(b, { shape: bShape }), {
            transposeA,
            transposeB,
          })
          .data(),
        expected: matmulReference(a, aShape, b, bShape, transposeA, transposeB),
      }),
    });
    const dims = matmulShape(aShape, bShape, transposeA, transposeB);
    const dc = values(nextSeed(), dims.rows * dims.cols, -2, 2);
    const [expectedA, expectedB] = matmulGradReference(
      a,
      aShape,
      b,
      bShape,
      transposeA,
      transposeB,
      dc,
    );
    for (const [target, expected] of [
      ["a", expectedA],
      ["b", expectedB],
    ])
      cases.push({
        name: `gradient of matmul ${label} with respect to ${target}`,
        tolerance: 1e-5,
        gpuTolerance: 1e-5,
        run: async (nn, device) => {
          const x = device.tensor(a, { shape: aShape });
          const y = device.tensor(b, { shape: bShape });
          const out = x.matmul(y, { transposeA, transposeB });
          const [gradient] = nn.gradients(
            out,
            [target === "a" ? x : y],
            device.tensor(dc, { shape: [dims.rows, dims.cols] }),
          );
          return { actual: await gradient.data(), expected };
        },
      });
  }
  return cases;
}

/** An MLP layer, `relu(x w + bias)`, and the gradient of its mean with respect to w. */
function layerCases() {
  const [x, w, bias] = [
    values(nextSeed(), 12 * 20, -1, 1),
    values(nextSeed(), 20 * 7, -1, 1),
    values(nextSeed(), 7, -1, 1),
  ];
  return [
    {
      name: "matmul, add a bias, relu",
      tolerance: 0,
      gpuTolerance: 1e-5,
      run: async (_nn, device) => {
        const y = device
          .tensor(x, { shape: [12, 20] })
          .matmul(device.tensor(w, { shape: [20, 7] }))
          .add(device.tensor(bias, { shape: [7] }))
          .relu();
        const product = matmulReference(x, [12, 20], w, [20, 7]);
        return {
          actual: await y.data(),
          expected: product.map((v, i) => Math.max(f32(v + bias[i % 7]), 0)),
        };
      },
    },
  ];
}

export function reductionAndMatmulCases() {
  seed = 2000;
  return [
    ...reductionCases(),
    ...reductionGradientCases(),
    ...matmulCases(),
    ...layerCases(),
  ];
}

// ─── softmax and cross entropy ────────────────────────────────────────────────────────────────

const SOFTMAX_AXES = [
  [[5], -1],
  [[3, 4], 0],
  [[3, 4], 1],
  [[3, 4], -1],
  [[2, 3, 4], 0],
  [[2, 3, 4], 1],
  [[2, 3, 4], 2],
];

/** @returns {Case[]} */
function softmaxCases() {
  const cases = [];
  for (const [name, log] of [
    ["softmax", false],
    ["logSoftmax", true],
  ]) {
    for (const [shape, axis] of SOFTMAX_AXES) {
      const x = values(nextSeed(), sizeOf(shape), -4, 4);
      const dy = values(nextSeed(), sizeOf(shape), -2, 2);
      const label = `shape [${shape}], axis ${axis}`;
      cases.push({
        name: `${name}, ${label}`,
        tolerance: 1e-5,
        gpuTolerance: 1e-5,
        run: async (_nn, device) => ({
          actual: await device.tensor(x, { shape })[name](axis).data(),
          expected: softmaxReference(x, shape, axis, log),
        }),
      });
      cases.push({
        name: `gradient of ${name}, ${label}`,
        tolerance: 1e-5,
        gpuTolerance: 1e-5,
        run: async (nn, device) => {
          const input = device.tensor(x, { shape });
          const [gradient] = nn.gradients(
            input[name](axis),
            [input],
            device.tensor(dy, { shape }),
          );
          return {
            actual: await gradient.data(),
            expected: softmaxGradReference(
              softmaxReference(x, shape, axis, log),
              dy,
              shape,
              axis,
              log,
            ),
          };
        },
      });
    }
  }
  // A large logit must not overflow.
  cases.push({
    name: "softmax of large logits",
    tolerance: 1e-5,
    gpuTolerance: 1e-5,
    run: async (_nn, device) => ({
      actual: await device
        .tensor([1000, 1001, 999, -1000], { shape: [4] })
        .softmax()
        .data(),
      expected: softmaxReference([1000, 1001, 999, -1000], [4], 0),
    }),
  });
  return cases;
}

/** @returns {Case[]} */
function lossCases() {
  const cases = [];
  const [rows, columns] = [6, 5];
  const logits = values(nextSeed(), rows * columns, -3, 3);
  // Rows of a one-hot label, of a distribution, and of a weight other than 1.
  const labels = new Float32Array(rows * columns);
  [0, 3, 4, 1, 2, 0].forEach(
    (label, row) => (labels[row * columns + label] = 1),
  );
  labels.set([0.2, 0.3, 0.1, 0.25, 0.15], 1 * columns);
  labels.set([0, 0, 2, 0, 0], 5 * columns);
  const dy = values(nextSeed(), rows, -2, 2);
  const expected = crossEntropyReference(logits, labels, rows, columns, dy);
  const loss = (device, options) =>
    device
      .tensor(logits, { shape: [rows, columns] })
      .softmaxCrossEntropyWithLogits(
        device.tensor(labels, { shape: [rows, columns] }),
        options,
      );
  cases.push({
    name: "softmax cross entropy of each row",
    tolerance: 1e-5,
    gpuTolerance: 1e-5,
    run: async (_nn, device) => ({
      actual: await loss(device).data(),
      expected: expected.losses,
    }),
  });
  for (const reduction of ["sum", "mean"])
    cases.push({
      name: `softmax cross entropy, ${reduction}`,
      tolerance: 1e-5,
      gpuTolerance: 1e-5,
      run: async (_nn, device) => {
        const total = expected.losses.reduce((p, q) => p + q, 0);
        return {
          actual: await loss(device, { reduction }).data(),
          expected: [reduction === "sum" ? total : total / rows],
        };
      },
    });
  for (const [target, gradient] of [
    ["logits", expected.gradientLogits],
    ["labels", expected.gradientLabels],
  ])
    cases.push({
      name: `gradient of softmax cross entropy with respect to the ${target}`,
      tolerance: 1e-5,
      gpuTolerance: 1e-5,
      run: async (nn, device) => {
        const x = device.tensor(logits, { shape: [rows, columns] });
        const y = device.tensor(labels, { shape: [rows, columns] });
        const out = x.softmaxCrossEntropyWithLogits(y);
        const [actual] = nn.gradients(
          out,
          [target === "logits" ? x : y],
          device.tensor(dy, { shape: [rows] }),
        );
        return { actual: await actual.data(), expected: gradient };
      },
    });
  // The loss of a mean over the rows: the gradient with respect to the logits is the mean of them.
  cases.push({
    name: "gradient of the mean softmax cross entropy",
    tolerance: 1e-5,
    gpuTolerance: 1e-5,
    run: async (nn, device) => {
      const x = device.tensor(logits, { shape: [rows, columns] });
      const out = x.softmaxCrossEntropyWithLogits(
        device.tensor(labels, { shape: [rows, columns] }),
        { reduction: "mean" },
      );
      const [actual] = nn.gradients(out, [x]);
      const ones = new Float32Array(rows).fill(1 / rows);
      return {
        actual: await actual.data(),
        expected: crossEntropyReference(logits, labels, rows, columns, ones)
          .gradientLogits,
      };
    },
  });
  return cases;
}

export function softmaxAndLossCases() {
  seed = 3000;
  return [...softmaxCases(), ...lossCases()];
}

// ─── conv2d and pooling ───────────────────────────────────────────────────────────────────────

/** [input shape, filter shape, options]: NHWC input, HWIO filter. */
const CONVOLUTIONS = [
  [[2, 5, 6, 3], [3, 2, 3, 4], {}],
  [[2, 5, 6, 3], [3, 2, 3, 4], { padding: "same" }],
  [[2, 5, 6, 3], [3, 2, 3, 4], { strides: 2 }],
  [[2, 5, 6, 3], [3, 2, 3, 4], { strides: [2, 3], padding: "same" }],
  [[1, 7, 7, 2], [1, 1, 2, 3], {}],
  [[1, 4, 4, 1], [4, 4, 1, 2], { padding: "same" }],
  [[1, 28, 28, 1], [5, 5, 1, 6], { padding: "same" }],
];

const label = (shape, filter, options) =>
  `input [${shape}], filter [${filter}], ${JSON.stringify(options)}`;

/** @returns {Case[]} */
function convolutionCases() {
  const cases = [];
  for (const [shape, filterShape, options] of CONVOLUTIONS) {
    const input = values(nextSeed(), sizeOf(shape), -1, 1);
    const filter = values(nextSeed(), sizeOf(filterShape), -1, 1);
    const { out, outShape } = conv2dReference(
      input,
      shape,
      filter,
      filterShape,
      options,
    );
    cases.push({
      name: `conv2d ${label(shape, filterShape, options)}`,
      tolerance: 0,
      gpuTolerance: 1e-5,
      run: async (_nn, device) => ({
        actual: await device
          .tensor(input, { shape })
          .conv2d(device.tensor(filter, { shape: filterShape }), options)
          .data(),
        expected: out,
      }),
    });
    const dy = values(nextSeed(), sizeOf(outShape), -1, 1);
    const [expectedInput, expectedFilter] = conv2dGradReference(
      input,
      shape,
      filter,
      filterShape,
      options,
      dy,
    );
    for (const [target, expected] of [
      ["input", expectedInput],
      ["filter", expectedFilter],
    ])
      cases.push({
        name: `gradient of conv2d with respect to the ${target}, ${label(shape, filterShape, options)}`,
        tolerance: 1e-5,
        gpuTolerance: 1e-5,
        run: async (nn, device) => {
          const x = device.tensor(input, { shape });
          const w = device.tensor(filter, { shape: filterShape });
          const [actual] = nn.gradients(
            x.conv2d(w, options),
            [target === "input" ? x : w],
            device.tensor(dy, { shape: outShape }),
          );
          return { actual: await actual.data(), expected };
        },
      });
  }
  // `conv2dGradients` is the same pair as the gradients of the graph.
  const [shape, filterShape, options] = CONVOLUTIONS[1];
  const input = values(nextSeed(), sizeOf(shape), -1, 1);
  const filter = values(nextSeed(), sizeOf(filterShape), -1, 1);
  const { outShape } = conv2dReference(
    input,
    shape,
    filter,
    filterShape,
    options,
  );
  const dy = values(nextSeed(), sizeOf(outShape), -1, 1);
  const expected = conv2dGradReference(
    input,
    shape,
    filter,
    filterShape,
    options,
    dy,
  );
  for (const [index, target] of ["input", "filter"].entries())
    cases.push({
      name: `conv2dGradients, the ${target} gradient`,
      tolerance: 1e-5,
      gpuTolerance: 1e-5,
      run: async (_nn, device) => {
        const gradients = device
          .tensor(input, { shape })
          .conv2dGradients(
            device.tensor(filter, { shape: filterShape }),
            device.tensor(dy, { shape: outShape }),
            options,
          );
        return {
          actual: await gradients[index].data(),
          expected: expected[index],
        };
      },
    });
  return cases;
}

/** [input shape, options]: pooling windows of every padding, with a window larger than the stride
 *  and a window smaller than it. */
const POOLS = [
  [[2, 6, 5, 3], { filterSize: 2 }],
  [[2, 6, 5, 3], { filterSize: 2, padding: "same" }],
  [[2, 6, 5, 3], { filterSize: 3, strides: 1, padding: "same" }],
  [[2, 6, 5, 3], { filterSize: 3, strides: 2, padding: "same" }],
  [[2, 6, 5, 3], { filterSize: [2, 3], strides: [1, 2] }],
  [[1, 7, 7, 2], { filterSize: 3, strides: 1 }],
  [[1, 4, 4, 1], { filterSize: 2, strides: 1, padding: "same" }],
];

/** @returns {Case[]} */
function poolingCases() {
  const cases = [];
  for (const [kind, method] of [
    ["max", "maxPool"],
    ["avg", "avgPool"],
  ])
    for (const [shape, options] of POOLS) {
      // Rounded to a few digits so that equal elements occur, which a maximum must break the same
      // way on every tier.
      const input = values(nextSeed(), sizeOf(shape), -2, 2).map((v) =>
        f32(Math.round(v * 4) / 4),
      );
      const { out, outShape } = poolReference(kind, input, shape, options);
      cases.push({
        name: `${method} input [${shape}], ${JSON.stringify(options)}`,
        tolerance: 0,
        gpuTolerance: kind === "avg" ? 1e-5 : 0,
        run: async (_nn, device) => ({
          actual: await device.tensor(input, { shape })[method](options).data(),
          expected: out,
        }),
      });
      const dy = values(nextSeed(), sizeOf(outShape), -1, 1);
      cases.push({
        name: `gradient of ${method} input [${shape}], ${JSON.stringify(options)}`,
        tolerance: 1e-5,
        gpuTolerance: 1e-5,
        run: async (nn, device) => {
          const x = device.tensor(input, { shape });
          const [actual] = nn.gradients(
            x[method](options),
            [x],
            device.tensor(dy, { shape: outShape }),
          );
          return {
            actual: await actual.data(),
            expected: poolGradReference(kind, input, shape, options, dy),
          };
        },
      });
      cases.push({
        name: `${method}Grad input [${shape}], ${JSON.stringify(options)}`,
        tolerance: 1e-5,
        gpuTolerance: 1e-5,
        run: async (_nn, device) => ({
          actual: await device
            .tensor(input, { shape })
            [`${method}Grad`](device.tensor(dy, { shape: outShape }), options)
            .data(),
          expected: poolGradReference(kind, input, shape, options, dy),
        }),
      });
    }
  return cases;
}

export function spatialCases() {
  seed = 4000;
  return [...convolutionCases(), ...poolingCases()];
}
