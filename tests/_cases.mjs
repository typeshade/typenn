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
  SCALED_MAPS,
  binaryGradReference,
  binaryReference,
  broadcastShape,
  f32,
  sizeOf,
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
