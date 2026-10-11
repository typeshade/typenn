import type {
  FusionFixture,
  FusionFixtureRef,
} from "./verify-elementwise-fusion-oracle.js";

export interface FusionTensorLike {
  add(other: FusionTensorLike): FusionTensorLike;
  mul(other: FusionTensorLike): FusionTensorLike;
  relu(): FusionTensorLike;
  data(): Promise<ArrayLike<number>>;
}

export interface FusionRuntimeLike {
  tensor(
    values: ArrayLike<number>,
    options: { readonly shape: readonly number[] },
  ): FusionTensorLike;
}

export interface FusionFixtureCase {
  readonly id: string;
  readonly fixture: FusionFixture;
  readonly inputs: Readonly<Record<string, readonly number[]>>;
}

const input = (name: string): FusionFixtureRef => ({ input: name });
const value = (name: string): FusionFixtureRef => ({ value: name });

export const FUSION_CASES: readonly FusionFixtureCase[] = [
  {
    id: "add-relu",
    fixture: {
      nodes: [
        { id: "sum", op: "add", inputs: [input("a"), input("b")] },
        { id: "out", op: "relu", inputs: [value("sum")] },
      ],
      output: value("out"),
    },
    inputs: { a: [2, -4, 0.5], b: [3, 1, -0.25] },
  },
  {
    id: "add-mul-relu",
    fixture: {
      nodes: [
        { id: "sum", op: "add", inputs: [input("a"), input("b")] },
        { id: "product", op: "mul", inputs: [value("sum"), input("c")] },
        { id: "out", op: "relu", inputs: [value("product")] },
      ],
      output: value("out"),
    },
    inputs: {
      a: [1, 2, -3, 4],
      b: [5, -3, 2, -8],
      c: [2, 1, -2, 0.5],
    },
  },
  {
    id: "mul-add-relu",
    fixture: {
      nodes: [
        { id: "product", op: "mul", inputs: [input("a"), input("b")] },
        { id: "sum", op: "add", inputs: [value("product"), input("c")] },
        { id: "out", op: "relu", inputs: [value("sum")] },
      ],
      output: value("out"),
    },
    inputs: {
      a: [2, -3, 4, -2],
      b: [3, 2, -1, -4],
      c: [-1, 4, 5, -9],
    },
  },
  {
    id: "reordered-four-op",
    fixture: {
      nodes: [
        { id: "positive", op: "relu", inputs: [input("a")] },
        { id: "product", op: "mul", inputs: [value("positive"), input("b")] },
        { id: "sum", op: "add", inputs: [value("product"), input("c")] },
        { id: "out", op: "relu", inputs: [value("sum")] },
      ],
      output: value("out"),
    },
    inputs: { a: [-2, 3, -1, 4], b: [9, 2, 5, -3], c: [1, -1, 2, 0.5] },
  },
  {
    id: "repeated-add-mul",
    fixture: {
      nodes: [
        { id: "sum0", op: "add", inputs: [input("a"), input("b")] },
        { id: "product0", op: "mul", inputs: [value("sum0"), input("a")] },
        { id: "sum1", op: "add", inputs: [value("product0"), input("c")] },
        { id: "product1", op: "mul", inputs: [value("sum1"), input("b")] },
        { id: "out", op: "relu", inputs: [value("product1")] },
      ],
      output: value("out"),
    },
    inputs: {
      a: [1, -2, 3, 4],
      b: [2, 5, -1, 3],
      c: [1, -2, 4, -5],
    },
  },
  {
    id: "repeated-leaf",
    fixture: {
      nodes: [
        { id: "twice", op: "add", inputs: [input("a"), input("a")] },
        { id: "product", op: "mul", inputs: [value("twice"), input("a")] },
        { id: "out", op: "relu", inputs: [value("product")] },
      ],
      output: value("out"),
    },
    inputs: { a: [2, -3, 0, 4] },
  },
  {
    id: "diamond-shared-node",
    fixture: {
      nodes: [
        { id: "shared", op: "add", inputs: [input("a"), input("b")] },
        { id: "left", op: "mul", inputs: [value("shared"), input("a")] },
        { id: "right", op: "mul", inputs: [value("shared"), input("b")] },
        { id: "join", op: "add", inputs: [value("left"), value("right")] },
        { id: "out", op: "relu", inputs: [value("join")] },
      ],
      output: value("out"),
    },
    inputs: { a: [2, 1, 3], b: [3, -4, 2] },
  },
];

export function buildFusionFixture(
  runtime: FusionRuntimeLike,
  testCase: FusionFixtureCase,
  shapeLength = Object.values(testCase.inputs)[0]!.length,
) {
  const leaves = new Map<string, FusionTensorLike>();
  for (const [name, values] of Object.entries(testCase.inputs))
    leaves.set(name, runtime.tensor(values, { shape: [shapeLength] }));

  const values = new Map<string, FusionTensorLike>();
  const resolve = (reference: FusionFixtureRef): FusionTensorLike => {
    const result =
      "input" in reference
        ? leaves.get(reference.input)
        : values.get(reference.value);
    if (!result)
      throw new Error(
        `Unknown fixture reference ${JSON.stringify(reference)}.`,
      );
    return result;
  };
  for (const node of testCase.fixture.nodes) {
    const args = node.inputs.map(resolve);
    let result: FusionTensorLike;
    if (node.op === "relu") {
      if (args.length !== 1)
        throw new Error("relu test case must have one input.");
      result = args[0]!.relu();
    } else {
      if (args.length !== 2)
        throw new Error(`${node.op} test case must have two inputs.`);
      result =
        node.op === "add" ? args[0]!.add(args[1]!) : args[0]!.mul(args[1]!);
    }
    values.set(node.id, result);
  }
  return { leaves, values, output: resolve(testCase.fixture.output) };
}
