/** Test-only f32 DAG model. It consumes authored fixture specs, never product IR/cache data. */
export type FusionFixtureRef =
  { readonly input: string } | { readonly value: string };

export interface FusionFixtureNode {
  readonly id: string;
  readonly op: "add" | "mul" | "relu";
  readonly inputs: readonly FusionFixtureRef[];
}

export interface FusionFixture {
  readonly nodes: readonly FusionFixtureNode[];
  readonly output: FusionFixtureRef;
}

/** Evaluate each authored DAG node with f32 rounding, independently of TypeNN and TypeShade. */
export function evaluateFusionFixtureF32(
  fixture: FusionFixture,
  inputs: Readonly<Record<string, ArrayLike<number>>>,
): Float32Array {
  const first = Object.values(inputs)[0];
  if (!first) throw new Error("A fusion fixture needs at least one input.");
  const length = first.length;
  for (const [name, values] of Object.entries(inputs))
    if (values.length !== length)
      throw new Error(`Fixture input ${name} has inconsistent length.`);

  const result = new Float32Array(length);
  for (let index = 0; index < length; index++) {
    const values = new Map<string, number>();
    const read = (reference: FusionFixtureRef): number => {
      if ("input" in reference) {
        const input = inputs[reference.input];
        if (!input)
          throw new Error(`Fixture input ${reference.input} is missing.`);
        return Math.fround(input[index]!);
      }
      const value = values.get(reference.value);
      if (value === undefined)
        throw new Error(
          `Fixture value ${reference.value} is used before definition.`,
        );
      return value;
    };

    for (const node of fixture.nodes) {
      const args = node.inputs.map(read);
      let value: number;
      if (node.op === "relu") {
        if (args.length !== 1) throw new Error("relu fixtures need one input.");
        value = Math.max(0, args[0]!);
      } else {
        if (args.length !== 2)
          throw new Error(`${node.op} fixtures need two inputs.`);
        value = node.op === "add" ? args[0]! + args[1]! : args[0]! * args[1]!;
      }
      values.set(node.id, Math.fround(value));
    }
    result[index] = Math.fround(read(fixture.output));
  }
  return result;
}

export function assertFloat32ArraysEqual(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  label: string,
): void {
  if (actual.length !== expected.length)
    throw new Error(`${label}: length ${actual.length} != ${expected.length}`);
  for (let index = 0; index < expected.length; index++) {
    const actualValue = Math.fround(actual[index]!);
    const expectedValue = Math.fround(expected[index]!);
    if (!Object.is(actualValue, expectedValue))
      throw new Error(
        `${label}[${index}]: f32 ${actualValue} != ${expectedValue}`,
      );
  }
}
