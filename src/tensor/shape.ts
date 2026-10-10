export type Shape = readonly number[];

export const MAX_BROADCAST_RANK = 8;

export interface BroadcastLayout {
  readonly shape: number[];
  /** Axis metadata is stored from innermost to outermost for the TypeShade kernel. */
  readonly dimensions: number[];
  readonly aStrides: number[];
  readonly bStrides: number[];
}

export function checkedShape(shape: Shape): number[] {
  if (!Array.isArray(shape))
    throw new TypeError("Tensor shape must be an array.");
  const copy = [...shape];
  for (const [axis, size] of copy.entries()) {
    if (!Number.isSafeInteger(size) || size < 1)
      throw new RangeError(
        `Shape dimension ${axis} must be a positive safe integer.`,
      );
  }
  return copy;
}

export function sizeOf(shape: Shape): number {
  const size = shape.reduce((total, dimension) => total * dimension, 1);
  if (!Number.isSafeInteger(size))
    throw new RangeError(
      "Tensor element count exceeds the safe integer range.",
    );
  return size;
}

export function sameShape(a: Shape, b: Shape): boolean {
  return (
    a.length === b.length &&
    a.every((dimension, index) => dimension === b[index])
  );
}

export function broadcastLayout(a: Shape, b: Shape): BroadcastLayout {
  const rank = Math.max(a.length, b.length);
  if (rank > MAX_BROADCAST_RANK)
    throw new RangeError(
      `Binary operations support broadcasting up to rank ${MAX_BROADCAST_RANK}; got rank ${rank}.`,
    );

  const aOffset = rank - a.length;
  const bOffset = rank - b.length;
  const output = new Array<number>(rank);
  const aStrides = new Array<number>(rank);
  const bStrides = new Array<number>(rank);
  let aStride = 1;
  let bStride = 1;

  for (let axis = rank - 1; axis >= 0; axis--) {
    const aAxis = axis - aOffset;
    const bAxis = axis - bOffset;
    const aDimension = aAxis < 0 ? 1 : a[aAxis]!;
    const bDimension = bAxis < 0 ? 1 : b[bAxis]!;
    if (aDimension !== bDimension && aDimension !== 1 && bDimension !== 1)
      throw new RangeError(
        `Binary operation shapes ${shapeLabel(a)} and ${shapeLabel(b)} are not broadcast-compatible.`,
      );

    const outputDimension = Math.max(aDimension, bDimension);
    output[axis] = outputDimension;
    aStrides[axis] =
      aAxis < 0 || (aDimension === 1 && outputDimension !== 1) ? 0 : aStride;
    bStrides[axis] =
      bAxis < 0 || (bDimension === 1 && outputDimension !== 1) ? 0 : bStride;
    aStride *= aDimension;
    bStride *= bDimension;
  }

  const padReverse = (values: readonly number[], fill: number) => [
    ...values.slice().reverse(),
    ...new Array<number>(MAX_BROADCAST_RANK - values.length).fill(fill),
  ];
  sizeOf(output);
  return {
    shape: output,
    dimensions: padReverse(output, 1),
    aStrides: padReverse(aStrides, 0),
    bStrides: padReverse(bStrides, 0),
  };
}

export function shapeLabel(shape: Shape): string {
  return `[${shape.join(", ")}]`;
}
