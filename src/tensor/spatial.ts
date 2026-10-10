export interface Conv2DOptions {
  readonly strides?: number | readonly [number, number];
  readonly padding?: "valid" | "same";
}

export interface Pool2DOptions {
  readonly filterSize: number | readonly [number, number];
  readonly strides?: number | readonly [number, number];
  readonly padding?: "valid" | "same";
}

export interface SpatialWindow {
  readonly outputHeight: number;
  readonly outputWidth: number;
  readonly strideY: number;
  readonly strideX: number;
  readonly padTop: number;
  readonly padLeft: number;
}

type SpatialOptions = Conv2DOptions | Pool2DOptions;

export interface PoolingWindow extends SpatialWindow {
  readonly filterHeight: number;
  readonly filterWidth: number;
}

/** Resolve output dimensions and leading padding for NHWC spatial operations. */
export function spatialWindow(
  height: number,
  width: number,
  filterHeight: number,
  filterWidth: number,
  options: SpatialOptions,
  operation: string,
): SpatialWindow {
  const [strideY, strideX] =
    typeof options.strides === "number"
      ? [options.strides, options.strides]
      : (options.strides ?? [1, 1]);
  if (
    !Number.isSafeInteger(strideY) ||
    strideY < 1 ||
    !Number.isSafeInteger(strideX) ||
    strideX < 1
  )
    throw new RangeError(
      `${operation} strides must be positive safe integers.`,
    );

  const padding = options.padding ?? "valid";
  if (padding !== "valid" && padding !== "same")
    throw new TypeError(
      `Unknown ${operation} padding mode: ${String(padding)}.`,
    );

  const outputHeight =
    padding === "same"
      ? Math.ceil(height / strideY)
      : Math.floor((height - filterHeight) / strideY) + 1;
  const outputWidth =
    padding === "same"
      ? Math.ceil(width / strideX)
      : Math.floor((width - filterWidth) / strideX) + 1;
  if (outputHeight < 1 || outputWidth < 1)
    throw new RangeError(
      `${operation} filter must fit the input for valid padding.`,
    );

  const totalPadHeight = Math.max(
    (outputHeight - 1) * strideY + filterHeight - height,
    0,
  );
  const totalPadWidth = Math.max(
    (outputWidth - 1) * strideX + filterWidth - width,
    0,
  );
  return {
    outputHeight,
    outputWidth,
    strideY,
    strideX,
    padTop: Math.floor(totalPadHeight / 2),
    padLeft: Math.floor(totalPadWidth / 2),
  };
}

export function poolingWindow(
  height: number,
  width: number,
  options: Pool2DOptions,
  operation: string,
): PoolingWindow {
  const [filterHeight, filterWidth] =
    typeof options.filterSize === "number"
      ? [options.filterSize, options.filterSize]
      : options.filterSize;
  if (
    !Number.isSafeInteger(filterHeight) ||
    filterHeight < 1 ||
    !Number.isSafeInteger(filterWidth) ||
    filterWidth < 1
  )
    throw new RangeError(
      `${operation} filterSize values must be positive safe integers.`,
    );
  return {
    filterHeight,
    filterWidth,
    ...spatialWindow(
      height,
      width,
      filterHeight,
      filterWidth,
      options,
      operation,
    ),
  };
}
