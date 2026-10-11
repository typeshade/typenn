export interface MNISTSplit {
  readonly images: Uint8Array;
  readonly labels: Uint8Array;
  readonly count: number;
  readonly rows: number;
  readonly columns: number;
}

export interface MNISTBatch {
  readonly images: Float32Array;
  readonly labels: Uint32Array;
  readonly size: number;
}

const MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist";

async function downloadGzip(fileName: string): Promise<ArrayBuffer> {
  const response = await fetch(`${MNIST_BASE_URL}/${fileName}.gz`);
  if (!response.ok)
    throw new Error(`MNIST download failed (${response.status}): ${fileName}`);
  const stream = response.body?.pipeThrough(new DecompressionStream("gzip"));
  if (!stream) throw new Error("This runtime does not support gzip streams.");
  return new Response(stream).arrayBuffer();
}

export async function loadMNIST(train: boolean): Promise<MNISTSplit> {
  const prefix = train ? "train" : "t10k";
  const [imageBuffer, labelBuffer] = await Promise.all([
    downloadGzip(`${prefix}-images-idx3-ubyte`),
    downloadGzip(`${prefix}-labels-idx1-ubyte`),
  ]);
  const imageHeader = new DataView(imageBuffer);
  const labelHeader = new DataView(labelBuffer);
  if (imageHeader.getUint32(0) !== 2051 || labelHeader.getUint32(0) !== 2049)
    throw new Error("Downloaded MNIST files have invalid IDX headers.");
  const count = imageHeader.getUint32(4);
  const rows = imageHeader.getUint32(8);
  const columns = imageHeader.getUint32(12);
  if (labelHeader.getUint32(4) !== count)
    throw new Error("MNIST image and label counts do not match.");
  return {
    images: new Uint8Array(imageBuffer, 16),
    labels: new Uint8Array(labelBuffer, 8),
    count,
    rows,
    columns,
  };
}

export class DataLoader implements AsyncIterable<MNISTBatch> {
  constructor(
    private readonly dataset: MNISTSplit,
    readonly batchSize: number,
    readonly shuffle: boolean,
  ) {
    if (!Number.isSafeInteger(batchSize) || batchSize < 1)
      throw new RangeError("batchSize must be a positive integer.");
  }

  async *[Symbol.asyncIterator](): AsyncIterator<MNISTBatch> {
    const order = Uint32Array.from({ length: this.dataset.count }, (_, i) => i);
    if (this.shuffle) shuffle(order);
    const pixelsPerImage = this.dataset.rows * this.dataset.columns;
    for (let offset = 0; offset < order.length; offset += this.batchSize) {
      const size = Math.min(this.batchSize, order.length - offset);
      const images = new Float32Array(size * pixelsPerImage);
      const labels = new Uint32Array(size);
      for (let batchIndex = 0; batchIndex < size; batchIndex++) {
        const imageIndex = order[offset + batchIndex]!;
        const imageOffset = imageIndex * pixelsPerImage;
        for (let pixel = 0; pixel < pixelsPerImage; pixel++)
          images[batchIndex * pixelsPerImage + pixel] =
            this.dataset.images[imageOffset + pixel]! / 255;
        labels[batchIndex] = this.dataset.labels[imageIndex]!;
      }
      yield { images, labels, size };
    }
  }
}

function shuffle(values: Uint32Array): void {
  for (let index = values.length - 1; index > 0; index--) {
    const other = Math.floor(Math.random() * (index + 1));
    [values[index], values[other]] = [values[other]!, values[index]!];
  }
}
