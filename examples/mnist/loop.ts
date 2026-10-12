import * as nn from "typenn";
import { DataLoader, type MNISTSplit } from "./data.js";
import { SimpleMLP } from "./model.js";

export interface TrainOptions {
  /** Where the kernel functions run: `auto` (WebGPU, else the CPU tier), `webgpu`, `webgl2`, `cpu`. */
  readonly device?: nn.DevicePreference;
  readonly batchSize?: number;
  readonly epochs?: number;
  /** Stop after this many training steps in all. */
  readonly steps?: number;
  /** Visit the training images in a random order. On by default. */
  readonly shuffle?: boolean;
  /** Evaluate on this many test images, from the first. All of them by default. */
  readonly testSamples?: number;
  readonly log?: (line: string) => void;
}

export interface TrainResult {
  readonly backend: nn.BackendInfo;
  /** The loss of each training step. */
  readonly losses: readonly number[];
  /** Correct predictions over the test images, from 0 to 1. */
  readonly accuracy: number;
  /** What TypeShade's fuse scope did for the last `data()` of the last training step. */
  readonly lastReport: nn.FuseReport | undefined;
}

/**
 * Train the MLP of `model.ts` as PyTorch trains one: `zeroGrad()`, forward, loss, `backward()`,
 * `step()`. Every `data()` runs as one TypeShade fuse scope, so the report of the device says what
 * the plan of a step was.
 */
export async function train(
  trainData: MNISTSplit,
  testData: MNISTSplit,
  options: TrainOptions = {},
): Promise<TrainResult> {
  const { batchSize = 64, epochs = 5, log = () => {} } = options;
  const device = await nn.device({ device: options.device ?? "auto" });
  const model = new SimpleMLP().to(device);
  const parameters = model.parameters();
  const criterion = new nn.CrossEntropyLoss();
  const optimizer = nn.optim.Adam(parameters, { lr: 0.001 });
  const losses: number[] = [];
  let accuracy = 0;
  let lastReport: nn.FuseReport | undefined;

  try {
    log(`TypeNN on ${device.backend.name}`);
    training: for (let epoch = 1; epoch <= epochs; epoch++) {
      model.train();
      let totalLoss = 0;
      let totalSamples = 0;
      for await (const batch of new DataLoader(
        trainData,
        batchSize,
        options.shuffle ?? true,
      )) {
        await optimizer.zeroGrad();
        const images = device.tensor(batch.images, {
          shape: [batch.size, 1, 28, 28],
        });
        const logits = model.forward(images);
        const loss = criterion.forward(logits, batch.labels);
        const [batchLoss] = await loss.data();
        await loss.backward();
        await optimizer.step();
        lastReport = device.lastReport;
        losses.push(batchLoss!);
        totalLoss += batchLoss! * batch.size;
        totalSamples += batch.size;
        await loss.disposeGraph(parameters);
        if (losses.length === options.steps) {
          log(
            `Epoch ${epoch}/${epochs}, ${totalSamples} samples, loss ${(totalLoss / totalSamples).toFixed(4)}`,
          );
          break training;
        }
      }
      log(
        `Epoch ${epoch}/${epochs}, loss ${(totalLoss / totalSamples).toFixed(4)}`,
      );
    }

    model.eval();
    let correct = 0;
    let total = 0;
    const limit = options.testSamples ?? testData.count;
    for await (const batch of new DataLoader(testData, batchSize, false)) {
      if (total >= limit) break;
      const size = Math.min(batch.size, limit - total);
      const images = device.tensor(batch.images.subarray(0, size * 784), {
        shape: [size, 1, 28, 28],
      });
      const logits = model.forward(images);
      const values = await logits.data();
      for (let row = 0; row < size; row++) {
        let predicted = 0;
        for (let digit = 1; digit < 10; digit++) {
          if (values[row * 10 + digit]! > values[row * 10 + predicted]!)
            predicted = digit;
        }
        if (predicted === batch.labels[row]) correct++;
      }
      total += size;
      await logits.disposeGraph(parameters);
    }
    accuracy = correct / total;
    log(`Test accuracy ${(100 * accuracy).toFixed(2)}%`);
  } finally {
    await optimizer.zeroGrad();
    await device.dispose();
  }
  return { backend: device.backend, losses, accuracy, lastReport };
}
