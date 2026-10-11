import * as nn from "typenn";
import { DataLoader, loadMNIST } from "./data.js";
import { SimpleMLP } from "./model.js";

const batchSize = 64;
const epochs = 5;
const device = await nn.device({ device: "auto" });
const model = new SimpleMLP().to(device);
const parameters = model.parameters();
const criterion = new nn.CrossEntropyLoss();
const optimizer = nn.optim.Adam(parameters, { lr: 0.001 });

try {
  const trainData = await loadMNIST(true);
  const testData = await loadMNIST(false);

  for (let epoch = 1; epoch <= epochs; epoch++) {
    model.train();
    let totalLoss = 0;
    let totalSamples = 0;
    for await (const batch of new DataLoader(trainData, batchSize, true)) {
      await optimizer.zeroGrad();
      const images = device.tensor(batch.images, {
        shape: [batch.size, 1, 28, 28],
      });
      const logits = model.forward(images);
      const loss = criterion.forward(logits, batch.labels);
      const [batchLoss] = await loss.data();
      await loss.backward();
      await optimizer.step();
      totalLoss += batchLoss! * batch.size;
      totalSamples += batch.size;
      await loss.disposeGraph(parameters);
    }
    console.log(
      `Epoch ${epoch}/${epochs}, loss ${(totalLoss / totalSamples).toFixed(4)}`,
    );
  }

  model.eval();
  let correct = 0;
  let total = 0;
  for await (const batch of new DataLoader(testData, batchSize, false)) {
    const images = device.tensor(batch.images, {
      shape: [batch.size, 1, 28, 28],
    });
    const logits = model.forward(images);
    const values = await logits.data();
    for (let row = 0; row < batch.size; row++) {
      let predicted = 0;
      for (let digit = 1; digit < 10; digit++) {
        if (values[row * 10 + digit]! > values[row * 10 + predicted]!)
          predicted = digit;
      }
      if (predicted === batch.labels[row]) correct++;
    }
    total += batch.size;
    await logits.disposeGraph(parameters);
  }
  console.log(`Test accuracy ${((100 * correct) / total).toFixed(2)}%`);
} finally {
  await optimizer.zeroGrad();
  await device.dispose();
}
