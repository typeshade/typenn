import { random } from "./_reference.mjs";

/**
 * A stand-in for an MNIST split (`examples/mnist/data.ts`): 28 x 28 images of ten classes, in the
 * order of the labels 0 to 9 and round. Class `c` is a bright block at its own place on a noisy
 * background, so a small network learns it in a few steps and the test needs no download.
 */
export function syntheticMnist(count, seed) {
  const next = random(seed);
  const images = new Uint8Array(count * 28 * 28);
  const labels = new Uint8Array(count);
  for (let sample = 0; sample < count; sample++) {
    const label = sample % 10;
    labels[sample] = label;
    for (let pixel = 0; pixel < 28 * 28; pixel++)
      images[sample * 784 + pixel] = Math.floor(next() * 40);
    const left = 2 + (label % 5) * 5;
    const top = 3 + Math.floor(label / 5) * 12;
    for (let y = top; y < top + 8; y++)
      for (let x = left; x < left + 5; x++)
        images[sample * 784 + y * 28 + x] = 200 + Math.floor(next() * 56);
  }
  return { images, labels, count, rows: 28, columns: 28 };
}
