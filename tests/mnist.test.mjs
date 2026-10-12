import test, { before } from "node:test";
import assert from "node:assert/strict";
import { build } from "esbuild";
import { pathToFileURL } from "node:url";
import { join } from "node:path";
import { syntheticMnist } from "./_mnist.mjs";

// The MNIST example (`examples/mnist/loop.ts`) on the CPU tier in Node. It is bundled here with
// `typenn` left as it is: the bundle imports the built library through the package's own name, so
// the example runs against the library it documents. The data is synthetic (`_mnist.mjs`).

const root = join(import.meta.dirname, "..");
let train;
before(async () => {
  const outfile = join(root, ".typenn-build", "mnist", "loop.mjs");
  await build({
    entryPoints: [join(root, "examples", "mnist", "loop.ts")],
    outfile,
    bundle: true,
    format: "esm",
    platform: "node",
    external: ["typenn"],
    logLevel: "silent",
  });
  ({ train } = await import(pathToFileURL(outfile).href));
});

const mean = (values) =>
  values.reduce((sum, value) => sum + value, 0) / values.length;

test("the MNIST MLP trains on the CPU tier: its loss falls and it learns the classes", async () => {
  const trainData = syntheticMnist(320, 21);
  const testData = syntheticMnist(100, 22);
  const lines = [];
  const result = await train(trainData, testData, {
    device: "cpu",
    batchSize: 32,
    steps: 30,
    shuffle: false,
    log: (line) => lines.push(line),
  });
  assert.deepEqual(result.backend, { provider: "typeshade", name: "cpu" });
  assert.equal(result.losses.length, 30);
  assert.ok(result.losses.every(Number.isFinite));
  assert.ok(
    Math.abs(result.losses[0] - Math.log(10)) < 0.5,
    `the first loss ${result.losses[0]}`,
  );
  assert.ok(
    mean(result.losses.slice(-5)) < 0.7 * mean(result.losses.slice(0, 5)),
    `the loss went from ${mean(result.losses.slice(0, 5))} to ${mean(result.losses.slice(-5))}`,
  );
  assert.ok(result.accuracy > 0.6, `the test accuracy ${result.accuracy}`);
  assert.equal(lines[0], "TypeNN on cpu");
  // The CPU tier runs the plan one call at a time: no dispatch and no submission.
  assert.equal(result.lastReport.tier, "cpu");
  assert.equal(result.lastReport.dispatches, 0);
});
