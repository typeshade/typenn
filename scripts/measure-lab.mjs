// Measures the Lab's training path on WebGPU in headless Chromium, for two builds of the Lab:
// the old one (host batches) and the device-resident one (batches built on the device). It
// reports, per dataset, the bytes uploaded per steady-state step, the median time of a step after
// warm-up, and the one-time upload of the dataset.
//
//   npm run build
//   TYPESHADE_CHROMIUM=/path/to/chrome node scripts/measure-lab.mjs \
//     --main /path/to/main-checkout [--datasets mnist,cifar10] [--steps 50] [--warmup 10] [--out file.json]
//
// `--main` is a checkout of the old build with its `dist/` built (`npm run build`) and its
// `website/src/lab/train.ts`. The current checkout is the new build. A dataset is synthetic, with
// the shape of the Lab dataset; the timing does not depend on the values. The browser's WebGPU is
// SwiftShader here, so a time is a software-device time and not a hardware GPU benchmark.

import { mkdir, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { build } from "esbuild";
import { inPage } from "../tests/gpu/_browser.mjs";

const root = join(import.meta.dirname, "..");
const { values: flags } = parseArgs({
  options: {
    main: { type: "string" },
    datasets: { type: "string", default: "mnist,cifar10" },
    steps: { type: "string", default: "50" },
    warmup: { type: "string", default: "10" },
    out: { type: "string" },
  },
});
if (!flags.main) throw new Error("Pass --main <checkout of the old build>.");
const mainRoot = resolve(flags.main);
const steps = Number(flags.steps);
const warmup = Number(flags.warmup);

const trees = [
  { label: "main", root: mainRoot, dist: join(mainRoot, "dist") },
  { label: "branch", root: root, dist: join(root, "dist") },
];

async function bundle(label, labRoot) {
  const dir = join(root, ".typenn-build", "measure-lab");
  await mkdir(dir, { recursive: true });
  const entry = join(dir, `${label}-entry.ts`);
  await writeFile(
    entry,
    `import * as lab from ${JSON.stringify(join(labRoot, "website", "src", "lab", "train.ts"))};\n` +
      `import * as nn from "typenn";\n` +
      `import { bench } from ${JSON.stringify(join(root, "scripts", "measure-lab-page.ts"))};\n` +
      `export const run = (options: any) => bench(lab, nn, options);\n`,
  );
  const outfile = join(dir, `${label}.mjs`);
  await build({
    entryPoints: [entry],
    outfile,
    bundle: true,
    format: "esm",
    platform: "browser",
    external: ["typenn"],
    logLevel: "silent",
  });
  return `/.typenn-build/measure-lab/${label}.mjs`;
}

// The body runs in the page. It has no closure: it imports the bundle named in its argument.
const body = async (argument) => {
  const module = await import(argument.url);
  return module.run(argument);
};

const results = [];
for (const tree of trees) {
  const url = await bundle(tree.label, tree.root);
  for (const dataset of flags.datasets.split(",")) {
    const { result, logged } = await inPage(
      body,
      { url, dataset, steps, warmup },
      { dist: tree.dist },
    );
    const errors = logged.filter(
      (line) => line.startsWith("pageerror") || line.startsWith("error"),
    );
    if (errors.length)
      throw new Error(`${tree.label}/${dataset}: ${errors.join("; ")}`);
    results.push({ build: tree.label, ...result });
    console.log(
      `${tree.label} ${dataset}: median ${result.medianMs.toFixed(1)} ms/step, ` +
        `${result.bytesPerStep.toFixed(0)} B/step, one-time ${result.oneTime ? `${result.oneTime.bytes} B in ${(result.oneTime.seconds * 1000).toFixed(0)} ms` : "none (host batches)"}`,
    );
  }
}
if (flags.out)
  await writeFile(resolve(flags.out), JSON.stringify(results, null, 2) + "\n");
console.log(JSON.stringify(results, null, 2));
