# TypeNN website

This is the source for [nn.typeshade.dev](https://nn.typeshade.dev). It is an Astro site
with React islands, TailwindCSS v4, and Ant Design v5. TensorFlow.js is used as an information
architecture reference; the site's product claims and visual system are TypeNN-specific.

## Develop

```sh
npm ci
npm run dev
```

English routes are served under `/`; the separately written Korean routes are under `/ko/`.
Run `npm run build` to prerender the full route set into `dist/`.

## Browser Playground runtime

The Playground uses the actual compiled TypeNN package. Its ESM bundle is vendored under
`src/vendor/typenn/`, with the source commit and Apache-2.0 license recorded alongside it. This
lets the static site run without an npm package release or remote CDN at demo time.

To refresh it from a local TypeNN checkout:

```sh
cd /path/to/typenn
npm ci
npm run build
cd website
npm ci
npm run build
```

The website prebuild syncs TypeNN's compiled JavaScript bundle. `sync:typenn` also works directly;
set `TYPE_NN_ROOT` when the TypeNN checkout is not the website's parent or adjacent checkout. The
Playground supports small 2×2 MatMul, Add, and MatMul→ReLU operations. It reads the displayed
result with `Tensor.data()` and releases the graph afterward. It is a correctness demonstration,
not a performance benchmark.

## Datasets

The lab trains TypeNN models in the browser on the Keras datasets. Five of them download directly
from the Keras dataset storage. The browser uses the JSON API URL of each file, because the XML URL
sends no CORS headers and the JSON API URL does. `src/lab/keras.ts` does the work. It checks each
file's SHA-256, caches the file in the Cache API and decodes it. The decoder reads `.npz` and
`.npy` files, the pickle of numpy object arrays, and the IDX `.gz` files. It also repeats the Keras
split with numpy's `RandomState` shuffle.

CIFAR-10 and CIFAR-100 stay mirrored on this site. Their only source, cs.toronto.edu, does not allow
cross-origin downloads. `scripts/fetch_datasets.py` builds the mirror under `public/data/`, and
Cloudflare serves it at `/data/`. The deploy builds only the two CIFAR datasets:

```sh
npm run datasets                              # cifar10 and cifar100, into public/data/
python3 scripts/fetch_datasets.py --all       # all seven, for the parity test
python3 scripts/fetch_datasets.py --only mnist
```

The script needs Python 3 and numpy. It downloads each source to `.dataset-cache/` (skipped when
present), checks its SHA-256 against a table in the script, and writes per dataset a
`manifest.json` and binary parts. `public/data/index.json` lists the mirrored datasets. Cloudflare
limits one static asset to 25 MiB, so the script keeps every output file at 20 MiB or less and
splits larger arrays into `.partN.bin` files.

`npm run test:keras` checks the browser decoder against that mirror. It reads the source files
from the cache and the parts from `public/data/`, and it needs both. It decodes each of the five
datasets and compares every array byte for byte. Override the directories with
`TYPENN_DATASET_CACHE` and `TYPENN_DATASET_MIRROR`.

| Dataset             | Source                                             | License and citation                                                                                                               |
| ------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| MNIST               | `mnist.npz` (Keras storage, direct)                | CC BY-SA 3.0, per the Keras documentation. LeCun, Cortes and Burges.                                                               |
| Fashion-MNIST       | the four IDX files (Keras storage, direct)         | MIT, Zalando SE. Xiao, Rasul and Vollgraf, 2017.                                                                                   |
| CIFAR-10, CIFAR-100 | the binary editions (mirrored from cs.toronto.edu) | No explicit license. Krizhevsky, 2009 (tech report; cite it).                                                                      |
| IMDB                | `imdb.npz` (Keras storage, direct)                 | No explicit license that we know of. Maas et al., ACL 2011. Vocabulary capped at 10000 words.                                      |
| Reuters             | `reuters.npz` (Keras storage, direct)              | Reuters-21578: free distribution for research purposes, per its readme (not a standard license). Vocabulary capped at 10000 words. |
| California housing  | `california_housing.npz` (Keras storage, direct)   | StatLib, Pace and Barry, 1997. No explicit license found.                                                                          |

The text datasets use the Keras loader's split, seed 113 and token encoding. Their token ids
are `uint16` and the vocabulary is capped at 10000 words for the bag-of-words input.

## Training lab

`/lab/` and `/ko/lab/` train TypeNN models in the browser on the full training split of each
dataset. The page loads the five Keras datasets from the Keras storage and the CIFAR mirror from
`/data/`, trains with `Module`, `Sequential`, `Linear`, `Conv2d`
and `MaxPool2d` (site-defined over `Tensor.conv2d` and `Tensor.maxPool`), the losses and the Adam
or SGD optimizer, and evaluates on the full test split after each epoch. `src/lab/train.ts` holds
the training loop, `src/lab/datasets.ts` the loader for the mirror, `src/lab/keras.ts` the loader for the Keras storage, and
`src/components/Lab.tsx` the page.

The CNN preset uses `Conv2d` and `MaxPool2d`. The TypeNN declarations do not export them, so
`src/lab/train.ts` defines both as `Module` subclasses. It creates their trainable leaves with the
runtime's `createParameter`, as TypeNN's own `Linear` does.

### Lab automation

The page exposes `window.__typennLab`. Its `run` method uses the same code path as the Start
button, and it updates the page too:

```ts
interface LabApi {
  ready: Promise<void>; // resolves when the island has mounted
  datasets(): Promise<DatasetInfo[]>; // the five Keras datasets, then the entries of /data/index.json
  run(config: Partial<RunConfig> & { dataset: string }): Promise<RunResult>; // missing fields take the dataset defaults
  stop(): void; // stops after the current step; the result has completed: false
  readonly results: RunResult[]; // the finished and stopped runs of this page, oldest first
}
```

A `RunResult` holds `config`, `tier`, `epochs` (`trainLoss` and `test` for each epoch), `steps`,
`samplesSeen`, `seconds`, `test`, `completed`, `userAgent`, `typennCommit` and `finishedAt`.
`typennCommit` comes from `src/vendor/typenn/version.json`, which `npm run sync:typenn` writes.

## Project documents

`PRODUCT.md` and `DESIGN.md` describe verified capabilities, current limits, the page structure,
and the shared IBM Plex/Ant Design visual system. Site code lives in this directory; TypeNN's
implementation is maintained in the parent repository. Browser software-renderer checks must be
identified as such, never as physical-GPU performance measurements.
