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

CIFAR-10 and CIFAR-100 are mirrored. Their only source, cs.toronto.edu, does not allow
cross-origin downloads. The mirror is the R2 bucket `typenn-data`, prefix `datasets/v1/`
(`index.json`, `cifar10/`, `cifar100/`). The bucket is served at
`https://data.typeshade.dev/datasets/v1/`, and its CORS rule allows `https://nn.typeshade.dev`,
`http://localhost:4321` and `http://127.0.0.1:4321`. `MIRROR_BASE` in `src/lab/datasets.ts` holds
the URL. The site deploy builds and uploads no dataset. To rebuild the mirror:

1. Run `npm run datasets`. It writes `cifar10/`, `cifar100/` and `index.json` under `public/data/`.
2. Upload the contents of `public/data/` to the bucket `typenn-data` under `datasets/v1/`.
3. Do not commit `public/data/`. A local build can read another mirror: set
   `PUBLIC_TYPENN_DATA_BASE` (with a trailing slash) when you run `npm run build`.

```sh
npm run datasets                              # cifar10 and cifar100, into public/data/
python3 scripts/fetch_datasets.py --all       # all seven, for the parity test
python3 scripts/fetch_datasets.py --only mnist
```

The script needs Python 3 and numpy. It downloads each source to `.dataset-cache/` (skipped when
present), checks its SHA-256 against a table in the script, and writes per dataset a
`manifest.json` and binary parts. `public/data/index.json` lists the mirrored datasets. The script
keeps every output file at 20 MiB or less and splits larger arrays into `.partN.bin` files.

`npm run test:keras` checks the browser decoder. It takes the five source files from
`.dataset-cache/` when they are there, and downloads the missing ones from the Keras storage into a
temporary directory (about 60 MB). It checks each file's SHA-256, decodes each dataset and compares
the SHA-256 and the length of every array with `scripts/keras-expected.json`. CI runs it in the
`cpu` job. The fixture comes from the mirror, which equals Keras 3.15.1 `load_data()` element for
element. When the five mirrored datasets are in `public/data/` (`python3 scripts/fetch_datasets.py
--all`), the test also compares the arrays with the mirror directly. Rebuild the fixture from the
mirror with `npm run test:keras -- --write-fixture`. Override the directories with
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
`data.typeshade.dev`, trains with `Module`, `Sequential`, `Linear`, `Conv2d`
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
