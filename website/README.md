# TypeNN website

This is the source for [typenn.typeshade.dev](https://typenn.typeshade.dev). It is an Astro site
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

The Playground trains TypeNN models in the browser on the Keras datasets. Browsers cannot fetch
the Keras sources (they send no CORS headers), so the deploy mirrors them under
`public/data/`, served at `/data/`. `scripts/fetch_datasets.py` builds the mirror:

```sh
npm run datasets                              # all datasets, into public/data/
python3 scripts/fetch_datasets.py --only mnist,imdb
```

The script needs Python 3 and numpy. It downloads each source to `.dataset-cache/` (skipped when
present), checks its SHA-256 against a table in the script, and writes per dataset a
`manifest.json` and binary parts. `public/data/index.json` lists the datasets. The deploy job
runs the script before the build. Cloudflare limits one static asset to 25 MiB, so the script
keeps every output file at 20 MiB or less and splits larger arrays into `.partN.bin` files.

| Dataset             | Source                                  | License and citation                                                                                                               |
| ------------------- | --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| MNIST               | `mnist.npz` (Keras)                     | CC BY-SA 3.0, per the Keras documentation. LeCun, Cortes and Burges.                                                               |
| Fashion-MNIST       | the four IDX files (Keras base URL)     | MIT, Zalando SE. Xiao, Rasul and Vollgraf, 2017.                                                                                   |
| CIFAR-10, CIFAR-100 | the binary editions from cs.toronto.edu | No explicit license. Krizhevsky, 2009 (tech report; cite it).                                                                      |
| IMDB                | `imdb.npz` (Keras)                      | No explicit license that we know of. Maas et al., ACL 2011. Vocabulary capped at 10000 words.                                      |
| Reuters             | `reuters.npz` (Keras)                   | Reuters-21578: free distribution for research purposes, per its readme (not a standard license). Vocabulary capped at 10000 words. |
| California housing  | `california_housing.npz` (Keras)        | StatLib, Pace and Barry, 1997. No explicit license found.                                                                          |

The text datasets use the Keras loader's split, seed 113 and token encoding. Their token ids
are `uint16` and the vocabulary is capped at 10000 words for the bag-of-words input.

## Project documents

`PRODUCT.md` and `DESIGN.md` describe verified capabilities, current limits, the page structure,
and the shared IBM Plex/Ant Design visual system. Site code lives in this directory; TypeNN's
implementation is maintained in the parent repository. Browser software-renderer checks must be
identified as such, never as physical-GPU performance measurements.
