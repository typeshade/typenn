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

## Project documents

`PRODUCT.md` and `DESIGN.md` describe verified capabilities, current limits, the page structure,
and the shared IBM Plex/Ant Design visual system. Site code lives in this directory; TypeNN's
implementation is maintained in the parent repository. Browser software-renderer checks must be
identified as such, never as physical-GPU performance measurements.
