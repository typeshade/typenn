# Website source migration

## Change

This branch removes the Astro website from the TypeNN library repository. The site source now lives
in the TypeShade web monorepo at `apps/typenn`. The public URL, library implementation, npm package
configuration, CPU tests, and GPU tests remain owned by this repository or the web app as appropriate.

Website-specific files removed from the library repository:

- `website/`, including its Astro/React source, static assets, dataset mirror helpers, vendored
  browser runtime, and website-only package manifest and lockfile.
- `.github/workflows/deploy.yml` and `wrangler.jsonc`, which deployed `website/dist` to Cloudflare.
- The `wrangler` root dev dependency and its lockfile-only transitive dependencies.
- The library CI step that ran the website's Keras dataset decoder test.
- The website-dependent Lab measurement harness was moved to
  `web/apps/typenn/scripts/measure-lab.mjs` and `measure-lab-page.ts`, with the `measure:lab` app
  script. It remains an opt-in browser measurement tool; its historical measurements remain in the
  library README.

Vite remains a root dev dependency because `scripts/build.mjs` uses the TypeShade Vite plugin to
generate executable library bundles. `playwright-core` remains because the library's GPU tests use
it. The library build, package tests, GPU tests, and npm publish workflow are unchanged.

The website's `test:keras`, `test:datasets`, and `test:graph` checks already exist in the web app at
`apps/typenn/scripts/` and remain in that app's scripts. They are not moved into the library test
suite because they test website-owned decoders and graph UI helpers.

## Ownership guidance

`README.md` and `AGENTS.md` now direct website work to the TypeShade web repository. The repository
root no longer has a website package, site lockfile, site deploy workflow, or Wrangler config.

## Comparison notes

The source at this branch starts from library commit `eea7f19`. Its library code includes the
`DeviceDataset` API and changes after commit `61259c3`. The web app source already contains
device-resident dataset training code. During this migration, the web runtime pin was advanced to
`eea7f19`; its `typenn-site` check and static build both passed against that API.

Comparing the clean library website, the dirty original checkout's website source, and
`web/apps/typenn` showed that the web app already contains the Keras decoder, graph-layout and
dataset-helper checks, the training loop, the device-resident dataset flow, and the newer lab
components. The `61259c3` library website source already uses device-resident datasets; it is not a
host-batch baseline. The web app also has further page-template, layout, accessibility, and reviewed
copy changes. Those web-specific sources remain owned by the web repository; this migration does
not copy them back into the library repository.

## Validation

Validation from this cleanup branch:

- `npm ci` — passed.
- `npm run lint` — passed, 101 files.
- `npm test` — passed, 684/684 tests; it builds the library first.
- `TYPESHADE_CHROMIUM=/usr/bin/chromium npm run test:gpu` — passed, 9/9 tests on SwiftShader,
  including WebGPU/WebGL2 parity, dataset batching, and the MNIST example.
- `npm pack --dry-run` — passed; the package contains 57 library files and no website source.
- The web app's `typenn-site check` and `typenn-site build` passed after syncing the `eea7f19` runtime.
- The website's Keras decoder, dataset-helper, and graph-layout tests remain in the web app under
  `test:keras`, `test:datasets`, and `test:graph`.
- The moved measurement harness ran from `web/apps/typenn` with
  `TYPESHADE_CHROMIUM=/usr/bin/chromium node scripts/measure-lab.mjs --main /workspace/typenn --datasets mnist --steps 2 --warmup 1`.
  Both runs completed on SwiftShader without browser errors. The `main` run reported 140.1 ms/step
  and 36,072 uploaded bytes/step; the `web` run reported 213.2 ms/step and the same 36,072 bytes/step.
  Both reported a 55,178,912-byte one-time upload. This is a path and output smoke check only: the
  original dirty checkout's source is at `61259c3733148a8665466163ae1c5c2b920b129c`, but its
  pre-existing ignored `dist/` identifies itself as `1de47e704de1b8dfd8ea63841a439a16a8da55be`,
  while the web vendor is pinned to `eea7f190a1df2c0591e1e72f718b109e7d06db04`. The short run and
  mismatched runtime commits do not establish a performance comparison; rebuild the baseline from
  its intended source revision before using the harness for that purpose. The original dirty
  checkout was left untouched.
