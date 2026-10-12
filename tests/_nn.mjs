// TypeNN as a test meets it: the library built in `dist`, or, when TYPENN_SOURCE names a
// directory with a `src/` and a `vite.config.ts`, the sources of that directory loaded through
// Vite. The instrument test (`instrument.test.mjs`) uses the second form to run the tests against
// a copy of the sources with a fault planted in a kernel function.

import { createRequire } from "node:module";
import { mkdirSync } from "node:fs";
import { join, resolve } from "node:path";

let server;

export async function loadNn() {
  const source = process.env.TYPENN_SOURCE;
  if (source === undefined) return import("../dist/index.js");
  const root = resolve(source);
  const require = createRequire(join(root, "package.json"));
  const configDirectory = join(root, ".typenn-build");
  mkdirSync(configDirectory, { recursive: true });
  // The plugin is TypeScript source, so the config is bundled first, as `scripts/build.mjs` does.
  await require("esbuild").build({
    entryPoints: [join(root, "vite.config.ts")],
    outfile: join(configDirectory, "vite.config.mjs"),
    bundle: true,
    platform: "node",
    format: "esm",
    external: ["vite", "typescript"],
  });
  const vite = await import(require.resolve("vite"));
  server = await vite.createServer({
    root,
    configFile: join(configDirectory, "vite.config.mjs"),
    appType: "custom",
    server: { middlewareMode: true },
    logLevel: "error",
  });
  return server.ssrLoadModule("/src/index.ts");
}

export async function closeNn() {
  await server?.close();
  server = undefined;
}
