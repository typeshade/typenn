import { build as bundle } from "esbuild";
import { build as viteBuild } from "vite";

const configSource = `
  import { defineConfig } from "vite";
  import { typeshade } from "typeshade/vite";

  export default defineConfig({
    plugins: [typeshade()],
    build: {
      target: "esnext",
      outDir: ".typenn-build/elementwise-fusion-browser",
      emptyOutDir: true,
      rollupOptions: {
        input: {
          graphIr: "scripts/verify-elementwise-fusion.html",
          publishedBundle: "scripts/verify-elementwise-fusion-production.html",
        },
      },
    },
  });
`;

await bundle({
  stdin: {
    contents: configSource,
    resolveDir: process.cwd(),
    sourcefile: "elementwise-fusion-browser.config.ts",
  },
  outfile: ".typenn-build/elementwise-fusion-browser.config.mjs",
  bundle: true,
  platform: "node",
  format: "esm",
  external: ["vite", "typescript"],
});

await viteBuild({
  configFile: ".typenn-build/elementwise-fusion-browser.config.mjs",
});
