// Chromium with WebGPU for the GPU tests: the built library in `dist/`, the test modules in
// `tests/` and the bundled MNIST example are served from this repository, and a page runs a
// function of a test in the browser. TYPESHADE_CHROMIUM names the executable (locally
// /opt/pw-browsers/chromium-1194/chrome-linux/chrome, Chromium 141); left out, Playwright's own
// is used, which CI installs. WebGPU is software here: SwiftShader, as in TypeShade's own gates,
// so a test holds a plan and a value, and never a time.

import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize, sep } from "node:path";
import { build } from "esbuild";
import { chromium } from "playwright-core";

const root = join(import.meta.dirname, "..", "..");
const TYPES = {
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
};
const SERVED = ["dist", "tests", ".typenn-build"];

/** The MNIST example as a module for a page: `typenn` stays an import, which the page's import
 *  map points at `dist/index.js`. */
async function bundleExample() {
  await build({
    entryPoints: [join(root, "examples", "mnist", "loop.ts")],
    outfile: join(root, ".typenn-build", "mnist", "loop-browser.mjs"),
    bundle: true,
    format: "esm",
    platform: "browser",
    external: ["typenn"],
    logLevel: "silent",
  });
}

/**
 * Open a page with WebGPU, call `body(argument)` in it and close everything. `body` cannot close
 * over anything of the caller; it imports what it needs (`/dist/index.js`, `/tests/...`). Resolves
 * to what `body` returns, and to the errors the page logged. `options.dist` serves another build
 * of the library as `/dist`.
 */
export async function inPage(body, argument, options = {}) {
  const dist = options.dist ?? join(root, "dist");
  await bundleExample();
  const server = createServer(async (request, response) => {
    const url = new URL(request.url ?? "/", "http://localhost");
    if (url.pathname === "/favicon.ico") {
      response.statusCode = 204;
      return response.end();
    }
    if (url.pathname === "/") {
      response.setHeader("content-type", "text/html; charset=utf-8");
      return response.end(
        '<!doctype html><title>typenn</title><script type="importmap">{"imports":{"typenn":"/dist/index.js"}}</script>',
      );
    }
    const file = url.pathname.startsWith("/dist/")
      ? normalize(join(dist, url.pathname.slice("/dist/".length)))
      : normalize(join(root, url.pathname));
    const served =
      file.startsWith(dist + sep) ||
      SERVED.some((directory) => file.startsWith(join(root, directory) + sep));
    try {
      if (!served) throw new Error("not served");
      response.setHeader(
        "content-type",
        TYPES[extname(file)] ?? "application/octet-stream",
      );
      response.end(await readFile(file));
    } catch {
      response.statusCode = 404;
      response.end();
    }
  });
  await new Promise((done) => server.listen(0, "127.0.0.1", done));
  const browser = await chromium.launch({
    executablePath: process.env.TYPESHADE_CHROMIUM || undefined,
    args: options.hardware
      ? ["--enable-unsafe-webgpu"]
      : [
          "--enable-unsafe-webgpu",
          "--enable-unsafe-swiftshader",
          "--use-angle=swiftshader",
          "--use-vulkan=swiftshader",
          "--enable-features=Vulkan",
        ],
  });
  const logged = [];
  try {
    const page = await browser.newPage();
    // What the page says on its console as an error or a warning. Chromium's note that WebGPU is
    // experimental on this platform is no finding.
    page.on("console", (message) => {
      if (
        ["error", "warning"].includes(message.type()) &&
        !message.text().startsWith("WebGPU is experimental")
      )
        logged.push(`${message.type()}: ${message.text()}`);
    });
    page.on("pageerror", (error) => logged.push(`pageerror: ${error}`));
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const result = await page.evaluate(body, argument);
    return { result, logged };
  } finally {
    await browser.close();
    server.close();
  }
}
