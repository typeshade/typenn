import { build as bundle } from "esbuild";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import path from "node:path";

/** Build and run the source extractor; options are also used by isolated tests. */
export async function generateElementwiseOpIr(options = {}) {
  const outputPath =
    options.outputPath ??
    `${options.outputDirectory ?? "src/runtime"}/elementwise-op-declarations.generated.ts`;
  const scratchDirectory =
    options.scratchDirectory ?? ".typenn-build/elementwise-op-ir";
  const bundlePath = path.join(scratchDirectory, "generate.mjs");
  const optionsPath = path.join(scratchDirectory, "options.json");
  const manifestPath = path.join(scratchDirectory, "manifest.json");

  await mkdir(scratchDirectory, { recursive: true });
  await bundle({
    entryPoints: ["scripts/generate-elementwise-op-ir.mjs"],
    outfile: bundlePath,
    bundle: true,
    platform: "node",
    format: "esm",
    external: ["typescript"],
  });
  await writeFile(
    optionsPath,
    JSON.stringify({ ...options, outputPath }),
    "utf8",
  );
  try {
    const result = execFileSync(process.execPath, [bundlePath], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
      env: {
        ...process.env,
        TYPENN_ELEMENTWISE_OP_IR_OPTIONS: optionsPath,
        TYPENN_ELEMENTWISE_OP_IR_MANIFEST: manifestPath,
      },
    });
    if (result.stdout) process.stdout.write(result.stdout);
    if (result.stderr) process.stderr.write(result.stderr);
    return JSON.parse(await readFile(manifestPath, "utf8"));
  } catch (error) {
    const stdout = String(error?.stdout ?? "");
    const stderr = String(error?.stderr ?? "");
    if (stdout) process.stdout.write(stdout);
    const diagnostics = stderr.trim() || error?.message || String(error);
    throw new Error(
      `Elementwise TypeShade source generation failed:\n${diagnostics}`,
      { cause: error },
    );
  } finally {
    await rm(bundlePath, { force: true });
    await rm(optionsPath, { force: true });
    await rm(manifestPath, { force: true });
    if (!options.scratchDirectory)
      await rm(scratchDirectory, { recursive: true, force: true });
  }
}
