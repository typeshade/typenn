import { copyFile, mkdir, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { resolve } from "node:path";

async function findTypeNNRoot() {
  if (process.env.TYPE_NN_ROOT) return resolve(process.env.TYPE_NN_ROOT);
  for (const candidate of ["..", "../../typenn"]) {
    const root = resolve(candidate);
    try {
      const metadata = JSON.parse(await readFile(resolve(root, "package.json"), "utf8"));
      if (metadata.name === "typenn") return root;
    } catch {
      // Try the other conventional checkout location.
    }
  }
  throw new Error("Set TYPE_NN_ROOT to a TypeNN checkout before syncing its browser runtime.");
}

const typeNNRoot = await findTypeNNRoot();
const distribution = resolve(process.env.TYPE_NN_DIST ?? `${typeNNRoot}/dist`);
const destination = resolve("src/vendor/typenn");
const generated = (await readdir(distribution)).filter((name) => name.endsWith(".js"));
if (!generated.includes("index.js")) {
  throw new Error(`TypeNN build output is missing ${distribution}/index.js. Run npm run build in TypeNN first.`);
}

await mkdir(destination, { recursive: true });
for (const name of await readdir(destination)) {
  if (name.endsWith(".js")) await rm(resolve(destination, name));
}
for (const name of generated) await copyFile(resolve(distribution, name), resolve(destination, name));
await copyFile(resolve(typeNNRoot, "LICENSE"), resolve(destination, "LICENSE"));

let commit = "unknown";
try {
  commit = execFileSync("git", ["-C", typeNNRoot, "rev-parse", "HEAD"], { encoding: "utf8" }).trim();
} catch {
  // An exported TypeNN checkout can still be bundled; its provenance must be set explicitly.
  commit = process.env.TYPE_NN_SOURCE_COMMIT ?? "unknown";
}
await writeFile(resolve(destination, "version.json"), `${JSON.stringify({ commit, source: "https://github.com/typeshade/typenn" }, null, 2)}\n`);
await writeFile(resolve(destination, "README.md"), `# Vendored TypeNN browser runtime\n\nGenerated from [typeshade/typenn](https://github.com/typeshade/typenn) at commit \`${commit}\`.\n\nRebuild TypeNN, then run \`npm run sync:typenn\` from the website directory. The original Apache-2.0 license is included as \`LICENSE\`. The declaration file \`index.d.ts\` is maintained by the website for its narrow Playground import surface.\n`);
console.log(`Synced ${generated.length} TypeNN runtime JavaScript files from ${commit}.`);
