import { readdir, readFile } from "node:fs/promises";
import prettier from "prettier";

async function filesIn(directory) {
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = `${directory}/${entry.name}`;
    if (entry.isDirectory()) files.push(...(await filesIn(path)));
    else if (entry.isFile() && !entry.name.endsWith(".shade.typeshade.ts"))
      files.push(path);
  }
  return files;
}

function removeTypeShadeDecorators(source) {
  return source
    .replace(/^@compute\([^\n]+\)\r?\n/gm, "")
    .replace(/@builtin\("global_invocation_id"\)\s*/g, "");
}

const sourceFiles = await filesIn("src");
const scriptFiles = await filesIn("scripts");
const exampleFiles = await filesIn("examples");
const configFiles = [
  "README.md",
  "TODO.md",
  "package.json",
  "package-lock.json",
  "tsconfig.json",
  "vite.config.ts",
];
const files = [...sourceFiles, ...scriptFiles, ...exampleFiles, ...configFiles];
const failures = [];

for (const path of files) {
  const source = await readFile(path, "utf8");
  const formattedSource = path.endsWith(".shade.ts")
    ? removeTypeShadeDecorators(source)
    : source;
  if (!(await prettier.check(formattedSource, { filepath: path })))
    failures.push(path);
}

if (failures.length > 0) {
  process.stderr.write(
    `Formatting errors in:\n${failures.map((path) => `- ${path}`).join("\n")}\n`,
  );
  process.exitCode = 1;
} else {
  process.stdout.write(`Formatting check passed for ${files.length} files.\n`);
}
