import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { compileTsSource } from "typeshade";
import path from "node:path";

export const ELEMENTWISE_HELPER_SPECS = Object.freeze([
  {
    operation: "add",
    file: "src/kernels/add.shade.ts",
    helper: "typennAddF32",
    output: "result",
    inputs: ["left", "right"],
  },
  {
    operation: "mul",
    file: "src/kernels/mul.shade.ts",
    helper: "typennMulF32",
    output: "result",
    inputs: ["left", "right"],
  },
  {
    operation: "relu",
    file: "src/kernels/relu.shade.ts",
    helper: "typennReluF32",
    output: "result",
    inputs: ["inputValue"],
  },
]);

function isF32(type) {
  return type?.kind === "scalar" && type.scalar === "f32";
}

function visitExpression(expression, spec, reads) {
  switch (expression.op) {
    case "lit":
      return;
    case "param":
    case "varref":
      if (expression.name === spec.output)
        throw new Error(`${spec.helper} reads its output before assigning it.`);
      if (!spec.inputs.includes(expression.name))
        throw new Error(
          `${spec.helper} reads undeclared parameter ${expression.name}.`,
        );
      reads.add(expression.name);
      return;
    case "binop":
    case "compare":
    case "logical":
      visitExpression(expression.a, spec, reads);
      visitExpression(expression.b, spec, reads);
      return;
    case "unop":
      visitExpression(expression.a, spec, reads);
      return;
    case "select":
      visitExpression(expression.cond, spec, reads);
      visitExpression(expression.ifTrue, spec, reads);
      visitExpression(expression.ifFalse, spec, reads);
      return;
    case "call":
      if (expression.declRef !== undefined || expression.fn !== "max")
        throw new Error(
          `${spec.helper} calls unsupported function ${expression.fn}.`,
        );
      for (const arg of expression.args) visitExpression(arg, spec, reads);
      return;
    default:
      throw new Error(
        `${spec.helper} uses unsupported expression ${expression.op}; helpers must be pure scalar expressions.`,
      );
  }
}

export function validateElementwiseHelper(func, spec) {
  if (func.stage !== undefined || func.ret?.kind !== "void")
    throw new Error(`${spec.helper} must be an ordinary void helper.`);

  if (func.params.length !== spec.inputs.length + 1)
    throw new Error(`${spec.helper} has an unexpected parameter count.`);

  const output = func.params[0];
  if (
    output?.name !== spec.output ||
    output.mode !== "inout" ||
    !isF32(output.type)
  )
    throw new Error(
      `${spec.helper} must begin with one @inout f32 output parameter named ${spec.output}.`,
    );

  for (let index = 0; index < spec.inputs.length; index++) {
    const parameter = func.params[index + 1];
    if (
      parameter?.name !== spec.inputs[index] ||
      parameter.mode !== undefined ||
      !isF32(parameter.type)
    )
      throw new Error(
        `${spec.helper} input ${index} must be by-value f32 ${spec.inputs[index]}.`,
      );
  }

  if (func.body.length !== 1 || func.body[0]?.s !== "assign")
    throw new Error(
      `${spec.helper} must assign its output exactly once in a straight-line body.`,
    );

  const assignment = func.body[0];
  if (
    assignment.target.op !== "varref" ||
    assignment.target.name !== spec.output
  )
    throw new Error(
      `${spec.helper} may only write its designated output parameter ${spec.output}.`,
    );

  const reads = new Set();
  visitExpression(assignment.expr, spec, reads);
  if (
    reads.size !== spec.inputs.length ||
    spec.inputs.some((name) => !reads.has(name))
  )
    throw new Error(
      `${spec.helper} must derive its output from every named input parameter.`,
    );
}

/** Compile the actual shader sources and extract validated scalar operation FuncDecls. */
export async function generateElementwiseOpIr({
  sourceOverrides = {},
  outputDirectory = "src/runtime",
  outputPath = path.join(
    outputDirectory,
    "elementwise-op-declarations.generated.ts",
  ),
} = {}) {
  const declarations = [];
  const sourceManifest = [];
  for (const spec of ELEMENTWISE_HELPER_SPECS) {
    const source =
      sourceOverrides[spec.operation] ?? (await readFile(spec.file, "utf8"));
    const compiled = compileTsSource(source, {
      fileName: spec.file,
      emit: false,
    });
    const errors = compiled.diagnostics.filter(
      (diagnostic) => diagnostic.category === "error",
    );
    if (errors.length > 0)
      throw new Error(
        `${spec.file} TypeShade diagnostics:\n${errors.map((error) => error.message).join("\n")}`,
      );

    const helper = compiled.funcs.find((func) => func.name === spec.helper);
    if (!helper)
      throw new Error(`${spec.file} did not compile helper ${spec.helper}.`);
    validateElementwiseHelper(helper, spec);
    declarations.push(helper);
    sourceManifest.push({ ...spec, source });
  }

  const fingerprint = createHash("sha256")
    .update(JSON.stringify(sourceManifest))
    .digest("hex");
  const generated = `import type { FuncDecl } from "typeshade";\n\nexport const ELEMENTWISE_OP_IR_FINGERPRINT = ${JSON.stringify(fingerprint)};\nexport const ELEMENTWISE_OP_DECLARATIONS = JSON.parse(${JSON.stringify(JSON.stringify(declarations))}) as readonly FuncDecl[];\n`;

  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeFile(outputPath, generated, "utf8");
  return { declarations, fingerprint, outputPath };
}

// The build wrapper passes options through a short-lived JSON file so tests can
// exercise this exact production generator with isolated sources and outputs.
const optionsPath = process.env.TYPENN_ELEMENTWISE_OP_IR_OPTIONS;
if (optionsPath) {
  const options = JSON.parse(await readFile(optionsPath, "utf8"));
  const result = await generateElementwiseOpIr(options);
  const manifestPath = process.env.TYPENN_ELEMENTWISE_OP_IR_MANIFEST;
  if (manifestPath)
    await writeFile(
      manifestPath,
      JSON.stringify({
        fingerprint: result.fingerprint,
        declarations: result.declarations,
      }),
      "utf8",
    );
  process.stdout.write(
    `Generated ${result.declarations.length} validated TypeShade operation FuncDecls (${result.fingerprint.slice(0, 12)}).\n`,
  );
}
