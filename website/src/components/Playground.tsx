import { Button, ConfigProvider, Input, Select } from "antd";
import Prism from "prismjs";
import "prismjs/components/prism-typescript";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Device, Tensor } from "../vendor/typenn/index.js";

type DevicePreference = "auto" | "cpu" | "webgpu" | "webgl2";
type Operation = "matmul" | "add" | "matmul-relu";
type LoadedTypeNN = typeof import("../vendor/typenn/index.js");
type DeviceEntry = { preference: DevicePreference; device: Device };

const initialA = ["1", "2", "3", "4"];
const initialB = ["5", "6", "7", "8"];

const text = {
  en: {
    input: "Inputs",
    operation: "Operation",
    backend: "Execution tier",
    run: "Run TypeNN",
    running: "Running…",
    values: "Matrix values",
    a: "Matrix A · 2 × 2",
    b: "Matrix B · 2 × 2",
    choose: "Choose an operation",
    matmul: "A × B · MatMul",
    add: "A + B · Add",
    relu: "ReLU(A × B)",
    auto: "Auto (WebGPU, then CPU)",
    cpu: "CPU",
    webgpu: "WebGPU",
    webgl2: "WebGL2",
    result: "Result",
    waiting: "Run an operation to see its output.",
    compiling: "Loading the TypeNN runtime and preparing the selected backend…",
    success: "Executed by TypeNN",
    snippet: "Equivalent TypeScript",
    source: "Runtime source",
    errorTitle: "Execution failed",
    errorHelp: "Try CPU, check the matrix values, or select a backend available in this browser.",
    readback: "The result is read by an explicit data() call. Intermediate tensors are disposed after the read.",
    software: "Browser support varies. A software renderer is not a physical GPU benchmark.",
    shapeError: "Enter four finite numbers for each matrix.",
  },
  ko: {
    input: "입력",
    operation: "연산",
    backend: "실행 계층",
    run: "TypeNN 실행",
    running: "실행 중…",
    values: "행렬 값",
    a: "행렬 A · 2 × 2",
    b: "행렬 B · 2 × 2",
    choose: "연산 선택",
    matmul: "A × B · MatMul",
    add: "A + B · Add",
    relu: "ReLU(A × B)",
    auto: "자동 (WebGPU, 이후 CPU)",
    cpu: "CPU",
    webgpu: "WebGPU",
    webgl2: "WebGL2",
    result: "결과",
    waiting: "연산을 실행하면 결과가 표시됩니다.",
    compiling: "TypeNN 런타임을 불러오고 백엔드를 준비합니다…",
    success: "TypeNN 실행 완료",
    snippet: "동일한 TypeScript 코드",
    source: "런타임 소스",
    errorTitle: "실행하지 못했습니다",
    errorHelp: "CPU를 시도하거나 행렬 값을 확인하고, 브라우저에서 사용할 수 있는 백엔드를 선택하세요.",
    readback: "명시적인 data() 호출로 결과를 읽습니다. 읽기가 끝나면 중간 Tensor를 해제합니다.",
    software: "브라우저 지원은 환경마다 다릅니다. 소프트웨어 렌더러 결과는 물리 GPU 벤치마크가 아닙니다.",
    shapeError: "각 행렬에 유한한 숫자 네 개를 입력하세요.",
  },
};

export default function Playground({ locale }: { locale: "en" | "ko" }) {
  const words = text[locale];
  const [aValues, setAValues] = useState(initialA);
  const [bValues, setBValues] = useState(initialB);
  const [operation, setOperation] = useState<Operation>("matmul-relu");
  const [preference, setPreference] = useState<DevicePreference>("auto");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<number[] | null>(null);
  const [activeBackend, setActiveBackend] = useState<Device["backend"]["name"] | null>(null);
  const [fallback, setFallback] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const deviceEntry = useRef<DeviceEntry | null>(null);

  function clearOutput() {
    setResult(null);
    setError(null);
    setActiveBackend(null);
    setFallback(null);
  }

  useEffect(() => () => {
    const entry = deviceEntry.current;
    deviceEntry.current = null;
    if (entry) void entry.device.dispose();
  }, []);

  const snippet = useMemo(() => makeSnippet(operation, aValues, bValues, preference), [operation, aValues, bValues, preference]);
  const highlightedCode = useMemo(() => {
    const html = Prism.highlight(snippet, Prism.languages.typescript, "typescript");
    return { __html: html };
  }, [snippet]);

  async function run() {
    const leftValues = aValues.map(Number);
    const rightValues = bValues.map(Number);
    if ([...aValues, ...bValues].some((value) => value.trim() === "") || [...leftValues, ...rightValues].some((value) => !Number.isFinite(value))) {
      clearOutput();
      setError(words.shapeError);
      return;
    }

    setRunning(true);
    setError(null);
    setResult(null);
    let inputA: Tensor | undefined;
    let inputB: Tensor | undefined;
    let output: Tensor | undefined;
    try {
      const current = deviceEntry.current;
      if (current && current.preference !== preference) {
        await current.device.dispose();
        deviceEntry.current = null;
      }

      const runtime: LoadedTypeNN = await import("../vendor/typenn/index.js");
      let device = deviceEntry.current?.device;
      if (!device) {
        device = await runtime.device({ device: preference });
        deviceEntry.current = { device, preference };
      }

      inputA = device.tensor(leftValues, { shape: [2, 2] });
      inputB = device.tensor(rightValues, { shape: [2, 2] });
      if (operation === "add") output = inputA.add(inputB);
      else {
        const product = inputA.matmul(inputB);
        output = operation === "matmul-relu" ? product.relu() : product;
      }
      if (!output) throw new Error("The selected operation did not produce an output Tensor.");
      const values = await output.data();
      setResult(Array.from(values));
      setActiveBackend(device.backend.name);
      setFallback(device.backend.fallbackReason ?? null);
      await output.disposeGraph();
      output = undefined;
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause);
      setError(message);
      setActiveBackend(null);
      if (output) await output.disposeGraph().catch(() => undefined);
      else {
        if (inputA) await inputA.disposeGraph().catch(() => undefined);
        if (inputB) await inputB.disposeGraph().catch(() => undefined);
      }
    } finally {
      setRunning(false);
    }
  }

  return (
    <ConfigProvider theme={{ token: { colorPrimary: "#1677ff", borderRadius: 6, fontFamily: '"IBM Plex Sans", "IBM Plex Sans KR", sans-serif' } }}>
      <section className="playground-app" aria-label={words.input}>
        <div className="playground-workbench">
          <div className="playground-controls">
            <div className="control-heading"><span className="eyebrow">TypeNN Runtime</span><h2>{words.input}</h2></div>
            <MatrixEditor title={words.a} values={aValues} onChange={(values) => { setAValues(values); clearOutput(); }} labels={words.values} disabled={running} />
            <MatrixEditor title={words.b} values={bValues} onChange={(values) => { setBValues(values); clearOutput(); }} labels={words.values} disabled={running} />
            <div className="select-grid">
              <label className="field-label">{words.operation}
                <Select<Operation> value={operation} disabled={running} onChange={(value) => { setOperation(value); clearOutput(); }} options={[
                  { value: "matmul", label: words.matmul },
                  { value: "add", label: words.add },
                  { value: "matmul-relu", label: words.relu },
                ]} />
              </label>
              <label className="field-label">{words.backend}
                <Select<DevicePreference> value={preference} disabled={running} onChange={(value) => { setPreference(value); clearOutput(); }} options={[
                  { value: "auto", label: words.auto },
                  { value: "cpu", label: words.cpu },
                  { value: "webgpu", label: words.webgpu },
                  { value: "webgl2", label: words.webgl2 },
                ]} />
              </label>
            </div>
            <Button type="primary" size="large" loading={running} disabled={running} onClick={() => void run()} className="run-button">
              {running ? words.running : words.run}
            </Button>
            <p className="field-help">{words.software}</p>
          </div>

          <div className="playground-output">
            <div className="output-header"><div><span className="eyebrow">{words.result}</span><h2>{operation === "add" ? "A + B" : operation === "matmul" ? "A × B" : "ReLU(A × B)"}</h2></div>
              {activeBackend && <span className="backend-chip"><i aria-hidden="true"></i>{activeBackend}{fallback ? " · fallback" : ""}</span>}
            </div>
            {running ? <div className="result-empty" role="status"><span className="loading-line"></span><p>{words.compiling}</p></div> : result ? (
              <div className="matrix-result" role="status" aria-live="polite">
                <span className="result-caption">Float32Array · [2, 2]</span>
                <div className="result-grid">{result.map((value, index) => <output key={index}>{formatNumber(value)}</output>)}</div>
                <span className="result-success"><i aria-hidden="true"></i>{words.success}</span>
                {fallback && <p className="field-help">{fallback}</p>}
              </div>
            ) : error ? (
              <div className="result-error" role="alert"><strong>{words.errorTitle}</strong><code>{error}</code><p>{words.errorHelp}</p></div>
            ) : <div className="result-empty"><span className="empty-mark" aria-hidden="true">[ ]</span><p>{words.waiting}</p></div>}
            <p className="readback-note">{words.readback}</p>
          </div>
        </div>

        <div className="playground-code">
          <div className="code-caption"><span>{words.snippet}</span><span className="code-file">TypeScript · TypeNN API</span></div>
          <pre><code className="language-typescript" dangerouslySetInnerHTML={highlightedCode} /></pre>
          <a className="text-link" href="https://github.com/typeshade/typenn">{words.source} ↗</a>
        </div>
      </section>
    </ConfigProvider>
  );
}

function MatrixEditor({
  title,
  values,
  onChange,
  labels,
  disabled,
}: {
  title: string;
  values: string[];
  onChange: (values: string[]) => void;
  labels: string;
  disabled: boolean;
}) {
  return (
    <fieldset className="matrix-editor">
      <legend>{title}</legend>
      <span className="field-help">{labels}</span>
      <div className="matrix-input-grid">
        {values.map((value, index) => <Input
          key={index}
          type="number"
          step="any"
          value={value}
          disabled={disabled}
          aria-label={`${title}, row ${Math.floor(index / 2) + 1}, column ${index % 2 + 1}`}
          onChange={(event) => onChange(values.map((item, itemIndex) => itemIndex === index ? event.target.value : item))}
        />)}
      </div>
    </fieldset>
  );
}

function formatNumber(value: number) {
  return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(5)));
}

function makeSnippet(operation: Operation, aValues: string[], bValues: string[], preference: DevicePreference) {
  const safeValues = (values: string[]) => values.map((value) => {
    const number = Number(value);
    return value.trim() !== "" && Number.isFinite(number) ? String(number) : "0";
  }).join(", ");
  const expression = operation === "add" ? "a.add(b)" : operation === "matmul" ? "a.matmul(b)" : "a.matmul(b).relu()";
  return `const device = await nn.device({ device: "${preference}" });
const a = device.tensor([${safeValues(aValues)}], { shape: [2, 2] });
const b = device.tensor([${safeValues(bValues)}], { shape: [2, 2] });
const output = ${expression};
const values = await output.data();`;
}
