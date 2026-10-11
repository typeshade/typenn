import { Button, ConfigProvider, Tabs } from "antd";
import { useState } from "react";

type SnippetKey = "tensor" | "kernel" | "training";

type Props = {
  labels: {
    title: string;
    operation: string;
    trainingTab: string;
    tensor: string;
    kernel: string;
    training: string;
    tensorFile: string;
    kernelFile: string;
    trainingFile: string;
    copy: string;
    copied: string;
    copyFailed: string;
    output: string;
    operationOutput: string;
    compileNote: string;
  };
};

const snippets: Record<SnippetKey, string> = {
  tensor: `import * as nn from "typenn";

const device = await nn.device();
const a = device.tensor([1, 2, 3, 4], { shape: [2, 2] });
const b = device.tensor([5, 6, 7, 8], { shape: [2, 2] });
const y = a.matmul(b).relu();

console.log(await y.data()); // Float32Array [19, 22, 43, 50]
await device.dispose();`,
  kernel: `"use typeshade";

class Shape {
  length: u32;
}

declare const shape: uniform<Shape>;
declare const input: storage<array<f32>>;
declare const output: storage<array<f32>, "read_write">;

@compute([64])
export function reluKernel(@builtin("global_invocation_id") gid: vec3u) {
  if (gid.x >= shape.length) {
    return;
  }
  let value: f32 = input[gid.x];
  output[gid.x] = max(value, 0.0);
}`,
  training: `class SimpleMLP extends nn.Module {
  private readonly flatten = this.register("flatten", new nn.Flatten());
  private readonly fc1 = this.register("fc1", new nn.Linear(784, 128));
  private readonly relu = this.register("relu", new nn.ReLU());
  private readonly dropout = this.register("dropout", new nn.Dropout(0.2));
  private readonly fc2 = this.register("fc2", new nn.Linear(128, 10));

  forward(input: nn.Tensor): nn.Tensor {
    return this.fc2.forward(
      this.dropout.forward(
        this.relu.forward(this.fc1.forward(this.flatten.forward(input))),
      ),
    );
  }
}

const model = new SimpleMLP().to(device);
const optimizer = nn.optim.Adam(model.parameters(), { lr: 0.001 });
const criterion = new nn.CrossEntropyLoss();
model.train();

for await (const batch of batches) {
  await optimizer.zeroGrad();
  const images = device.tensor(batch.images, {
    shape: [batch.size, 1, 28, 28],
  });
  const logits = model.forward(images);
  const loss = criterion.forward(logits, batch.labels);
  await loss.backward();
  await optimizer.step();
  await loss.disposeGraph(model.parameters());
}`,
};

export default function CodeShowcase({ labels }: Props) {
  const [copyState, setCopyState] = useState<{
    key: SnippetKey;
    state: "copied" | "failed";
  } | null>(null);
  const [activeKey, setActiveKey] = useState<"operation" | "training">("operation");

  async function copyCode(key: SnippetKey) {
    try {
      await navigator.clipboard.writeText(snippets[key]);
      setCopyState({ key, state: "copied" });
    } catch {
      setCopyState({ key, state: "failed" });
    }
    window.setTimeout(() => setCopyState(null), 1800);
  }

  function copyButton(key: SnippetKey) {
    const state = copyState?.key === key ? copyState.state : null;
    const label = state === "copied"
      ? labels.copied
      : state === "failed"
        ? labels.copyFailed
        : labels.copy;

    return (
      <Button
        type="text"
        size="small"
        className="!text-white/70 hover:!text-white"
        onClick={() => void copyCode(key)}
        aria-label={`${label}: ${key}`}
      >
        {label}
      </Button>
    );
  }

  const items = [
    {
      key: "operation",
      label: labels.operation,
      children: (
        <div className="grid gap-5 xl:grid-cols-2">
          <CodePanel
            title={labels.tensor}
            file={labels.tensorFile}
            code={snippets.tensor}
            copy={copyButton("tensor")}
          />
          <CodePanel
            title={labels.kernel}
            file={labels.kernelFile}
            code={snippets.kernel}
            copy={copyButton("kernel")}
          />
        </div>
      ),
    },
    {
      key: "training",
      label: labels.trainingTab,
      children: (
        <CodePanel
          title={labels.training}
          file={labels.trainingFile}
          code={snippets.training}
          copy={copyButton("training")}
        />
      ),
    },
  ];

  return (
    <ConfigProvider theme={{ token: { colorPrimary: "#1677ff", borderRadius: 6 } }}>
      <section
        aria-label={labels.title}
        className="code-showcase overflow-hidden rounded-lg border border-white/10 bg-[var(--code-bg)] text-white shadow-2xl shadow-slate-950/10"
      >
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 border-b border-white/10 px-5 py-3">
          <h2 className="text-sm font-semibold text-white">{labels.title}</h2>
          <span className="font-mono text-[11px] text-white/45">TypeScript · TypeShade</span>
        </div>
        <Tabs
          activeKey={activeKey}
          items={items}
          onChange={(key) => {
            if (key === "operation" || key === "training") setActiveKey(key);
          }}
        />
        <div className="border-t border-white/10 px-5 py-3 text-xs leading-6 text-white/55">
          {activeKey === "operation" && (
            <p>
              {labels.output}: <code className="font-mono text-[#95de64]">{labels.operationOutput}</code>
            </p>
          )}
          <p className="mt-1">{labels.compileNote}</p>
        </div>
      </section>
    </ConfigProvider>
  );
}

function CodePanel({
  title,
  file,
  code,
  copy,
}: {
  title: string;
  file: string;
  code: string;
  copy: React.ReactNode;
}) {
  return (
    <section className="min-w-0 overflow-hidden rounded-md border border-white/10 bg-black/10" aria-label={title}>
      <div className="flex min-h-12 flex-wrap items-center justify-between gap-x-2 border-b border-white/10 px-3 py-2">
        <div className="min-w-0">
          <h3 className="text-xs font-semibold text-white/90">{title}</h3>
          <p className="truncate font-mono text-[10px] text-white/45">{file}</p>
        </div>
        {copy}
      </div>
      <pre className="max-h-[390px] overflow-auto p-4 text-[11px] leading-[1.75] text-[#d6e4ff] sm:text-xs">
        <code>{code}</code>
      </pre>
    </section>
  );
}
