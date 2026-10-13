import {
  Button,
  ConfigProvider,
  Input,
  InputNumber,
  Progress,
  Select,
  Table,
  theme as antdTheme,
} from "antd";
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";
import {
  isCached,
  labels,
  listDatasets,
  loadDataset,
  loadManifest,
  type DatasetInfo,
  type LoadedDataset,
  type Manifest,
} from "../lab/datasets";
import KernelGraph from "./KernelGraph";
import { mergeSnapshots, sampleIndices } from "../lab/graph-layout";
import { encode, bagOf, wordIndex } from "../lab/text";
import {
  buildModel,
  fillInputs,
  inputSize,
  mulberry32,
  predict,
  presetsFor,
  specFor,
  targetScale,
  train,
  type ModelSpec,
  type Preset,
  type RunConfig,
  type RunResult,
  type StepInfo,
  type TierPreference,
  type Metrics,
  type NN,
  type Device,
} from "../lab/train";
import type {
  DevicePlan,
  GraphSnapshot,
  Module,
} from "../vendor/typenn/index.js";

type Locale = "en" | "ko";
type Words = Record<string, any>;
type GraphKind = "forward" | "step";
interface GraphTab {
  readonly title: string;
  readonly snapshot: GraphSnapshot | null;
  readonly plan: DevicePlan | undefined;
}

const text: { en: Record<string, any>; ko: Record<string, any> } = {
  en: {
    intro:
      "Train a TypeNN model in this page, on the full training split of a real dataset. The data is the Keras datasets. The sources do not allow cross-origin downloads, so this site mirrors them. Each card names the citation and the license of its dataset.",
    datasetTitle: "1. Dataset",
    download: "Download",
    downloading: "Downloading",
    loaded: "Loaded in this page",
    cachedNote: "Cached in this browser. No download is needed.",
    notCached: "Not cached yet. The first download fetches the files.",
    bytes: "Size",
    counts: "Train / test",
    license: "License",
    citation: "Citation",
    modelTitle: "2. Model and training",
    preset: "Model",
    mlp: "MLP: Flatten, Linear, ReLU, Dropout, Linear",
    cnn: "CNN: two Conv2d and MaxPool2d blocks, then Linear",
    bag: "Bag-of-words MLP: Linear, ReLU, Dropout, Linear",
    regression: "MLP for regression: Linear, ReLU, Linear",
    hidden: "Hidden width",
    optimizer: "Optimizer",
    learningRate: "Learning rate",
    batchSize: "Batch size",
    epochs: "Epochs",
    tier: "Execution tier",
    auto: "Auto (WebGPU, then CPU)",
    cpu: "CPU",
    webgpu: "WebGPU",
    webgl2: "WebGL2",
    runTitle: "3. Run",
    start: "Start",
    stop: "Stop",
    stopping: "Stopping after this step",
    idle: "Not running.",
    preparing: "Preparing the dataset and the runtime…",
    status: "Epoch {epoch}/{epochs}, step {step}/{steps}",
    rate: "samples per second",
    eta: "ETA",
    elapsed: "Elapsed",
    lossChart: "Training loss per step (smoothed)",
    metricChart: "Test metric per epoch",
    accuracyAxis: "accuracy",
    rmseAxis: "RMSE, USD",
    stepAxis: "step",
    epochAxis: "epoch",
    resultsTitle: "4. Results",
    noResults: "No finished runs yet.",
    copyJson: "Copy as JSON",
    copied: "Copied.",
    copyFailed: "The browser did not allow the copy.",
    col: {
      dataset: "Dataset",
      model: "Model",
      tier: "Tier",
      epochs: "Epochs",
      metric: "Test metric",
      time: "Time, s",
      status: "Status",
    },
    completed: "Completed",
    stopped: "Stopped",
    tryTitle: "5. Try the model",
    noModel: "Run a model first. Then you can try it here.",
    trainedOn: "Trained on",
    clear: "Clear",
    randomTest: "Random test image",
    randomRow: "Random test row",
    draw: "Draw a digit in the box, or pick a random test image.",
    classify: "Classify",
    trueLabel: "True label",
    none: "none",
    top: "Top 3 classes",
    reviewPlaceholder: "Type a review in English.",
    probabilities: "Class probabilities",
    features: "Features, in the units of the dataset",
    predictedPrice: "Predicted price",
    actualPrice: "Actual price",
    resultTitle: "Result",
    errorTitle: "The run failed",
    errorHelp: "Try CPU, a smaller batch size, or fewer epochs.",
    emptyLoss: "The loss curve appears when a run starts.",
    emptyMetric: "The test metric appears after the first epoch.",
    slowCpu:
      "The CPU tier is slow for this model: about 2 s per step for the text model and about 4 s per step for the CNN, measured on one machine. Choose Auto on a browser with WebGPU.",
    graphTitle: "Kernel graph",
    graphIntro:
      "The graph shows how TypeNN runs one read: each box is a tensor, and each group is a set of calls that one dispatch runs on WebGPU. A fused group is one dispatch. A dispatch is one GPU call.",
    graphReadme: "How a read runs, on GitHub",
    graphForward: "Forward of one batch",
    graphStep: "One training step (forward + loss + backward)",
    graphFresh: "A new, untrained model with the selected preset.",
    graphTrained: "The trained model of this page.",
    graphEmpty: "Choose a button to draw the graph of a read.",
    graphNoPlan2:
      "This build of the library does not let the page read the gradients in one plan, so plan 2 is not shown.",
    graphBatch:
      "Batch: {n} random samples of the test split (seed {seed}). Parameter checksum before and after: {sum}.",
    graphStepNote:
      "A training step reads twice. Plan 1 reads the loss. backward() only records the gradient operations. Plan 2 reads all gradients in one plan, as optimizer.step() does. This view runs plan 2 but does not update the parameters.",
    graphTab1: "Plan 1: loss",
    graphTab2: "Plan 2: gradients",
    graphForwardTab: "Plan: forward",
    graphError: "The graph could not be drawn.",
    running: "A run is in progress.",
    selectDataset: "Select a dataset.",
    unit: "USD",
    fallback: "Fallback",
  },
  ko: {
    intro:
      "이 페이지에서 실제 데이터셋의 학습 데이터 전체로 TypeNN 모델을 학습합니다. 데이터는 Keras 데이터셋입니다. 원본 서버가 브라우저의 직접 내려받기(CORS)를 허용하지 않아 이 사이트가 사본을 제공합니다. 각 카드에 데이터셋의 출처와 라이선스를 표시합니다.",
    datasetTitle: "1. 데이터셋",
    download: "내려받기",
    downloading: "내려받는 중",
    loaded: "이 페이지에 적재됨",
    cachedNote: "이 브라우저에 저장되어 있습니다. 다시 내려받지 않습니다.",
    notCached: "아직 저장되지 않았습니다. 처음 내려받을 때 파일을 가져옵니다.",
    bytes: "크기",
    counts: "학습 / 테스트",
    license: "라이선스",
    citation: "출처",
    modelTitle: "2. 모델과 학습",
    preset: "모델",
    mlp: "MLP: Flatten, Linear, ReLU, Dropout, Linear",
    cnn: "CNN: Conv2d와 MaxPool2d 블록 두 개, 이후 Linear",
    bag: "단어 가방 MLP: Linear, ReLU, Dropout, Linear",
    regression: "회귀 MLP: Linear, ReLU, Linear",
    hidden: "은닉층 너비",
    optimizer: "옵티마이저",
    learningRate: "학습률",
    batchSize: "배치 크기",
    epochs: "에포크",
    tier: "실행 계층",
    auto: "자동 (WebGPU, 이후 CPU)",
    cpu: "CPU",
    webgpu: "WebGPU",
    webgl2: "WebGL2",
    runTitle: "3. 실행",
    start: "시작",
    stop: "중지",
    stopping: "이 단계가 끝나면 멈춥니다",
    idle: "실행 중이 아닙니다.",
    preparing: "데이터셋과 런타임을 준비하고 있습니다…",
    status: "에포크 {epoch}/{epochs}, 단계 {step}/{steps}",
    rate: "초당 샘플",
    eta: "남은 시간",
    elapsed: "경과 시간",
    lossChart: "단계별 학습 손실 (평활화)",
    metricChart: "에포크별 테스트 지표",
    accuracyAxis: "정확도",
    rmseAxis: "RMSE, USD",
    stepAxis: "단계",
    epochAxis: "에포크",
    resultsTitle: "4. 결과",
    noResults: "끝난 실행이 아직 없습니다.",
    copyJson: "JSON으로 복사",
    copied: "복사했습니다.",
    copyFailed: "브라우저가 복사를 허용하지 않았습니다.",
    col: {
      dataset: "데이터셋",
      model: "모델",
      tier: "계층",
      epochs: "에포크",
      metric: "테스트 지표",
      time: "시간, 초",
      status: "상태",
    },
    completed: "완료",
    stopped: "중지됨",
    tryTitle: "5. 모델 실행해 보기",
    noModel: "먼저 모델을 실행하세요. 그러면 여기서 모델을 시험할 수 있습니다.",
    trainedOn: "학습한 데이터",
    clear: "지우기",
    randomTest: "테스트 이미지 무작위 선택",
    randomRow: "테스트 행 무작위 선택",
    draw: "상자에 숫자를 그리거나 테스트 이미지를 무작위로 고르세요.",
    classify: "분류",
    trueLabel: "정답",
    none: "없음",
    top: "상위 3개 클래스",
    reviewPlaceholder: "영어로 리뷰를 입력하세요.",
    probabilities: "클래스별 확률",
    features: "데이터셋 단위의 특성값",
    predictedPrice: "예측 가격",
    actualPrice: "실제 가격",
    resultTitle: "결과",
    errorTitle: "실행하지 못했습니다",
    errorHelp: "CPU를 선택하거나, 배치 크기를 줄이거나, 에포크를 줄여 보세요.",
    emptyLoss: "학습을 시작하면 손실 곡선이 나타납니다.",
    emptyMetric: "첫 에포크가 끝나면 테스트 지표가 나타납니다.",
    slowCpu:
      "이 모델은 CPU 계층에서 느립니다. 한 대의 기기에서 측정한 결과, 텍스트 모델은 한 스텝에 약 2초, CNN은 약 4초가 걸렸습니다. WebGPU를 지원하는 브라우저에서는 자동을 선택하세요.",
    graphTitle: "커널 그래프",
    graphIntro:
      "그래프는 TypeNN이 read 한 번을 실행하는 방식을 보여 줍니다. 상자 하나는 Tensor 하나이고, 그룹 하나는 WebGPU에서 디스패치 한 번으로 실행하는 호출 묶음입니다. 융합된 그룹은 디스패치 한 번입니다. 디스패치는 GPU 호출 한 번입니다.",
    graphReadme: "read 실행 방식 (GitHub)",
    graphForward: "배치 하나의 순방향",
    graphStep: "학습 한 단계 (순방향 + 손실 + 역방향)",
    graphFresh: "선택한 모델 구조로 새로 만든, 학습하지 않은 모델입니다.",
    graphTrained: "이 페이지에서 학습한 모델입니다.",
    graphEmpty: "버튼을 누르면 read의 그래프를 그립니다.",
    graphNoPlan2:
      "이 빌드의 라이브러리에서는 페이지가 기울기를 한 계획으로 읽을 수 없어서 계획 2를 표시하지 않습니다.",
    graphBatch:
      "배치: 테스트 분할에서 무작위로 고른 샘플 {n}개 (시드 {seed}). 전후 매개변수 체크섬: {sum}.",
    graphStepNote:
      "학습 한 단계는 읽기를 두 번 합니다. 계획 1은 손실을 읽습니다. backward()는 기울기 연산을 기록만 합니다. 계획 2는 optimizer.step()처럼 모든 기울기를 한 계획으로 읽습니다. 이 화면은 계획 2를 실행하지만 매개변수는 바꾸지 않습니다.",
    graphTab1: "계획 1: 손실",
    graphTab2: "계획 2: 기울기",
    graphForwardTab: "계획: 순방향",
    graphError: "그래프를 그리지 못했습니다.",
    running: "실행이 진행 중입니다.",
    selectDataset: "데이터셋을 고르세요.",
    unit: "USD",
    fallback: "대체 실행",
  },
};

const DEFAULTS: Record<string, Omit<RunConfig, "dataset" | "tier" | "seed">> = {
  mnist: {
    model: "mlp",
    hidden: 128,
    optimizer: "adam",
    learningRate: 0.001,
    batchSize: 64,
    epochs: 1,
  },
  fashion_mnist: {
    model: "mlp",
    hidden: 128,
    optimizer: "adam",
    learningRate: 0.001,
    batchSize: 64,
    epochs: 1,
  },
  cifar10: {
    model: "cnn",
    hidden: 128,
    optimizer: "adam",
    learningRate: 0.001,
    batchSize: 64,
    epochs: 1,
  },
  cifar100: {
    model: "cnn",
    hidden: 128,
    optimizer: "adam",
    learningRate: 0.001,
    batchSize: 64,
    epochs: 1,
  },
  imdb: {
    model: "bag",
    hidden: 64,
    optimizer: "adam",
    learningRate: 0.001,
    batchSize: 128,
    epochs: 2,
  },
  reuters: {
    model: "bag",
    hidden: 64,
    optimizer: "adam",
    learningRate: 0.001,
    batchSize: 128,
    epochs: 2,
  },
  california_housing: {
    model: "regression",
    hidden: 64,
    optimizer: "adam",
    learningRate: 0.003,
    batchSize: 64,
    epochs: 20,
  },
};

const SEED = 1;

export function defaultConfig(dataset: string): RunConfig {
  const base = DEFAULTS[dataset] ?? DEFAULTS.mnist!;
  return { dataset, ...base, tier: "auto", seed: SEED };
}

interface Session {
  readonly model: Module;
  readonly device: Device;
  readonly data: LoadedDataset;
  readonly spec: ModelSpec;
  readonly scale: { mean: number; std: number } | undefined;
  readonly result: RunResult;
}

interface Progress extends StepInfo {
  readonly totalSteps: number;
  readonly done: number;
  readonly epochs: number;
}

export interface LabApi {
  ready: Promise<void>;
  datasets(): Promise<DatasetInfo[]>;
  run(config: Partial<RunConfig> & { dataset: string }): Promise<RunResult>;
  stop(): void;
  readonly results: RunResult[];
}

declare global {
  interface Window {
    __typennLab?: LabApi;
  }
}

const NN_PROMISE = () => import("../vendor/typenn/index.js") as Promise<NN>;

export default function Lab({ locale }: { locale: Locale }) {
  const t: Words = text[locale];
  const [dark, setDark] = useState(false);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [manifests, setManifests] = useState<Record<string, Manifest>>({});
  const [cached, setCached] = useState<Record<string, boolean>>({});
  const [selected, setSelected] = useState("mnist");
  const [form, setForm] = useState<RunConfig>(() => defaultConfig("mnist"));
  const [loading, setLoading] = useState<{
    id: string;
    loaded: number;
    total: number;
  } | null>(null);
  const [loadedIds, setLoadedIds] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [losses, setLosses] = useState<number[]>([]);
  const [metrics, setMetrics] = useState<number[]>([]);
  const [graphTabs, setGraphTabs] = useState<GraphTab[]>([]);
  const [graphTab, setGraphTab] = useState(0);
  const [graphBusy, setGraphBusy] = useState<GraphKind | null>(null);
  const [graphSource, setGraphSource] = useState<string | null>(null);
  const [graphKind, setGraphKind] = useState<GraphKind | null>(null);
  const [graphBatch, setGraphBatch] = useState<string | null>(null);
  const [graphError, setGraphError] = useState<string | null>(null);
  const [results, setResults] = useState<RunResult[]>([]);
  const [runError, setRunError] = useState<string | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [stopping, setStopping] = useState(false);
  const [activeTier, setActiveTier] = useState<string | null>(null);

  const memory = useRef(new Map<string, LoadedDataset>());
  const runningRef = useRef(false);
  const stopRef = useRef(false);
  const sessionRef = useRef<Session | null>(null);
  const resultsRef = useRef<RunResult[]>([]);
  const runRef = useRef<(config: RunConfig) => Promise<RunResult>>(async () => {
    throw new Error("The lab is not ready.");
  });
  const readyResolve = useRef<() => void>(() => {});
  const readyPromise = useRef<Promise<void>>(
    new Promise<void>((resolve) => (readyResolve.current = resolve)),
  );

  useEffect(() => {
    const syncTheme = () =>
      setDark(document.documentElement.dataset.theme === "dark");
    syncTheme();
    window.addEventListener("typenn-theme-change", syncTheme);
    return () => window.removeEventListener("typenn-theme-change", syncTheme);
  }, []);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const list = await listDatasets();
        if (!alive) return;
        setDatasets(list);
        const loaded = await Promise.all(
          list.map(
            async (info) => [info.id, await loadManifest(info.id)] as const,
          ),
        );
        if (!alive) return;
        const byId = Object.fromEntries(loaded);
        setManifests(byId);
        const flags = await Promise.all(
          list.map(
            async (info) =>
              [info.id, await isCached(info.id, byId[info.id]!)] as const,
          ),
        );
        if (alive) setCached(Object.fromEntries(flags));
      } catch (cause) {
        if (alive)
          setNotice(cause instanceof Error ? cause.message : String(cause));
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    const api: LabApi = {
      ready: readyPromise.current,
      datasets: () => listDatasets(),
      run: async (config) => {
        const full = { ...defaultConfig(config.dataset), ...config };
        return runRef.current(full);
      },
      stop: () => (stopRef.current = true),
      get results() {
        return resultsRef.current;
      },
    };
    window.__typennLab = api;
    readyResolve.current();
    return () => {
      delete window.__typennLab;
      const current = sessionRef.current;
      sessionRef.current = null;
      if (current) void current.device.dispose();
    };
  }, []);

  /** Drop the arrays of every dataset except `keep` and the one the live session uses. */
  function releaseDatasets(keep: string) {
    const inUse = sessionRef.current?.data.info.id;
    for (const id of [...memory.current.keys()])
      if (id !== keep && id !== inUse) memory.current.delete(id);
    setLoadedIds((ids) => ids.filter((id) => memory.current.has(id)));
  }

  async function ensureData(id: string): Promise<LoadedDataset> {
    const known = memory.current.get(id);
    if (known) return known;
    setLoading({ id, loaded: 0, total: 1 });
    try {
      const data = await loadDataset(id, {
        onProgress: (loaded, total) => setLoading({ id, loaded, total }),
      });
      memory.current.set(id, data);
      setLoadedIds((ids) => (ids.includes(id) ? ids : [...ids, id]));
      setCached((flags) => ({ ...flags, [id]: true }));
      return data;
    } finally {
      setLoading(null);
    }
  }

  async function runTraining(config: RunConfig): Promise<RunResult> {
    if (runningRef.current) throw new Error(t.running);
    runningRef.current = true;
    stopRef.current = false;
    setStopping(false);
    setRunning(true);
    setRunError(null);
    setLosses([]);
    setMetrics([]);
    setProgress(null);
    setActiveTier(null);
    setForm(config);
    setSelected(config.dataset);
    let device: Device | undefined;
    try {
      const data = await ensureData(config.dataset);
      const nn = await NN_PROMISE();
      const previous = sessionRef.current;
      sessionRef.current = null;
      setSession(null);
      if (previous) await previous.device.dispose();
      releaseDatasets(config.dataset);
      device = await nn.device({ device: config.tier });
      setActiveTier(device.backend.name);
      const trainDevice = device;
      const spec = specFor(data, config);
      const scale =
        config.model === "regression" ? targetScale(data.train) : undefined;
      let model: Module | undefined;
      const totalSteps =
        config.epochs * Math.ceil(data.train.count / config.batchSize);
      const result = await train(config, data, nn, trainDevice, {
        shouldStop: () => stopRef.current,
        onModel: (m) => (model = m),
        onStep: (info) => {
          const done = (info.epoch - 1) * info.stepsPerEpoch + info.step;
          setProgress({ ...info, totalSteps, done, epochs: config.epochs });
          setLosses((values) => [...values, info.loss]);
        },
        onEpoch: (info) => setMetrics((values) => [...values, info.testMetric]),
      });
      resultsRef.current = [...resultsRef.current, result];
      setResults(resultsRef.current);
      if (model) {
        const next: Session = {
          model,
          device: trainDevice,
          data,
          spec,
          scale,
          result,
        };
        sessionRef.current = next;
        setSession(next);
        device = undefined;
      }
      return result;
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause);
      setRunError(message);
      throw cause;
    } finally {
      if (device) await device.dispose().catch(() => undefined);
      runningRef.current = false;
      setRunning(false);
      setStopping(false);
    }
  }
  runRef.current = runTraining;

  /**
   * Draw the graph of one read, on a seeded random batch of the test split. The forward reads the
   * logits. The step reads the loss (plan 1), runs backward(), then reads all gradients in one
   * plan (plan 2), as optimizer.step() does, but updates nothing. A trained model of this page is
   * used when its dataset is selected, and its parameters stay bit-identical; otherwise a fresh
   * model is built and disposed afterwards.
   */
  async function captureGraph(kind: GraphKind) {
    if (runningRef.current || graphBusy) return;
    setGraphBusy(kind);
    setGraphError(null);
    let fresh: Device | undefined;
    try {
      const nn = await NN_PROMISE();
      const data = await ensureData(selected);
      const current = sessionRef.current;
      const trained = current !== null && current.data.info.id === selected;
      const config = form.dataset === selected ? form : defaultConfig(selected);
      let model: Module;
      let device: Device;
      let spec: ModelSpec;
      if (current && trained) {
        model = current.model;
        device = current.device;
        spec = current.spec;
      } else {
        spec = specFor(data, config);
        fresh = await nn.device({ device: config.tier });
        device = fresh;
        model = buildModel(spec, nn).to(device);
      }
      const parameters = model.parameters();
      const wasTraining = model.isTraining();
      const n = Math.min(config.batchSize, data.test.count);
      const indices = sampleIndices(
        data.test.count,
        n,
        mulberry32(config.seed),
      );
      const size = inputSize(spec);
      const batch = fillInputs(
        data.test,
        indices,
        spec,
        new Float32Array(n * size),
      );
      const input = device.tensor(batch.subarray(0, n * size), {
        shape: [n, ...spec.inputShape],
      });
      const source = trained ? t.graphTrained : t.graphFresh;
      const before = await parameterChecksum(parameters);
      // An optimizer that reads the gradients and updates nothing: zeroGrad() frees them.
      const reader = nn.optim.SGD(parameters, { lr: 1 });
      try {
        if (kind === "forward") {
          model.eval();
          const logits = model.forward(input);
          const snapshot = nn.graph(logits);
          await logits.data();
          const plan = device.lastPlan;
          await logits.disposeGraph(parameters);
          setGraphTabs([{ title: t.graphForwardTab, snapshot, plan }]);
        } else {
          model.train();
          await reader.zeroGrad();
          const logits = model.forward(input);
          const loss =
            spec.preset === "regression"
              ? (() => {
                  const scale = targetScale(data.train);
                  return new nn.MSELoss().forward(
                    logits,
                    Float32Array.from(
                      indices,
                      (i) => (data.test.y[i]! - scale.mean) / scale.std,
                    ),
                  );
                })()
              : new nn.CrossEntropyLoss().forward(
                  logits,
                  labels(data.test, indices),
                );
          const snapshot = nn.graph(loss);
          await loss.data();
          const forwardPlan = device.lastPlan;
          // backward() is lazy. optimizer.step() reads every gradient in one plan: plan 2.
          await loss.backward();
          const gradients = parameters.flatMap((p) => (p.grad ? [p.grad] : []));
          const gradientSnapshot = mergeSnapshots(gradients.map(nn.graph));
          const read = await readGradients(reader);
          const gradientPlan = read ? device.lastPlan : undefined;
          await loss.disposeGraph(parameters);
          await reader.zeroGrad();
          const tabs: GraphTab[] = [
            { title: t.graphTab1, snapshot, plan: forwardPlan },
          ];
          if (gradientPlan && gradientPlan !== forwardPlan)
            tabs.push({
              title: t.graphTab2,
              snapshot: gradientSnapshot,
              plan: gradientPlan,
            });
          setGraphTabs(tabs);
        }
        const after = await parameterChecksum(parameters);
        if (after !== before)
          throw new Error(
            `The parameters changed (checksum ${before} to ${after}).`,
          );
        setGraphTab(0);
        setGraphKind(kind);
        setGraphSource(source);
        setGraphBatch(
          t.graphBatch
            .replace("{n}", String(n))
            .replace("{seed}", String(config.seed))
            .replace("{sum}", after),
        );
      } finally {
        await reader.zeroGrad().catch(() => undefined);
        if (wasTraining) model.train();
        else model.eval();
      }
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause);
      setGraphError(`${t.graphError} ${message}`);
    } finally {
      if (fresh) await fresh.dispose().catch(() => undefined);
      setGraphBusy(null);
    }
  }

  function selectDataset(id: string) {
    if (running) return;
    releaseDatasets(id);
    setSelected(id);
    setForm(defaultConfig(id));
  }

  const info = datasets.find((d) => d.id === selected);
  const manifest = manifests[selected];
  const presets = manifest ? presetsFor(manifest.task) : [];
  const words: Words = t;

  return (
    <ConfigProvider
      theme={{
        algorithm: dark ? antdTheme.darkAlgorithm : antdTheme.defaultAlgorithm,
        token: {
          colorPrimary: "#1677ff",
          borderRadius: 6,
          fontFamily: '"IBM Plex Sans", "IBM Plex Sans KR", sans-serif',
        },
      }}
    >
      <div className="lab-app">
        <p className="lab-intro">{words.intro}</p>
        {notice && (
          <p className="lab-error" role="alert">
            {notice}
          </p>
        )}

        <section className="lab-section" aria-labelledby="lab-dataset">
          <h2 id="lab-dataset">{words.datasetTitle}</h2>
          <div className="lab-cards">
            {datasets.map((d) => {
              const m = manifests[d.id];
              const isLoading = loading?.id === d.id;
              const percent =
                isLoading && loading.total > 0
                  ? Math.round((100 * loading.loaded) / loading.total)
                  : 0;
              return (
                <article
                  key={d.id}
                  className={`lab-card${selected === d.id ? " is-selected" : ""}`}
                >
                  <button
                    type="button"
                    className="lab-card-select"
                    aria-pressed={selected === d.id}
                    onClick={() => selectDataset(d.id)}
                    disabled={running}
                  >
                    <span className="lab-card-title">{d.title}</span>
                    <span className="lab-card-task">{d.task}</span>
                  </button>
                  {m && (
                    <dl className="lab-facts">
                      <dt>{words.counts}</dt>
                      <dd>
                        {m.splits.train.count.toLocaleString()} /{" "}
                        {m.splits.test.count.toLocaleString()}
                      </dd>
                      <dt>{words.bytes}</dt>
                      <dd>{formatBytes(d.totalBytes)}</dd>
                      <dt>{words.license}</dt>
                      <dd>{m.license}</dd>
                      <dt>{words.citation}</dt>
                      <dd>{m.citation}</dd>
                    </dl>
                  )}
                  <div className="lab-card-actions">
                    {isLoading ? (
                      <Progress
                        percent={percent}
                        size="small"
                        format={() =>
                          `${words.downloading} ${formatBytes(loading.loaded)} / ${formatBytes(loading.total)} (${percent}%)`
                        }
                      />
                    ) : loadedIds.includes(d.id) ? (
                      <span className="lab-note">{words.loaded}</span>
                    ) : (
                      <Button
                        size="small"
                        onClick={() =>
                          void ensureData(d.id).catch((cause) =>
                            setNotice(String(cause)),
                          )
                        }
                        disabled={running || loading !== null}
                      >
                        {words.download}
                      </Button>
                    )}
                    <span className="lab-note">
                      {cached[d.id] ? words.cachedNote : words.notCached}
                    </span>
                  </div>
                </article>
              );
            })}
          </div>
        </section>

        <section className="lab-section" aria-labelledby="lab-model">
          <h2 id="lab-model">{words.modelTitle}</h2>
          <div className="lab-form">
            <label className="lab-field">
              {words.preset}
              <Select<Preset>
                value={form.model}
                disabled={running || presets.length < 2}
                onChange={(value) => setForm({ ...form, model: value })}
                options={(presets.length ? presets : [form.model]).map((p) => ({
                  value: p,
                  label: words[p],
                }))}
              />
            </label>
            <label className="lab-field">
              {words.hidden}
              <InputNumber
                min={1}
                max={4096}
                step={16}
                value={form.hidden}
                disabled={running || form.model === "cnn"}
                onChange={(v) => v && setForm({ ...form, hidden: v })}
              />
            </label>
            <label className="lab-field">
              {words.optimizer}
              <Select
                value={form.optimizer}
                disabled={running}
                onChange={(v) => setForm({ ...form, optimizer: v })}
                options={[
                  { value: "adam", label: "Adam" },
                  { value: "sgd", label: "SGD" },
                ]}
              />
            </label>
            <label className="lab-field">
              {words.learningRate}
              <InputNumber
                min={0.00001}
                max={1}
                step={0.0001}
                value={form.learningRate}
                disabled={running}
                onChange={(v) => v && setForm({ ...form, learningRate: v })}
              />
            </label>
            <label className="lab-field">
              {words.batchSize}
              <InputNumber
                min={1}
                max={1024}
                value={form.batchSize}
                disabled={running}
                onChange={(v) => v && setForm({ ...form, batchSize: v })}
              />
            </label>
            <label className="lab-field">
              {words.epochs}
              <InputNumber
                min={1}
                max={100}
                value={form.epochs}
                disabled={running}
                onChange={(v) => v && setForm({ ...form, epochs: v })}
              />
            </label>
            <label className="lab-field">
              {words.tier}
              <Select<TierPreference>
                value={form.tier}
                disabled={running}
                onChange={(v) => setForm({ ...form, tier: v })}
                options={[
                  { value: "auto", label: words.auto },
                  { value: "cpu", label: words.cpu },
                  { value: "webgpu", label: words.webgpu },
                  { value: "webgl2", label: words.webgl2 },
                ]}
              />
            </label>
          </div>
          {info && (
            <p className="lab-note">
              {info.id} · {form.model}
            </p>
          )}
          {(form.tier === "cpu" || activeTier === "cpu") &&
            (form.model === "cnn" || form.model === "bag") && (
              <p className="lab-note lab-warning" role="note">
                {words.slowCpu}
              </p>
            )}
        </section>

        <section className="lab-section" aria-labelledby="lab-run">
          <h2 id="lab-run">{words.runTitle}</h2>
          <div className="lab-run-bar">
            {running ? (
              <Button
                danger
                onClick={() => {
                  stopRef.current = true;
                  setStopping(true);
                }}
                disabled={stopping}
              >
                {stopping ? words.stopping : words.stop}
              </Button>
            ) : (
              <Button
                type="primary"
                onClick={() =>
                  void runTraining({ ...form, dataset: selected }).catch(
                    () => undefined,
                  )
                }
                disabled={!info}
              >
                {words.start}
              </Button>
            )}
            <span className="lab-status" role="status" aria-live="polite">
              {runError ? (
                <strong className="lab-error">
                  {words.errorTitle}: {runError}
                </strong>
              ) : progress ? (
                statusLine(progress, words)
              ) : running ? (
                words.preparing
              ) : (
                words.idle
              )}
            </span>
          </div>
          {progress && (
            <Progress
              percent={Math.round((100 * progress.done) / progress.totalSteps)}
              size="small"
            />
          )}
          <div className="lab-charts">
            <figure>
              <figcaption>{words.lossChart}</figcaption>
              <LineChart
                values={losses}
                smooth
                axisX={words.stepAxis}
                label={words.lossChart}
                empty={words.emptyLoss}
              />
            </figure>
            <figure>
              <figcaption>{words.metricChart}</figcaption>
              <LineChart
                values={metrics}
                axisX={words.epochAxis}
                label={words.metricChart}
                empty={words.emptyMetric}
                points
                lowerIsBetter={form.model === "regression"}
              />
            </figure>
          </div>
        </section>

        <section className="lab-section" aria-labelledby="lab-graph">
          <h2 id="lab-graph">{words.graphTitle}</h2>
          <p className="lab-note">
            {words.graphIntro}{" "}
            <a
              className="text-link"
              href="https://github.com/typeshade/typenn#how-a-read-runs"
            >
              {words.graphReadme} ↗
            </a>
          </p>
          <div className="lab-graph-actions">
            <Button
              onClick={() => void captureGraph("forward")}
              disabled={running || graphBusy !== null}
              loading={graphBusy === "forward"}
            >
              {words.graphForward}
            </Button>
            <Button
              onClick={() => void captureGraph("step")}
              disabled={running || graphBusy !== null}
              loading={graphBusy === "step"}
            >
              {words.graphStep}
            </Button>
          </div>
          {graphSource && (
            <p className="lab-note" data-testid="graph-source">
              {graphSource} {graphBatch}
            </p>
          )}
          {graphError && (
            <p className="lab-error" role="alert">
              {graphError}
            </p>
          )}
          {graphTabs.length > 1 && (
            <div className="kg-tabs" role="tablist">
              {graphTabs.map((tab, index) => (
                <button
                  key={tab.title}
                  type="button"
                  role="tab"
                  aria-selected={graphTab === index}
                  onClick={() => setGraphTab(index)}
                >
                  {tab.title}
                </button>
              ))}
            </div>
          )}
          {graphTabs[graphTab] ? (
            <KernelGraph
              key={`${graphTab}-${graphTabs[graphTab]!.title}`}
              snapshot={graphTabs[graphTab]!.snapshot}
              plan={graphTabs[graphTab]!.plan}
              locale={locale}
              label={`${words.graphTitle}: ${graphTabs[graphTab]!.title}`}
            />
          ) : (
            <p className="lab-note">{words.graphEmpty}</p>
          )}
          {graphKind === "step" && (
            <p className="lab-note">
              {words.graphStepNote}
              {graphTabs.length < 2 && ` ${words.graphNoPlan2}`}
            </p>
          )}
        </section>

        <section className="lab-section" aria-labelledby="lab-results">
          <h2 id="lab-results">{words.resultsTitle}</h2>
          {results.length === 0 ? (
            <p className="lab-note">{words.noResults}</p>
          ) : (
            <Table
              size="small"
              pagination={false}
              rowKey={(row) => `${row.finishedAt}-${row.config.dataset}`}
              dataSource={results}
              columns={[
                {
                  title: words.col.dataset,
                  render: (_, r) => r.config.dataset,
                },
                { title: words.col.model, render: (_, r) => r.config.model },
                {
                  title: words.col.tier,
                  render: (_, r) =>
                    r.fallbackReason ? `${r.tier} (${words.fallback})` : r.tier,
                },
                {
                  title: words.col.epochs,
                  render: (_, r) => `${r.epochs.length}/${r.config.epochs}`,
                },
                {
                  title: words.col.metric,
                  render: (_, r) =>
                    formatTest(r.test, r.config.model === "regression", locale),
                },
                {
                  title: words.col.time,
                  render: (_, r) => r.seconds.toFixed(1),
                },
                {
                  title: words.col.status,
                  render: (_, r) =>
                    r.completed ? words.completed : words.stopped,
                },
              ]}
            />
          )}
          <div className="lab-run-bar">
            <Button
              onClick={() =>
                void copyJson(resultsRef.current, words, setNotice)
              }
              disabled={results.length === 0}
            >
              {words.copyJson}
            </Button>
          </div>
        </section>

        <section className="lab-section" aria-labelledby="lab-try">
          <h2 id="lab-try">{words.tryTitle}</h2>
          {!session ? (
            <p className="lab-note">{words.noModel}</p>
          ) : (
            <TryModel
              session={session}
              words={words}
              locale={locale}
              manifest={manifests[session.data.info.id]!}
            />
          )}
        </section>
      </div>
    </ConfigProvider>
  );
}

function TryModel({
  session,
  words,
  locale,
  manifest,
}: {
  session: Session;
  words: Words;
  locale: Locale;
  manifest: Manifest;
}) {
  const { data } = session;
  const task = manifest.task;
  return (
    <div className="lab-try">
      <p className="lab-note">
        {words.trainedOn}: {data.info.title}
      </p>
      {task === "image-classification" && (
        <ImageTry session={session} words={words} />
      )}
      {task === "text-classification" && (
        <TextTry session={session} words={words} />
      )}
      {task === "regression" && (
        <RegressionTry session={session} words={words} locale={locale} />
      )}
    </div>
  );
}

function ImageTry({ session, words }: { session: Session; words: Words }) {
  const { data, model, device, spec } = session;
  const [h, w, c] = spec.inputShape as [number, number, number];
  const drawable = data.info.id === "mnist" || data.info.id === "fashion_mnist";
  const canvas = useRef<HTMLCanvasElement | null>(null);
  const drawing = useRef(false);
  const [truth, setTruth] = useState<number | null>(null);
  const [top, setTop] = useState<{ label: string; p: number }[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const classes = data.manifest.classes ?? [];

  useEffect(() => {
    const el = canvas.current;
    const ctx = el?.getContext("2d", { willReadFrequently: true });
    if (el && ctx) {
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, el.width, el.height);
    }
  }, [session]);

  function point(event: ReactPointerEvent<HTMLCanvasElement>) {
    const el = canvas.current!;
    const rect = el.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) / rect.width) * el.width,
      y: ((event.clientY - rect.top) / rect.height) * el.height,
    };
  }

  function paint(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (!drawing.current || !canvas.current) return;
    const ctx = canvas.current.getContext("2d", { willReadFrequently: true })!;
    const { x, y } = point(event);
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.arc(x, y, 1.1, 0, Math.PI * 2);
    ctx.fill();
    setTruth(null);
  }

  function clear() {
    const el = canvas.current;
    const ctx = el?.getContext("2d", { willReadFrequently: true });
    if (el && ctx) {
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, el.width, el.height);
    }
    setTruth(null);
    setTop(null);
  }

  function randomTest() {
    const el = canvas.current;
    const ctx = el?.getContext("2d", { willReadFrequently: true });
    if (!el || !ctx) return;
    const index = Math.floor(Math.random() * data.test.count);
    const size = h * w * c;
    const image = ctx.createImageData(w, h);
    for (let p = 0; p < h * w; p++) {
      for (let k = 0; k < 3; k++)
        image.data[p * 4 + k] =
          data.test.x[index * size + p * c + (c === 3 ? k : 0)]!;
      image.data[p * 4 + 3] = 255;
    }
    el.width = w;
    el.height = h;
    ctx.putImageData(image, 0, 0);
    setTruth(data.test.y[index]!);
    setTop(null);
    void classify();
  }

  async function classify() {
    const el = canvas.current;
    const ctx = el?.getContext("2d", { willReadFrequently: true });
    if (!el || !ctx) return;
    try {
      setError(null);
      const pixels = ctx.getImageData(0, 0, w, h).data;
      const values = new Float32Array(h * w * c);
      for (let p = 0; p < h * w; p++)
        for (let k = 0; k < c; k++)
          values[p * c + k] = pixels[p * 4 + (c === 3 ? k : 0)]! / 255;
      const probs = await predict(
        model,
        device,
        { values, shape: [1, h, w, c] },
        "probabilities",
      );
      setTop(topK(probs, classes, 3));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }

  const displayScale = drawable ? 10 : 8;
  return (
    <div className="lab-image-try">
      <canvas
        ref={canvas}
        width={w}
        height={h}
        className="lab-canvas"
        style={{ width: w * displayScale, maxWidth: "100%", height: "auto" }}
        aria-label={drawable ? words.draw : data.info.title}
        onPointerDown={(e) => {
          if (!drawable) return;
          drawing.current = true;
          e.currentTarget.setPointerCapture(e.pointerId);
          paint(e);
        }}
        onPointerMove={(e) => paint(e)}
        onPointerUp={() => {
          drawing.current = false;
        }}
        onPointerCancel={() => {
          drawing.current = false;
        }}
      />
      <div className="lab-try-side">
        {drawable ? <p className="lab-note">{words.draw}</p> : null}
        <div className="lab-run-bar">
          {drawable && <Button onClick={clear}>{words.clear}</Button>}
          <Button onClick={randomTest}>{words.randomTest}</Button>
          <Button type="primary" onClick={() => void classify()}>
            {words.classify}
          </Button>
        </div>
        {truth !== null && (
          <p className="lab-note">
            {words.trueLabel}: {classes[truth] ?? truth}
          </p>
        )}
        {error && <p className="lab-error">{error}</p>}
        {top && <TopList title={words.top} rows={top} />}
      </div>
    </div>
  );
}

function TextTry({ session, words }: { session: Session; words: Words }) {
  const { data, model, device, spec } = session;
  const [review, setReview] = useState("");
  const [rows, setRows] = useState<{ label: string; p: number }[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const vocab = useRef<{ index: Map<string, number>; list: string[] } | null>(
    null,
  );
  const classes = data.manifest.classes ?? [];

  async function classify() {
    try {
      setError(null);
      if (!vocab.current) {
        const file = data.manifest.vocabularyFile?.file ?? "vocabulary.json";
        const list = (await (
          await fetch(`/data/${data.info.id}/${file}`)
        ).json()) as string[];
        vocab.current = { list, index: wordIndex(list) };
      }
      const encoding = data.manifest.tokenEncoding!;
      const ids = encode(review, vocab.current.index, encoding);
      const size = spec.inputShape[0]!;
      const values = bagOf(ids, size);
      const probs = await predict(
        model,
        device,
        { values, shape: [1, size] },
        "probabilities",
      );
      setRows(topK(probs, classes, classes.length <= 3 ? classes.length : 3));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }

  return (
    <div className="lab-text-try">
      <label className="lab-field lab-wide">
        {words.reviewPlaceholder}
        <Input.TextArea
          rows={4}
          value={review}
          onChange={(e) => setReview(e.target.value)}
          placeholder={words.reviewPlaceholder}
        />
      </label>
      <div className="lab-run-bar">
        <Button type="primary" onClick={() => void classify()}>
          {words.classify}
        </Button>
      </div>
      {error && <p className="lab-error">{error}</p>}
      {rows && <TopList title={words.probabilities} rows={rows} />}
    </div>
  );
}

function RegressionTry({
  session,
  words,
  locale,
}: {
  session: Session;
  words: Words;
  locale: Locale;
}) {
  const { data, model, device, spec, scale } = session;
  const features = data.manifest.features ?? [];
  const [values, setValues] = useState<string[]>(() =>
    rowValues(data, 0, features.length),
  );
  const [row, setRow] = useState<number | null>(0);
  const [predicted, setPredicted] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const money = (v: number) =>
    `${v.toLocaleString(locale === "ko" ? "ko-KR" : "en-US", { maximumFractionDigits: 0 })} ${words.unit}`;

  function randomRow() {
    const index = Math.floor(Math.random() * data.test.count);
    setRow(index);
    setValues(rowValues(data, index, features.length));
    setPredicted(null);
  }

  async function predictPrice() {
    try {
      setError(null);
      const mean = spec.features!.mean;
      const std = spec.features!.std;
      const input = new Float32Array(features.length);
      values.forEach((value, f) => {
        const raw = Number(value);
        input[f] = (raw - mean[f]!) / (std[f] === 0 ? 1 : std[f]!);
      });
      const [out] = await predict(
        model,
        device,
        { values: input, shape: [1, features.length] },
        "value",
      );
      setPredicted(out! * scale!.std + scale!.mean);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }

  const actual = row === null ? null : data.test.y[row]!;
  return (
    <div className="lab-regression-try">
      <p className="lab-note">{words.features}</p>
      <div className="lab-features">
        {features.map((name, f) => (
          <label key={name} className="lab-field">
            {name}
            <Input
              value={values[f]}
              onChange={(e) =>
                setValues(values.map((v, i) => (i === f ? e.target.value : v)))
              }
              inputMode="decimal"
            />
          </label>
        ))}
      </div>
      <div className="lab-run-bar">
        <Button onClick={randomRow}>{words.randomRow}</Button>
        <Button type="primary" onClick={() => void predictPrice()}>
          {words.predictedPrice}
        </Button>
      </div>
      {error && <p className="lab-error">{error}</p>}
      {predicted !== null && (
        <p className="lab-result">
          <strong>
            {words.predictedPrice}: {money(predicted)}
          </strong>
          {actual !== null && row !== null ? (
            <span>
              {" "}
              · {words.actualPrice}: {money(actual)}
            </span>
          ) : null}
        </p>
      )}
    </div>
  );
}

/**
 * Read all gradients in one plan, the way optimizer.step() reads them, and update nothing. The
 * library has no public call for this: `gradients()` is a protected method of its optimizers.
 * Returns false when this build of the library has no such method.
 */
async function readGradients(optimizer: object): Promise<boolean> {
  const read = (optimizer as { gradients?: () => Promise<unknown> }).gradients;
  if (typeof read !== "function") return false;
  await read.call(optimizer);
  return true;
}

/** FNV-1a over the bits of every parameter, as eight hex digits. Equal bits give equal sums. */
async function parameterChecksum(
  parameters: readonly { data(): Promise<Float32Array> }[],
): Promise<string> {
  let hash = 0x811c9dc5;
  for (const parameter of parameters) {
    const bits = new Uint32Array((await parameter.data()).slice().buffer);
    for (const word of bits) {
      hash = Math.imul(hash ^ word, 0x01000193) >>> 0;
    }
  }
  return hash.toString(16).padStart(8, "0");
}

function rowValues(data: LoadedDataset, row: number, count: number): string[] {
  return Array.from({ length: count }, (_, f) =>
    String(data.test.x[row * count + f]),
  );
}

function TopList({
  title,
  rows,
}: {
  title: string;
  rows: { label: string; p: number }[];
}) {
  return (
    <div className="lab-top">
      <p className="lab-note">{title}</p>
      <ol>
        {rows.map((row) => (
          <li key={row.label}>
            <span>{row.label}</span>
            <span>{(100 * row.p).toFixed(1)}%</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

function topK(probs: Float32Array, classes: readonly string[], k: number) {
  return Array.from(probs, (p, i) => ({ i, p }))
    .sort((a, b) => b.p - a.p)
    .slice(0, k)
    .map(({ i, p }) => ({ label: classes[i] ?? String(i), p }));
}

function LineChart({
  values,
  smooth = false,
  points = false,
  axisX,
  label,
  lowerIsBetter = false,
  empty,
}: {
  empty: string;
  values: number[];
  smooth?: boolean;
  points?: boolean;
  axisX: string;
  label: string;
  lowerIsBetter?: boolean;
}) {
  const width = 560;
  const height = 200;
  const pad = { left: 52, right: 12, top: 10, bottom: 28 };
  const series = useMemo(() => {
    if (!smooth) return values;
    let ema = values[0] ?? 0;
    return values.map((v) => (ema = 0.9 * ema + 0.1 * v));
  }, [values, smooth]);
  if (series.length === 0) {
    return (
      <svg
        className="lab-chart"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={label}
      >
        <text
          x={width / 2}
          y={height / 2}
          textAnchor="middle"
          className="lab-chart-empty"
        >
          {empty}
        </text>
      </svg>
    );
  }
  const min = Math.min(...series);
  const max = Math.max(...series);
  const span = max - min || 1;
  const n = Math.max(series.length - 1, 1);
  const x = (i: number) => pad.left + (i / n) * (width - pad.left - pad.right);
  const y = (v: number) =>
    pad.top + (1 - (v - min) / span) * (height - pad.top - pad.bottom);
  const path = series
    .map(
      (v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(v).toFixed(1)}`,
    )
    .join(" ");
  const ticks = [0, 0.5, 1].map((f) => min + f * span);
  return (
    <svg
      className="lab-chart"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`${label}, ${series.length}`}
    >
      {ticks.map((tick) => (
        <g key={tick}>
          <line
            x1={pad.left}
            x2={width - pad.right}
            y1={y(tick)}
            y2={y(tick)}
            className="lab-grid"
          />
          <text
            x={pad.left - 6}
            y={y(tick) + 4}
            textAnchor="end"
            className="lab-axis"
          >
            {formatTick(tick)}
          </text>
        </g>
      ))}
      <text
        x={width - pad.right}
        y={height - 6}
        textAnchor="end"
        className="lab-axis"
      >
        {axisX}
        {lowerIsBetter ? " ↓" : ""}: {series.length}
      </text>
      <path d={path} className="lab-line" fill="none" />
      {points &&
        series.map((v, i) => (
          <circle key={i} cx={x(i)} cy={y(v)} r={3.5} className="lab-dot" />
        ))}
    </svg>
  );
}

function formatTick(v: number) {
  if (Math.abs(v) >= 1000) return Math.round(v).toLocaleString("en-US");
  return v.toPrecision(3);
}

function statusLine(p: Progress, words: Words) {
  const elapsed = formatDuration(p.elapsedSeconds);
  const rate = p.done / Math.max(p.elapsedSeconds, 1e-9);
  const eta = formatDuration((p.totalSteps - p.done) / Math.max(rate, 1e-9));
  const status = String(words.status)
    .replace("{epoch}", String(p.epoch))
    .replace("{epochs}", String(p.epochs))
    .replace("{step}", String(p.step))
    .replace("{steps}", String(p.stepsPerEpoch));
  return `${status} · ${Math.round(p.samplesPerSecond)} ${words.rate} · ${words.eta} ${eta} · ${words.elapsed} ${elapsed}`;
}

function formatDuration(seconds: number) {
  if (!Number.isFinite(seconds)) return "--:--";
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function formatBytes(bytes: number) {
  if (bytes >= 1 << 20) return `${(bytes / (1 << 20)).toFixed(1)} MiB`;
  return `${(bytes / 1024).toFixed(0)} KiB`;
}

function formatTest(test: Metrics, regression: boolean, locale: Locale) {
  if (test.rmse === undefined && test.accuracy === undefined) return "-";
  if (regression)
    return `RMSE ${Math.round(test.rmse ?? 0).toLocaleString(locale === "ko" ? "ko-KR" : "en-US")} USD · MAE ${Math.round(test.mae ?? 0).toLocaleString(locale === "ko" ? "ko-KR" : "en-US")} USD`;
  return `${((test.accuracy ?? 0) * 100).toFixed(2)}%`;
}

async function copyJson(
  results: RunResult[],
  words: Words,
  notice: (message: string | null) => void,
) {
  try {
    await navigator.clipboard.writeText(JSON.stringify(results, null, 2));
    notice(words.copied);
  } catch {
    notice(words.copyFailed);
  }
}
