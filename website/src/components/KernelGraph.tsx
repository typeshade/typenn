import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";
import {
  groupNodes,
  layoutGraph,
  type GraphLayout,
  type LayoutOptions,
} from "../lab/graph-layout";
import type {
  DevicePlan,
  GraphNode,
  GraphSnapshot,
} from "../vendor/typenn/index.js";

type Locale = "en" | "ko";

const GROUP_COUNT = 8;
const MIN_ZOOM = 0.2;
const MAX_ZOOM = 3;

const text = {
  en: {
    region: "Kernel graph",
    tier: "Tier",
    dispatches: "Dispatches",
    submissions: "Submissions",
    calls: "Kernel calls",
    nodes: "Nodes",
    zoomIn: "Zoom in",
    zoomOut: "Zoom out",
    fit: "Fit",
    input: "input",
    param: "param",
    computed: "already computed",
    notRun: "not in this plan",
    inputs: "Inputs",
    group: "Group",
    dispatch: "Dispatch",
    callGroup: "Call group",
    fused: "fused",
    notFused: "not fused",
    why: "Why it starts here",
    noPlan: "No plan yet. Run the graph to see it.",
    noSnapshot: "The plan has no graph view. Its groups are listed below.",
    legend:
      "Each box is one tensor. A fused group runs its calls in one dispatch on WebGPU. Numbers mark the groups, so colour is not the only cue.",
    unaligned:
      "The plan and the graph do not line up. The groups are shown without the mapping.",
    none: "none",
    readme: "How a read runs",
    rows: "Drag to pan. Ctrl or ⌘ + wheel, or the buttons, to zoom.",
  },
  ko: {
    region: "커널 그래프",
    tier: "계층",
    dispatches: "디스패치",
    submissions: "제출",
    calls: "커널 호출",
    nodes: "노드",
    zoomIn: "확대",
    zoomOut: "축소",
    fit: "맞춤",
    input: "입력",
    param: "매개변수",
    computed: "이미 계산됨",
    notRun: "이 계획에 없음",
    inputs: "입력",
    group: "그룹",
    dispatch: "디스패치",
    callGroup: "호출 묶음",
    fused: "융합됨",
    notFused: "융합되지 않음",
    why: "이 지점에서 시작하는 이유",
    noPlan: "아직 계획이 없습니다. 그래프를 실행하면 표시됩니다.",
    noSnapshot: "이 계획에는 그래프 보기가 없습니다. 그룹을 아래에 나열합니다.",
    legend:
      "상자 하나는 Tensor 하나입니다. 융합된 그룹은 WebGPU에서 호출을 한 번의 디스패치로 실행합니다. 색만으로 구분하지 않도록 그룹에 번호를 붙였습니다.",
    unaligned: "계획과 그래프가 맞지 않습니다. 그룹은 대응 없이 표시합니다.",
    none: "없음",
    readme: "read 실행 방식",
    rows: "끌어서 이동합니다. Ctrl 또는 ⌘ + 휠이나 버튼으로 확대합니다.",
  },
};

type Words = (typeof text)["en"] & {
  readonly callCount: (n: number) => string;
};

/** "2 calls" in English, "호출 2개" in Korean. */
function callCount(n: number, locale: Locale): string {
  if (locale === "ko") return `호출 ${n}개`;
  return `${n} ${n === 1 ? "call" : "calls"}`;
}

/** "Dispatch" when the plan ran dispatches (WebGPU, WebGL2), "Call group" when it did not (CPU). */
function groupWord(
  plan: DevicePlan | undefined,
  words: (typeof text)["en"],
): string {
  return plan && plan.report.dispatches > 0 ? words.dispatch : words.callGroup;
}

const LAYOUT: LayoutOptions = {
  nodeWidth: 156,
  nodeHeight: 58,
  gapX: 56,
  gapY: 14,
  sweeps: 4,
  padding: 16,
};
const COMPACT: LayoutOptions = {
  nodeWidth: 116,
  nodeHeight: 48,
  gapX: 34,
  gapY: 10,
  sweeps: 4,
  padding: 12,
};

interface View {
  readonly k: number;
  readonly x: number;
  readonly y: number;
}

export interface KernelGraphProps {
  readonly snapshot: GraphSnapshot | null;
  readonly plan: DevicePlan | undefined;
  readonly locale: Locale;
  /** The name of the graph, read by screen readers. */
  readonly label: string;
  readonly compact?: boolean;
}

/**
 * The graph of one snapshot, coloured by the group of the plan that ran each op. Pan by drag,
 * zoom by wheel or buttons. The tooltip of a node shows on hover and on keyboard focus.
 */
export default function KernelGraph({
  snapshot,
  plan,
  locale,
  label,
  compact = false,
}: KernelGraphProps) {
  const words: Words = {
    ...text[locale],
    group: groupWord(plan, text[locale]),
    callCount: (n) => callCount(n, locale),
  };
  const options = compact ? COMPACT : LAYOUT;
  const mapping = useMemo(
    () => groupNodes(snapshot ?? { nodes: [] }, plan),
    [snapshot, plan],
  );
  const layout = useMemo<GraphLayout | null>(
    () =>
      snapshot
        ? layoutGraph(
            snapshot.nodes.map((n) => ({ id: n.id, inputs: n.inputs })),
            options,
          )
        : null,
    [snapshot, options],
  );
  const callNames = useMemo(() => {
    const names = new Map<number, string>();
    if (!plan) return names;
    let call = 0;
    for (const group of plan.report.groups)
      for (const name of group.calls) {
        const id = plan.nodes[call++];
        // A scratch copy comes before the call that reads it, so the later name wins.
        if (id !== undefined) names.set(id, name);
      }
    return names;
  }, [plan]);

  const viewport = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [view, setView] = useState<View>({ k: 1, x: 0, y: 0 });
  const [active, setActive] = useState<number | null>(null);
  const drag = useRef<{ x: number; y: number; vx: number; vy: number } | null>(
    null,
  );

  useLayoutEffect(() => {
    const element = viewport.current;
    if (!element) return;
    const measure = () =>
      setSize({ w: element.clientWidth, h: element.clientHeight });
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const fit = useCallback(() => {
    if (!layout || size.w === 0 || layout.width === 0) return;
    const k = Math.min(
      1,
      (size.w - 16) / layout.width,
      (size.h - 16) / layout.height,
    );
    setView({
      k,
      x: (size.w - layout.width * k) / 2,
      y: (size.h - layout.height * k) / 2,
    });
  }, [layout, size]);

  useEffect(() => {
    fit();
  }, [fit]);

  useEffect(() => {
    const element = viewport.current;
    if (!element) return;
    const onWheel = (event: WheelEvent) => {
      // A plain wheel scrolls the page. Ctrl or ⌘ + wheel (a trackpad pinch too) zooms the graph.
      if (!event.ctrlKey && !event.metaKey) return;
      event.preventDefault();
      const rect = element.getBoundingClientRect();
      const cx = event.clientX - rect.left;
      const cy = event.clientY - rect.top;
      setView((v) => {
        const k = clamp(v.k * (event.deltaY < 0 ? 1.1 : 1 / 1.1));
        return {
          k,
          x: cx - ((cx - v.x) * k) / v.k,
          y: cy - ((cy - v.y) * k) / v.k,
        };
      });
    };
    element.addEventListener("wheel", onWheel, { passive: false });
    return () => element.removeEventListener("wheel", onWheel);
  }, []);

  function zoom(factor: number) {
    setView((v) => {
      const k = clamp(v.k * factor);
      const cx = size.w / 2;
      const cy = size.h / 2;
      return {
        k,
        x: cx - ((cx - v.x) * k) / v.k,
        y: cy - ((cy - v.y) * k) / v.k,
      };
    });
  }

  function onPointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    if (event.button !== 0) return;
    drag.current = {
      x: event.clientX,
      y: event.clientY,
      vx: view.x,
      vy: view.y,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function onPointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    const start = drag.current;
    if (!start) return;
    setView((v) => ({
      ...v,
      x: start.vx + event.clientX - start.x,
      y: start.vy + event.clientY - start.y,
    }));
  }

  function endDrag() {
    drag.current = null;
  }

  if (!snapshot || !layout) {
    return (
      <div className="kg-panel">
        {plan ? (
          <>
            <p className="kg-note">{words.noSnapshot}</p>
            <Summary plan={plan} snapshotNodes={null} words={words} />
            <GroupList plan={plan} words={words} />
          </>
        ) : (
          <p className="kg-note">{words.noPlan}</p>
        )}
      </div>
    );
  }

  const nodesById = new Map<number, GraphNode>(
    snapshot.nodes.map((n) => [n.id, n]),
  );
  const activeNode = active === null ? undefined : nodesById.get(active);
  const activePlace = active === null ? undefined : layout.byId.get(active);
  const tooltipLeft = activePlace
    ? Math.min(
        view.x + (activePlace.x + activePlace.width) * view.k + 10,
        Math.max(0, size.w - 236),
      )
    : 0;
  const tooltipTop = activePlace ? view.y + activePlace.y * view.k : 0;

  return (
    <figure className="kg-panel" aria-label={label}>
      <div className="kg-summary" role="status">
        <span>
          <b>{words.tier}</b> {plan?.report.tier ?? words.none}
        </span>
        <span>
          <b>{words.dispatches}</b> {plan?.report.dispatches ?? 0}
        </span>
        <span>
          <b>{words.submissions}</b> {plan?.report.submissions ?? 0}
        </span>
        <span>
          <b>{words.calls}</b> {plan?.nodes.length ?? 0}
        </span>
        <span>
          <b>{words.nodes}</b> {snapshot.nodes.length}
        </span>
      </div>
      {!mapping.aligned && <p className="kg-note kg-warn">{words.unaligned}</p>}
      <div className="kg-toolbar">
        <span className="kg-hint">{words.rows}</span>
        <div className="kg-buttons">
          <button
            type="button"
            onClick={() => zoom(1.25)}
            aria-label={words.zoomIn}
          >
            +
          </button>
          <button
            type="button"
            onClick={() => zoom(1 / 1.25)}
            aria-label={words.zoomOut}
          >
            −
          </button>
          <button type="button" onClick={fit}>
            {words.fit}
          </button>
        </div>
      </div>
      <div
        ref={viewport}
        className="kg-viewport"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        style={{ height: compact ? 210 : 320 }}
      >
        <svg
          className="kg-svg"
          width={size.w}
          height={size.h}
          role="group"
          aria-label={label}
        >
          <g transform={`translate(${view.x} ${view.y}) scale(${view.k})`}>
            {layout.edges.map((edge) => {
              const from = nodesById.get(edge.from);
              const cls = edgeClass(mapping.byNode.get(edge.to)?.group, from);
              return (
                <g key={`${edge.from}-${edge.to}`} className={cls}>
                  <polyline
                    points={edge.points.map((p) => `${p.x},${p.y}`).join(" ")}
                    fill="none"
                    strokeWidth={edge.count > 1 ? 2.5 : 1.5}
                  />
                </g>
              );
            })}
            {snapshot.nodes.map((node) => {
              const place = layout.byId.get(node.id);
              if (!place) return null;
              const group = mapping.byNode.get(node.id);
              const kernel = callNames.get(node.id);
              const cls = nodeClass(node, group?.group);
              return (
                <g
                  key={node.id}
                  className={`kg-node ${cls}`}
                  transform={`translate(${place.x} ${place.y})`}
                  tabIndex={0}
                  role="img"
                  aria-label={nodeTitle(node, kernel, group?.group, words)}
                  aria-describedby={
                    active === node.id ? "kg-tooltip" : undefined
                  }
                  onPointerEnter={() => setActive(node.id)}
                  onPointerLeave={() =>
                    setActive((a) => (a === node.id ? null : a))
                  }
                  onFocus={() => setActive(node.id)}
                  onBlur={() => setActive((a) => (a === node.id ? null : a))}
                >
                  <rect width={place.width} height={place.height} rx={6} />
                  {group !== undefined && (
                    <>
                      <rect
                        className="kg-band"
                        width={place.width}
                        height={6}
                        rx={3}
                      />
                      <g
                        className="kg-badge"
                        transform={`translate(${place.width - 12} ${place.height - 12})`}
                      >
                        <circle r={9} />
                        <text textAnchor="middle" dominantBaseline="central">
                          {group.group + 1}
                        </text>
                      </g>
                    </>
                  )}
                  <text className="kg-title" x={10} y={22}>
                    {clip(
                      node.kind === "input"
                        ? words.input
                        : kernel && kernel !== node.op
                          ? `${node.op} · ${kernel}`
                          : node.op,
                      compact ? 16 : 22,
                    )}
                  </text>
                  <text className="kg-shape" x={10} y={38}>
                    {shapeText(node.shape)}
                  </text>
                  {node.parameter && (
                    <text
                      className="kg-tag"
                      x={place.width - 10}
                      y={14}
                      textAnchor="end"
                    >
                      {words.param}
                    </text>
                  )}
                </g>
              );
            })}
          </g>
        </svg>
        {activeNode && activePlace && (
          <div
            id="kg-tooltip"
            role="tooltip"
            className="kg-tooltip"
            style={{ left: tooltipLeft, top: tooltipTop }}
          >
            <TooltipBody
              node={activeNode}
              nodesById={nodesById}
              mapping={mapping}
              kernel={callNames.get(activeNode.id)}
              plan={plan}
              words={words}
            />
          </div>
        )}
      </div>
      <Legend mapping={mapping} words={words} />
      <p className="kg-note">{words.legend}</p>
    </figure>
  );
}

function Summary({
  plan,
  snapshotNodes,
  words,
}: {
  plan: DevicePlan;
  snapshotNodes: number | null;
  words: Words;
}) {
  return (
    <div className="kg-summary">
      <span>
        <b>{words.tier}</b> {plan.report.tier}
      </span>
      <span>
        <b>{words.dispatches}</b> {plan.report.dispatches}
      </span>
      <span>
        <b>{words.submissions}</b> {plan.report.submissions}
      </span>
      <span>
        <b>{words.calls}</b> {plan.nodes.length}
      </span>
      {snapshotNodes !== null && (
        <span>
          <b>{words.nodes}</b> {snapshotNodes}
        </span>
      )}
    </div>
  );
}

function GroupList({ plan, words }: { plan: DevicePlan; words: Words }) {
  return (
    <ol className="kg-groups">
      {plan.report.groups.map((group, index) => (
        <li key={index} className={`kg-group-item kg-g${index % GROUP_COUNT}`}>
          <span className="kg-swatch" aria-hidden="true">
            {index + 1}
          </span>
          <span>
            {words.group} {index + 1} ·{" "}
            {group.fused ? words.fused : words.notFused} ·{" "}
            {words.callCount(group.calls.length)}: {group.calls.join(", ")}
            {group.why && (
              <>
                <br />
                <small>
                  {words.why}: {group.why}
                </small>
              </>
            )}
          </span>
        </li>
      ))}
    </ol>
  );
}

function Legend({
  mapping,
  words,
}: {
  mapping: ReturnType<typeof groupNodes>;
  words: Words;
}) {
  if (mapping.groups.length === 0) return null;
  return (
    <ol className="kg-legend" aria-label={words.group}>
      {mapping.groups.map((group) => (
        <li
          key={group.index}
          className={`kg-group-item kg-g${group.index % GROUP_COUNT}`}
        >
          <span className="kg-swatch" aria-hidden="true">
            {group.index + 1}
          </span>
          <span>
            {words.group} {group.index + 1} ·{" "}
            {group.fused ? words.fused : words.notFused} ·{" "}
            {words.callCount(group.calls.length)}
          </span>
        </li>
      ))}
    </ol>
  );
}

function TooltipBody({
  node,
  nodesById,
  mapping,
  kernel,
  plan,
  words,
}: {
  node: GraphNode;
  nodesById: ReadonlyMap<number, GraphNode>;
  mapping: ReturnType<typeof groupNodes>;
  kernel: string | undefined;
  plan: DevicePlan | undefined;
  words: Words;
}) {
  const group = mapping.byNode.get(node.id);
  const inputs = node.inputs.map((id) => {
    const input = nodesById.get(id);
    return input
      ? `#${id} ${input.kind === "input" ? words.input : input.op}`
      : `#${id}`;
  });
  return (
    <>
      <strong>
        {node.kind === "input" ? words.input : node.op}
        {kernel && kernel !== node.op ? ` · ${kernel}` : ""}
      </strong>
      <span>
        {shapeText(node.shape)}
        {node.parameter ? ` · ${words.param}` : ""}
      </span>
      <span>
        {words.inputs}: {inputs.length ? inputs.join(", ") : words.none}
      </span>
      {group ? (
        <span>
          {words.group} {group.group + 1} ·{" "}
          {group.fused ? words.fused : words.notFused}
          {group.calls > 1 ? ` · ${words.callCount(group.calls)}` : ""}
        </span>
      ) : node.kind === "input" ? null : (
        <span>{node.computed ? words.computed : words.notRun}</span>
      )}
      {group?.why && plan && (
        <span>
          {words.why}: {group.why}
        </span>
      )}
    </>
  );
}

function nodeClass(node: GraphNode, group: number | undefined): string {
  if (node.kind === "input") return "kg-input";
  if (group === undefined) return "kg-idle";
  return `kg-op kg-g${group % GROUP_COUNT}`;
}

function edgeClass(
  group: number | undefined,
  from: GraphNode | undefined,
): string {
  if (group === undefined || from?.kind === "input") return "kg-edge";
  return `kg-edge kg-g${group % GROUP_COUNT}`;
}

function nodeTitle(
  node: GraphNode,
  kernel: string | undefined,
  group: number | undefined,
  words: Words,
): string {
  const what = node.kind === "input" ? words.input : node.op;
  const where =
    group === undefined
      ? node.kind === "input"
        ? ""
        : `, ${node.computed ? words.computed : words.notRun}`
      : `, ${words.group} ${group + 1}`;
  return `${what}${kernel && kernel !== node.op ? ` (${kernel})` : ""}, ${shapeText(node.shape)}${where}`;
}

function shapeText(shape: readonly number[]): string {
  return shape.length === 0 ? "[]" : `[${shape.join(" × ")}]`;
}

function clip(value: string, max: number): string {
  return value.length > max ? `${value.slice(0, max - 1)}…` : value;
}

function clamp(k: number): number {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, k));
}
