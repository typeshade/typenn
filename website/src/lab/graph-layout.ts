// Layout and plan mapping for the kernel graph view. Framework-free and pure: the same code runs in
// the island and in the node check (website/scripts/test-graph-layout.mjs).

/** One node of a graph, as the layout needs it. `inputs` are graph ids, in order. */
export interface LayoutInput {
  readonly id: number;
  readonly inputs: readonly number[];
}

export interface LayoutOptions {
  readonly nodeWidth: number;
  readonly nodeHeight: number;
  /** Space between two layers (columns), left to right. */
  readonly gapX: number;
  /** Space between two nodes of one layer, top to bottom. */
  readonly gapY: number;
  /** Ordering passes. Each pass sweeps the layers forward, then backward. */
  readonly sweeps?: number;
  /** Margin around the drawing. */
  readonly padding?: number;
}

export interface LayoutNode {
  readonly id: number;
  /** Longest path from an input, so inputs are in layer 0. */
  readonly layer: number;
  /** Position of the node inside its layer, from the top. */
  readonly row: number;
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

export interface LayoutEdge {
  readonly from: number;
  readonly to: number;
  /** How many inputs of `to` are `from` (a.add(a) has 2). */
  readonly count: number;
  /** Orthogonal polyline from the right side of `from` to the left side of `to`. */
  readonly points: readonly { readonly x: number; readonly y: number }[];
}

export interface GraphLayout {
  readonly width: number;
  readonly height: number;
  readonly layers: number;
  readonly nodes: readonly LayoutNode[];
  readonly byId: ReadonlyMap<number, LayoutNode>;
  readonly edges: readonly LayoutEdge[];
}

/**
 * Lay out a DAG in layers from left to right. A node sits one layer after its deepest input
 * (longest path). Nodes in a layer are ordered by the barycenter of their neighbours in the
 * adjacent layer, which lowers the number of crossing edges. `nodes` must be in topological
 * order, as `nn.graph()` returns them.
 */
export function layoutGraph(
  nodes: readonly LayoutInput[],
  options: LayoutOptions,
): GraphLayout {
  const { nodeWidth: w, nodeHeight: h, gapX, gapY } = options;
  const sweeps = options.sweeps ?? 4;
  const padding = options.padding ?? 16;
  const known = new Set(nodes.map((n) => n.id));
  const inputsOf = new Map<number, number[]>();
  const users = new Map<number, number[]>();
  for (const n of nodes) {
    const inputs = n.inputs.filter((i) => known.has(i));
    inputsOf.set(n.id, inputs);
    for (const i of inputs) {
      const list = users.get(i) ?? [];
      list.push(n.id);
      users.set(i, list);
    }
  }

  const layerOf = new Map<number, number>();
  for (const n of nodes) {
    let layer = 0;
    for (const i of inputsOf.get(n.id)!) {
      const below = layerOf.get(i);
      if (below !== undefined) layer = Math.max(layer, below + 1);
    }
    layerOf.set(n.id, layer);
  }

  const layerCount = nodes.length === 0 ? 0 : Math.max(...layerOf.values()) + 1;
  const columns: number[][] = Array.from({ length: layerCount }, () => []);
  for (const n of nodes) columns[layerOf.get(n.id)!]!.push(n.id);

  const position = new Map<number, number>();
  const refresh = (column: readonly number[]) =>
    column.forEach((id, row) => position.set(id, row));
  columns.forEach(refresh);

  const reorder = (
    layer: number,
    neighbours: ReadonlyMap<number, readonly number[]>,
  ) => {
    const column = columns[layer]!;
    const keyed = column.map((id, row) => {
      const near = (neighbours.get(id) ?? []).filter(
        (other) => layerOf.get(other) !== layer,
      );
      const sum = near.reduce(
        (total, other) => total + position.get(other)!,
        0,
      );
      return { id, row, key: near.length === 0 ? row : sum / near.length };
    });
    keyed.sort((a, b) => a.key - b.key || a.row - b.row);
    columns[layer] = keyed.map((entry) => entry.id);
    refresh(columns[layer]!);
  };

  for (let pass = 0; pass < sweeps; pass++) {
    for (let layer = 1; layer < layerCount; layer++) reorder(layer, inputsOf);
    for (let layer = layerCount - 2; layer >= 0; layer--) reorder(layer, users);
  }

  const placed: LayoutNode[] = [];
  const byId = new Map<number, LayoutNode>();
  columns.forEach((column, layer) =>
    column.forEach((id, row) => {
      const node: LayoutNode = {
        id,
        layer,
        row,
        x: padding + layer * (w + gapX),
        y: padding + row * (h + gapY),
        width: w,
        height: h,
      };
      placed.push(node);
      byId.set(id, node);
    }),
  );

  const edges: LayoutEdge[] = [];
  for (const n of nodes) {
    const counts = new Map<number, number>();
    for (const i of inputsOf.get(n.id)!)
      counts.set(i, (counts.get(i) ?? 0) + 1);
    for (const [from, count] of counts) {
      const a = byId.get(from)!;
      const b = byId.get(n.id)!;
      const startX = a.x + w;
      const startY = a.y + h / 2;
      const endX = b.x;
      const endY = b.y + h / 2;
      const midX = endX > startX ? (startX + endX) / 2 : startX + gapX / 2;
      edges.push({
        from,
        to: n.id,
        count,
        points: [
          { x: startX, y: startY },
          { x: midX, y: startY },
          { x: midX, y: endY },
          { x: endX, y: endY },
        ],
      });
    }
  }

  const rows = Math.max(0, ...columns.map((column) => column.length));
  const width =
    layerCount === 0
      ? 0
      : padding * 2 + layerCount * w + (layerCount - 1) * gapX;
  const height = rows === 0 ? 0 : padding * 2 + rows * h + (rows - 1) * gapY;
  return { width, height, layers: layerCount, nodes: placed, byId, edges };
}

/** The part of `nn.graph()` that the mapping reads. */
export interface SnapshotLike {
  readonly nodes: readonly {
    readonly id: number;
    readonly kind: "input" | "op";
  }[];
}

/** The part of `device.lastPlan` that the mapping reads. */
export interface PlanLike {
  readonly report: {
    readonly groups: readonly {
      readonly calls: readonly string[];
      readonly fused: boolean;
      readonly why?: string;
    }[];
  };
  /** Graph id of each kernel call, in call order. */
  readonly nodes: readonly number[];
}

export interface GroupInfo {
  readonly index: number;
  readonly calls: readonly string[];
  readonly fused: boolean;
  readonly why?: string;
}

export interface NodeGroup {
  /** Index into `groups`. */
  readonly group: number;
  readonly fused: boolean;
  readonly why?: string;
  /** Kernel calls that ran this node, including a scratch copy of it. */
  readonly calls: number;
}

export interface PlanMapping {
  readonly groups: readonly GroupInfo[];
  /** Op nodes of the snapshot that the plan ran, by graph id. */
  readonly byNode: ReadonlyMap<number, NodeGroup>;
  /** False when the call count of the report and the node list differ. */
  readonly aligned: boolean;
}

/**
 * Map each op node of a snapshot to the group of the plan that ran it. The k-th kernel call is
 * `plan.nodes[k]`, and the calls of the groups, in order, line up with it. Nodes that the plan
 * did not run were already computed, or are inputs.
 */
export function groupNodes(
  snapshot: SnapshotLike,
  plan: PlanLike | undefined,
): PlanMapping {
  if (!plan) return { groups: [], byNode: new Map(), aligned: true };
  const ops = new Set(
    snapshot.nodes.filter((n) => n.kind === "op").map((n) => n.id),
  );
  const groups = plan.report.groups.map((group, index) => ({
    index,
    calls: group.calls,
    fused: group.fused,
    ...(group.why !== undefined ? { why: group.why } : {}),
  }));
  const totalCalls = groups.reduce((sum, group) => sum + group.calls.length, 0);
  const aligned = totalCalls === plan.nodes.length;
  const byNode = new Map<number, NodeGroup>();
  let call = 0;
  for (const group of groups) {
    for (let k = 0; k < group.calls.length; k++, call++) {
      const id = plan.nodes[call];
      if (id === undefined || !ops.has(id)) continue;
      const previous = byNode.get(id);
      byNode.set(id, {
        group: group.index,
        fused: group.fused,
        ...(group.why !== undefined ? { why: group.why } : {}),
        calls: (previous?.calls ?? 0) + 1,
      });
    }
  }
  return { groups, byNode, aligned };
}

/**
 * Join the snapshots of several tensors that one plan reads, such as the gradient of each
 * parameter. A tensor that two snapshots share keeps its first place. The input of a tensor still
 * comes before it, because each snapshot is in topological order. `output` is the one of the last
 * snapshot: a view that reads several tensors has no single output.
 */
export function mergeSnapshots<
  S extends {
    readonly output: number;
    readonly nodes: readonly { readonly id: number }[];
  },
>(
  snapshots: readonly S[],
): { readonly output: number; readonly nodes: S["nodes"][number][] } {
  const seen = new Set<number>();
  const nodes: S["nodes"][number][] = [];
  for (const snapshot of snapshots)
    for (const node of snapshot.nodes) {
      if (seen.has(node.id)) continue;
      seen.add(node.id);
      nodes.push(node);
    }
  const last = snapshots[snapshots.length - 1];
  return { output: last ? last.output : -1, nodes };
}

/**
 * `n` different indices of `0 ... count - 1`, chosen by a partial Fisher-Yates shuffle with the
 * random numbers of `random` (a function that returns a number in [0, 1)). A seeded `random`
 * gives the same indices on every run.
 */
export function sampleIndices(
  count: number,
  n: number,
  random: () => number,
): Uint32Array {
  const take = Math.min(Math.max(0, Math.floor(n)), count);
  const pool = new Uint32Array(count);
  for (let i = 0; i < count; i++) pool[i] = i;
  for (let i = 0; i < take; i++) {
    const j = i + Math.floor(random() * (count - i));
    const swap = pool[i]!;
    pool[i] = pool[j]!;
    pool[j] = swap;
  }
  return pool.slice(0, take);
}
