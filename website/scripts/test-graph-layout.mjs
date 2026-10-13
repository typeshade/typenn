// Node check of the kernel graph layout and the plan mapping (src/lab/graph-layout.ts).
// Run with `npm run test:graph`.
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  groupNodes,
  layoutGraph,
  mergeSnapshots,
  sampleIndices,
} from "../src/lab/graph-layout.ts";

const options = {
  nodeWidth: 100,
  nodeHeight: 40,
  gapX: 30,
  gapY: 10,
  padding: 0,
};

test("a diamond is layered by longest path and has four edges", () => {
  const nodes = [
    { id: 0, inputs: [] },
    { id: 1, inputs: [0] },
    { id: 2, inputs: [0] },
    { id: 3, inputs: [1, 2] },
  ];
  const layout = layoutGraph(nodes, options);
  assert.equal(layout.layers, 3);
  assert.equal(layout.byId.get(0).layer, 0);
  assert.equal(layout.byId.get(1).layer, 1);
  assert.equal(layout.byId.get(2).layer, 1);
  assert.equal(layout.byId.get(3).layer, 2);
  assert.notEqual(layout.byId.get(1).row, layout.byId.get(2).row);
  assert.equal(layout.edges.length, 4);
  for (const edge of layout.edges) {
    assert.equal(edge.points.length, 4);
    assert.equal(
      edge.points[0].x,
      layout.byId.get(edge.from).x + options.nodeWidth,
    );
    assert.equal(edge.points[3].x, layout.byId.get(edge.to).x);
  }
  assert.equal(layout.width, 3 * 100 + 2 * 30);
});

test("a chain is one row and one layer per node", () => {
  const nodes = [
    { id: 0, inputs: [] },
    { id: 1, inputs: [0] },
    { id: 2, inputs: [1] },
    { id: 3, inputs: [2] },
  ];
  const layout = layoutGraph(nodes, options);
  assert.equal(layout.layers, 4);
  for (const node of layout.nodes) assert.equal(node.row, 0);
  assert.equal(layout.height, 40);
});

test("a.add(a) draws one edge with count 2", () => {
  const nodes = [
    { id: 0, inputs: [] },
    { id: 1, inputs: [0, 0] },
  ];
  const layout = layoutGraph(nodes, options);
  assert.equal(layout.edges.length, 1);
  assert.equal(layout.edges[0].count, 2);
});

test("ordering by barycenter removes a crossing", () => {
  const nodes = [
    { id: 0, inputs: [] },
    { id: 1, inputs: [] },
    { id: 2, inputs: [1] },
    { id: 3, inputs: [0] },
  ];
  const layout = layoutGraph(nodes, options);
  assert.ok(layout.byId.get(3).row < layout.byId.get(2).row);
});

test("an empty graph has no size", () => {
  const layout = layoutGraph([], options);
  assert.equal(layout.width, 0);
  assert.equal(layout.height, 0);
  assert.equal(layout.nodes.length, 0);
});

const snapshot = {
  nodes: [
    { id: 0, kind: "input" },
    { id: 1, kind: "input" },
    { id: 2, kind: "op" },
    { id: 3, kind: "op" },
  ],
};

test("a fused ReLU(A x B) is one group of two calls", () => {
  const plan = {
    report: {
      tier: "webgpu",
      dispatches: 1,
      submissions: 1,
      groups: [{ calls: ["matmul", "relu"], fused: true }],
    },
    nodes: [2, 3],
  };
  const mapping = groupNodes(snapshot, plan);
  assert.equal(mapping.aligned, true);
  assert.equal(mapping.groups.length, 1);
  assert.deepEqual(mapping.byNode.get(2), { group: 0, fused: true, calls: 1 });
  assert.deepEqual(mapping.byNode.get(3), { group: 0, fused: true, calls: 1 });
  assert.equal(mapping.byNode.has(0), false);
});

test("the CPU plan of ReLU(A x B) is two groups", () => {
  const plan = {
    report: {
      tier: "cpu",
      dispatches: 0,
      submissions: 0,
      groups: [
        {
          calls: ["matmul"],
          fused: false,
          why: "the CPU tier runs each call apart",
        },
        {
          calls: ["relu"],
          fused: false,
          why: "the CPU tier runs each call apart",
        },
      ],
    },
    nodes: [2, 3],
  };
  const mapping = groupNodes(snapshot, plan);
  assert.equal(mapping.byNode.get(2).group, 0);
  assert.equal(mapping.byNode.get(3).group, 1);
  assert.equal(mapping.byNode.get(3).fused, false);
  assert.equal(mapping.byNode.get(3).why, "the CPU tier runs each call apart");
});

test("a scratch copy takes the id of the op that reads it", () => {
  const plan = {
    report: {
      tier: "cpu",
      dispatches: 0,
      submissions: 0,
      groups: [{ calls: ["copy", "add"], fused: false }],
    },
    nodes: [3, 3],
  };
  const mapping = groupNodes(
    {
      nodes: [
        { id: 0, kind: "input" },
        { id: 3, kind: "op" },
      ],
    },
    plan,
  );
  assert.equal(mapping.aligned, true);
  assert.equal(mapping.byNode.get(3).calls, 2);
});

test("a plan whose call count differs from its node list is not aligned", () => {
  const plan = {
    report: {
      tier: "cpu",
      dispatches: 0,
      submissions: 0,
      groups: [{ calls: ["add", "mul"], fused: false }],
    },
    nodes: [3],
  };
  assert.equal(groupNodes(snapshot, plan).aligned, false);
});

test("no plan maps no node", () => {
  const mapping = groupNodes(snapshot, undefined);
  assert.equal(mapping.byNode.size, 0);
  assert.deepEqual(mapping.groups, []);
});

test("merged snapshots list a shared tensor once and keep inputs first", () => {
  const first = {
    output: 4,
    nodes: [{ id: 0 }, { id: 1 }, { id: 4 }],
  };
  const second = {
    output: 5,
    nodes: [{ id: 0 }, { id: 1 }, { id: 5 }],
  };
  const merged = mergeSnapshots([first, second]);
  assert.deepEqual(
    merged.nodes.map((n) => n.id),
    [0, 1, 4, 5],
  );
  assert.equal(merged.output, 5);
  assert.deepEqual(mergeSnapshots([]), { output: -1, nodes: [] });
});

test("sampled indices are different, in range, and the same for one seed", () => {
  let state = 7;
  const random = () => {
    state = (state * 1103515245 + 12345) % 2147483648;
    return state / 2147483648;
  };
  const a = sampleIndices(50, 10, random);
  assert.equal(a.length, 10);
  assert.equal(new Set(a).size, 10);
  for (const i of a) assert.ok(i >= 0 && i < 50);
  state = 7;
  assert.deepEqual([...sampleIndices(50, 10, random)], [...a]);
  assert.equal(sampleIndices(5, 10, random).length, 5);
});
