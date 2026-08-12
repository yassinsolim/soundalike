import assert from "node:assert/strict";
import { createHash, webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const html = readFileSync(new URL("../evaluate/index.html", import.meta.url), "utf8");
const script = html.match(/<script>\s*([\s\S]*?)\s*<\/script>/)?.[1];
const protocol = JSON.parse(
  readFileSync(new URL("../evaluate/protocol-v4.json", import.meta.url), "utf8"),
);
const pack = JSON.parse(
  readFileSync(new URL("../evaluate/active-pack.json", import.meta.url), "utf8"),
);

function context() {
  const elements = new Map();
  const storage = new Map();
  const element = () => ({
    checked: false,
    disabled: false,
    value: "",
    classList: { add() {}, remove() {}, toggle() {} },
    addEventListener() {},
  });
  const document = {
    createElement() {
      let text = "";
      return {
        click() {},
        set textContent(value) {
          text = String(value);
        },
        get innerHTML() {
          return text
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;");
        },
      };
    },
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, element());
      return elements.get(id);
    },
    querySelector() {
      return null;
    },
    querySelectorAll() {
      return [];
    },
  };
  const localStorage = {
    getItem(key) {
      return storage.get(key) ?? null;
    },
    setItem(key, value) {
      storage.set(key, String(value));
    },
  };
  const sandbox = vm.createContext({
    Blob,
    Date,
    Map,
    Number,
    Promise,
    Set,
    String,
    TextEncoder,
    URL,
    console,
    crypto: webcrypto,
    document,
    fetch: async () => {
      throw new Error("network disabled");
    },
    globalThis: null,
    localStorage,
  });
  sandbox.globalThis = sandbox;
  vm.runInContext(script, sandbox);
  return sandbox;
}

function clone(sandbox, value) {
  sandbox.__json = JSON.stringify(value);
  return vm.runInContext("JSON.parse(__json)", sandbox);
}

test("validates only the exact frozen active pack and protocol", async () => {
  const sandbox = context();
  const trustedProtocol = clone(sandbox, protocol);
  const trustedPack = clone(sandbox, pack);
  assert.equal(await sandbox.__v4Test.docHash(trustedProtocol), protocol.content_sha256);
  assert.equal(await sandbox.__v4Test.docHash(trustedPack), pack.content_sha256);
  assert.deepEqual(
    ["method_bindings", "control", "challenger"].filter((key) =>
      sandbox.__v4Test.nestedKeys(trustedPack).has(key),
    ),
    [],
  );
  assert.equal(
    await sandbox.__v4Test.validateStudy(
      trustedProtocol,
      trustedPack,
    ),
    true,
  );
  const tampered = clone(sandbox, pack);
  tampered.tasks[0].candidates[0].track_id += 1;
  await assert.rejects(
    sandbox.__v4Test.validateStudy(clone(sandbox, protocol), tampered),
    /pack|Choice|Track/,
  );
});

test("contains 16 unique comparisons and two interleaved repeated anchors", () => {
  const sandbox = context();
  const signatures = pack.tasks.map((task) =>
    sandbox.__v4Test.taskSignature(clone(sandbox, task)),
  );
  assert.equal(new Set(signatures).size, 16);
  const repeated = signatures
    .map((signature, index) => ({ signature, index }))
    .filter(
      (row, index, rows) =>
        rows.findIndex((candidate) => candidate.signature === row.signature) !==
        index,
    );
  assert.deepEqual(
    repeated.map((row) => row.index + 1),
    [7, 14],
  );
});

test("builds and imports strict full-ranking partial exports", async () => {
  const sandbox = context();
  const state = sandbox.__v4Test.emptyState();
  const task = pack.tasks[0];
  const stamp = Date.now();
  state.started_at = new Date(stamp - 2000).toISOString();
  state.last_activity_at = new Date(stamp - 1000).toISOString();
  state.task_ratings[task.task_id] = clone(sandbox, {
    outcome: "rated",
    ranked_choice_ids: task.candidates.map((choice) => choice.choice_id),
    worst_primary_reason: "tempo_pacing",
    skip_reason: null,
    completed_at: state.last_activity_at,
    interaction_ms: 1000,
  });

  sandbox.__v4Test.setStudy(protocol, pack, state);
  assert.equal(sandbox.__v4Test.validState(state), true);
  const exported = await sandbox.__v4Test.buildExport();
  assert.equal(sandbox.__v4Test.validExport(exported), true);
  const imported = await sandbox.__v4Test.importExport(exported);
  assert.equal(sandbox.__v4Test.validState(imported), true);
  assert.deepEqual(
    Array.from(imported.task_ratings[task.task_id].ranked_choice_ids),
    task.candidates.map((choice) => choice.choice_id),
  );
});

test("keeps draft autosave from resetting rated-task interaction time", () => {
  const sandbox = context();
  const state = sandbox.__v4Test.emptyState();
  const task = pack.tasks[0];
  const draft = clone(sandbox, {
    ranking: task.candidates.map((choice) => choice.choice_id),
    reason: "tempo_pacing",
  });
  state.lastInteractionAt = Date.now() - 5000;
  sandbox.__v4Test.setStudy(protocol, pack, state);
  sandbox.__v4Test.saveDraft(clone(sandbox, task), draft);
  assert.equal(
    sandbox.__v4Test.completeRatedTask(clone(sandbox, task), draft),
    true,
  );
  assert.ok(state.task_ratings[task.task_id].interaction_ms >= 4000);
});

test("loads frozen assets from the canonical evaluate route", () => {
  assert.equal(html.includes('fetch("/evaluate/protocol-v4.json"'), true);
  assert.equal(html.includes('fetch("/evaluate/active-pack.json"'), true);
  assert.equal(
    html.match(/<meta http-equiv="Content-Security-Policy"[^>]+>/)?.[0]
      .includes("frame-ancestors"),
    false,
  );
});

test("hides attribution until a task is complete and preserves excerpt bounds", () => {
  const sandbox = context();
  const track = pack.tracks[String(pack.tasks[0].seed_track_id)];
  const hidden = sandbox.__v4Test.trackHtml(clone(sandbox, track), "Seed", false);
  const revealed = sandbox.__v4Test.trackHtml(
    clone(sandbox, track),
    "Seed",
    true,
  );
  assert.equal(hidden.includes(track.title), false);
  assert.equal(hidden.includes(track.artist), false);
  assert.equal(revealed.includes(track.title), true);
  assert.equal(
    hidden.includes(`data-start="${track.audio.excerpt.start_seconds}"`),
    true,
  );
});

test("public V4 assets contain no method mapping or private key", () => {
  const text = [html, JSON.stringify(protocol), JSON.stringify(pack)].join("\n");
  for (const marker of [
    "method_bindings",
    "blinding_key",
    '"control"',
    '"challenger"',
  ]) {
    assert.equal(text.includes(marker), false);
  }
  assert.equal(text.includes("textarea"), false);
});

test("pacing V3 archive is byte-identical and version routed", () => {
  const expected = {
    "index.html": "575bf10c941ddc82ff31c2f196cedc204f4d15802a53dba30d18bcc6a86cd184",
    "pacing-pack.json": "3745fa4fa2df78e4f7feda4ccec924fac221ce91b5bf9d2c3316658e7a4e7525",
    "protocol-pacing-v3.json":
      "ba1db6bc3ad447c5eb2d1e2959d280bf6789a141c04182e0cf35976e9192bf02",
  };
  for (const [name, digest] of Object.entries(expected)) {
    const bytes = readFileSync(
      new URL(`../evaluate-pacing-v3/${name}`, import.meta.url),
    );
    assert.equal(createHash("sha256").update(bytes).digest("hex"), digest);
  }
  const config = JSON.parse(
    readFileSync(new URL("../vercel.json", import.meta.url)),
  );
  const routes = Object.fromEntries(
    config.rewrites.map((item) => [item.source, item.destination]),
  );
  assert.equal(routes["/evaluate"], "/evaluate/index.html");
  assert.equal(
    routes["/evaluate-pacing-v3"],
    "/evaluate-pacing-v3/index.html",
  );
});
