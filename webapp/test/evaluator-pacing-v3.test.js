import assert from "node:assert/strict";
import { createHash, webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const html = readFileSync(new URL("../evaluate/index.html", import.meta.url), "utf8");
const script = html.match(/<script>\s*([\s\S]*?)\s*<\/script>/)?.[1];
const protocol = JSON.parse(
  readFileSync(new URL("../evaluate/protocol-pacing-v3.json", import.meta.url), "utf8"),
);
const pack = JSON.parse(
  readFileSync(new URL("../evaluate/pacing-pack.json", import.meta.url), "utf8"),
);

function context() {
  const elements = new Map();
  const values = new Map();
  const element = () => ({
    checked: false,
    disabled: false,
    classList: { add() {}, remove() {}, contains() { return false; } },
  });
  const document = {
    addEventListener() {},
    createElement() {
      let text = "";
      return {
        click() {},
        set textContent(value) { text = String(value); },
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
    querySelector() { return null; },
    querySelectorAll() { return []; },
  };
  const localStorage = {
    getItem(key) { return values.has(key) ? values.get(key) : null; },
    setItem(key, value) { values.set(key, String(value)); },
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
    fetch: async () => { throw new Error("network must not run"); },
    localStorage,
    location: {
      hash: "",
      hostname: "localhost",
      href: "http://localhost:8788/evaluate/",
      origin: "http://localhost:8788",
      pathname: "/evaluate/",
      protocol: "http:",
      search: "",
    },
  });
  vm.runInContext(script, sandbox);
  return sandbox;
}

function realmClone(sandbox, value) {
  sandbox.__cloneJson = JSON.stringify(value);
  return vm.runInContext("JSON.parse(__cloneJson)", sandbox);
}

test("validates only the exact locked blinded documents", async () => {
  const sandbox = context();
  const trustedProtocol = realmClone(sandbox, protocol);
  const trustedPack = realmClone(sandbox, pack);
  assert.equal(
    await sandbox.__pacingV3Test.validateStudy(trustedProtocol, trustedPack),
    true,
  );
  const tampered = realmClone(sandbox, pack);
  tampered.seeds[0].lists[0].ranking[0].track_id += 1;
  await assert.rejects(
    sandbox.__pacingV3Test.validateStudy(trustedProtocol, tampered),
    /pack|Study/,
  );
});

test("randomizes anonymous list labels deterministically per seed and session", async () => {
  const sandbox = context();
  const state = sandbox.__pacingV3Test.emptyState();
  sandbox.__pacingV3Test.setStudy(protocol, pack, state);
  const lists = pack.seeds[0].lists;
  const first = await sandbox.__pacingV3Test.sessionOrder(
    lists,
    `list-order:${pack.seeds[0].seed_id}`,
    `pacing-session-${"1".repeat(24)}`,
  );
  const again = await sandbox.__pacingV3Test.sessionOrder(
    lists,
    `list-order:${pack.seeds[0].seed_id}`,
    `pacing-session-${"1".repeat(24)}`,
  );
  assert.deepEqual(first.map((item) => item.list_id), again.map((item) => item.list_id));
  const observed = new Set();
  for (let index = 1; index <= 8; index += 1) {
    const order = await sandbox.__pacingV3Test.sessionOrder(
      lists,
      `list-order:${pack.seeds[0].seed_id}`,
      `pacing-session-${String(index).repeat(24)}`,
    );
    observed.add(order.map((item) => item.list_id).join(","));
  }
  assert.equal(observed.size, 2);
});

test("supports strict partial list and result exports with synchronized result IDs", async () => {
  const sandbox = context();
  const state = sandbox.__pacingV3Test.emptyState();
  const time = Date.now();
  state.started_at = new Date(time - 2_000).toISOString();
  state.last_activity_at = new Date(time - 1_000).toISOString();
  const seed = pack.seeds.find((item) => item.matched_list_overlap > 0);
  const list = seed.lists[0];
  const resultId = list.ranking[0].result_id;
  state.list_ratings[list.list_id] = realmClone(sandbox, {
    score_0_10: 8,
    rated_at: state.last_activity_at,
    interaction_ms: 500,
  });
  state.result_ratings[resultId] = realmClone(sandbox, {
    score_0_10: 7,
    mismatch_reasons: ["tempo_pacing", "tone_timbre"],
    rated_at: state.last_activity_at,
    interaction_ms: 500,
  });
  sandbox.__pacingV3Test.setStudy(protocol, pack, state);
  assert.equal(sandbox.__pacingV3Test.validState(state), true);
  const exported = await sandbox.__pacingV3Test.buildExport();
  assert.equal(sandbox.__pacingV3Test.validExport(exported), true);
  assert.equal(Object.keys(exported.list_ratings).length, 1);
  assert.equal(Object.keys(exported.result_ratings).length, 1);
  assert.equal(exported.result_ratings[resultId].score_0_10, 7);
  const imported = await sandbox.__pacingV3Test.stateFromExport(exported);
  assert.equal(sandbox.__pacingV3Test.validState(imported), true);
});

test("does not release attribution until a list and all five results are complete", () => {
  const sandbox = context();
  const state = sandbox.__pacingV3Test.emptyState();
  const seed = pack.seeds.find((item) => item.matched_list_overlap > 0);
  const list = seed.lists[0];
  const stamp = state.started_at;
  sandbox.__pacingV3Test.setStudy(protocol, pack, state);
  state.list_ratings[list.list_id] = realmClone(sandbox, {
    score_0_10: 8,
    rated_at: stamp,
    interaction_ms: 1,
  });
  assert.equal(sandbox.__pacingV3Test.fullListComplete(list), false);
  for (const row of list.ranking) {
    state.result_ratings[row.result_id] = realmClone(sandbox, {
      score_0_10: 8,
      mismatch_reasons: [],
      rated_at: stamp,
      interaction_ms: 1,
    });
  }
  assert.equal(sandbox.__pacingV3Test.fullListComplete(list), true);
});

test("keeps identities out of public assets and preserves excerpt boundaries", () => {
  const publicText = [html, JSON.stringify(protocol), JSON.stringify(pack)].join("\n");
  for (const marker of [
    ["fulltrack", "audio", "study", "v2"].join("_"),
    ["pacing", "tone", "study", "v3"].join("_"),
    "method_bindings",
    "blinding_key_hex",
  ]) {
    assert.equal(publicText.includes(marker), false);
  }
  assert.equal(publicText.includes("textarea"), false);
  const sandbox = context();
  const state = sandbox.__pacingV3Test.emptyState();
  sandbox.__pacingV3Test.setStudy(protocol, pack, state);
  const track = pack.tracks[String(pack.seeds[0].seed_track_id)];
  const player = sandbox.__pacingV3Test.trackPlayerHtml(track, "Seed track");
  assert.equal(player.includes(track.title), false);
  assert.equal(player.includes(track.artist), false);
  assert.equal(player.includes(`data-excerpt-start="${track.playback_excerpt.start_seconds}"`), true);
});

test("rejects invalid numeric input without mutating a valid score", () => {
  const sandbox = context();
  const invalid = { value: "7.5" };
  assert.equal(sandbox.__pacingV3Test.inputScore(invalid, 6), undefined);
  assert.equal(invalid.value, "6");
  const outOfRange = { value: "11" };
  assert.equal(sandbox.__pacingV3Test.inputScore(outOfRange, null), undefined);
  assert.equal(outOfRange.value, "");
  assert.equal(sandbox.__pacingV3Test.inputScore({ value: "0" }, null), 0);
  assert.equal(sandbox.__pacingV3Test.inputScore({ value: "10" }, null), 10);
});

test("archives semantic v2 byte-for-byte and exposes all versioned routes", () => {
  const hashes = {
    "index.html": "cd5f5c553eef7264a75a4bd80ff2987e39f8a1363a4bf7c7dccd8c7056960e85",
    "pilot-pack.json": "d23d66768f15fd5e37e01ad2a8905d181b4ff278c85674386edcd7dc50b267d3",
    "protocol-semantic-v2.json": "36919e57883fb54028c98e495431638edaecd899d533a0523026d3ce81fdaa20",
    "protocol-v2.json": "a88108894e3875159a9ae5b3fae61b01522e9c22647d9ff32748d53d0a5c981c",
    "semantic-pack.json": "f07bf814eab2a363aa9fbec5acd946e57cfad3d3c3eef6dea4027a190d0e13b3",
  };
  for (const [name, expected] of Object.entries(hashes)) {
    const bytes = readFileSync(
      new URL(`../evaluate-semantic-v2/${name}`, import.meta.url),
    );
    assert.equal(createHash("sha256").update(bytes).digest("hex"), expected);
  }
  const config = JSON.parse(readFileSync(new URL("../vercel.json", import.meta.url)));
  const routes = Object.fromEntries(
    config.rewrites.map((item) => [item.source, item.destination]),
  );
  assert.equal(routes["/evaluate"], "/evaluate/index.html");
  assert.equal(routes["/evaluate-semantic-v2"], "/evaluate-semantic-v2/index.html");
  assert.equal(routes["/evaluate-semantic-v1"], "/evaluate-semantic-v1/index.html");
  assert.equal(routes["/evaluate-v2"], "/evaluate-v2/index.html");
  assert.equal(routes["/evaluate-v1"], "/evaluate-v1/index.html");
});
