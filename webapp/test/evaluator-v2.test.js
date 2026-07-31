import assert from "node:assert/strict";
import { createHash, webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const html = readFileSync(
  new URL("../evaluate/index.html", import.meta.url),
  "utf8",
);
const v1Html = readFileSync(
  new URL("../evaluate-v1/index.html", import.meta.url),
  "utf8",
);
const script = html.match(/<script>\s*([\s\S]*?)\s*<\/script>/)?.[1];
const protocol = JSON.parse(
  readFileSync(new URL("../evaluate/protocol-v2.json", import.meta.url), "utf8"),
);
const pack = JSON.parse(
  readFileSync(new URL("../evaluate/pilot-pack.json", import.meta.url), "utf8"),
);

function context() {
  const elements = new Map();
  const values = new Map();
  const element = () => ({
    checked: false,
    disabled: false,
    classList: {
      add() {},
      remove() {},
      contains() {
        return false;
      },
    },
  });
  const document = {
    addEventListener() {},
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
      return values.has(key) ? values.get(key) : null;
    },
    setItem(key, value) {
      values.set(key, String(value));
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
      throw new Error("network must not run");
    },
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

test("validates the exact locked public v2 documents and rejects tampering", async () => {
  const sandbox = context();
  const trustedProtocol = realmClone(sandbox, protocol);
  const trustedPack = realmClone(sandbox, pack);
  assert.equal(
    await sandbox.__fulltrackV2Test.validateStudy(trustedProtocol, trustedPack),
    true,
  );

  const wrongProtocol = realmClone(sandbox, protocol);
  wrongProtocol.schema_version = 17;
  await assert.rejects(
    sandbox.__fulltrackV2Test.validateStudy(wrongProtocol, trustedPack),
    /Protocol/,
  );

  const tamperedPack = realmClone(sandbox, pack);
  tamperedPack.seeds[0].lists[0].ranking[0].track_id += 1;
  await assert.rejects(
    sandbox.__fulltrackV2Test.validateStudy(trustedProtocol, tamperedPack),
    /pack/,
  );
});

test("uses stable opaque list orders that vary by session", async () => {
  const sandbox = context();
  const state = sandbox.__fulltrackV2Test.emptyState();
  sandbox.__fulltrackV2Test.setStudy(protocol, pack, state);
  const lists = pack.seeds[0].lists;
  const first = await sandbox.__fulltrackV2Test.sessionOrder(
    lists,
    "list-order:test",
    `fulltrack-session-${"1".repeat(24)}`,
  );
  const again = await sandbox.__fulltrackV2Test.sessionOrder(
    lists,
    "list-order:test",
    `fulltrack-session-${"1".repeat(24)}`,
  );
  const second = await sandbox.__fulltrackV2Test.sessionOrder(
    lists,
    "list-order:test",
    `fulltrack-session-${"2".repeat(24)}`,
  );
  assert.deepEqual(
    first.map((item) => item.list_id),
    again.map((item) => item.list_id),
  );
  assert.notDeepEqual(
    first.map((item) => item.list_id),
    second.map((item) => item.list_id),
  );
});

test("autosaves, exports and imports only strict signed v2 state", async () => {
  const sandbox = context();
  const state = sandbox.__fulltrackV2Test.emptyState();
  const time = Date.now();
  state.started_at = new Date(time - 1000).toISOString();
  const ratedAt = new Date(time).toISOString();
  state.last_activity_at = ratedAt;
  sandbox.__ratingJson = JSON.stringify({
    similarity: "somewhat_similar",
    score_0_10: null,
    unrelated_positions_1_to_5: [],
    rated_at: ratedAt,
    interaction_ms: 1,
  });
  state.list_ratings[pack.seeds[0].lists[0].list_id] = vm.runInContext(
    "JSON.parse(__ratingJson)",
    sandbox,
  );
  sandbox.__fulltrackV2Test.setStudy(protocol, pack, state);
  assert.equal(sandbox.__fulltrackV2Test.validState(state), true);
  sandbox.__fulltrackV2Test.save();
  assert.equal(sandbox.__fulltrackV2Test.restoreAutosave().session_id, state.session_id);

  const exported = await sandbox.__fulltrackV2Test.buildExport();
  assert.equal(sandbox.__fulltrackV2Test.validExport(exported), true);
  const imported = await sandbox.__fulltrackV2Test.stateFromExport(exported);
  assert.equal(sandbox.__fulltrackV2Test.validState(imported), true);
  assert.equal(imported.session_id, state.session_id);

  exported.list_ratings[pack.seeds[0].lists[0].list_id].similarity =
    "not_similar";
  await assert.rejects(
    sandbox.__fulltrackV2Test.stateFromExport(exported),
    /HMAC/,
  );
});

test("keeps attribution out of anonymous player markup until rating", () => {
  const sandbox = context();
  const state = sandbox.__fulltrackV2Test.emptyState();
  sandbox.__fulltrackV2Test.setStudy(protocol, pack, state);
  const track = pack.tracks[pack.seeds[0].seed_track_id];
  const player = sandbox.__fulltrackV2Test.trackPlayerHtml(track, "Seed track");
  assert.equal(player.includes(track.title), false);
  assert.equal(player.includes(track.artist), false);
  assert.equal(player.includes(track.license.attribution), false);
  assert.equal(player.includes(`trackid=${track.track_id}&amp;format=mp31`), true);
  assert.equal(player.includes('preload="none"'), true);

  const attribution = sandbox.__fulltrackV2Test.trackAttributionHtml(track);
  assert.equal(attribution.includes(track.title), true);
  assert.equal(attribution.includes(track.artist), true);
  assert.equal(attribution.includes(track.license.url), true);
});

test("preserves byte-locked v17 assets and isolates routes and state", () => {
  const hashes = {
    "index.html": "b6445a1400e0b92a7187e895ec22e8301e53abcc73f9974ceb13436fecc9f537",
    "protocol.json": "02fb2baa60d3a7bc2ae67f198ea470f5cd1837ff6c9704526f4c41b3281975a1",
    "served-lists.json":
      "1253cfd0501f320bf6cda4d451509d7b2fa552a1ecbe5636a9e3477137850f20",
  };
  for (const [name, expected] of Object.entries(hashes)) {
    const bytes = readFileSync(new URL(`../evaluate-v1/${name}`, import.meta.url));
    assert.equal(createHash("sha256").update(bytes).digest("hex"), expected);
  }
  assert.equal(v1Html.includes("soundalike-human-v17"), true);
  assert.equal(html.includes("soundalike-fulltrack-v2"), true);
  assert.equal(html.includes("soundalike-human-v17"), false);
  assert.equal(html.includes("/api/ratings-v2"), true);
  assert.equal(v1Html.includes('fetch("/api/ratings"'), true);

  const config = JSON.parse(
    readFileSync(new URL("../vercel.json", import.meta.url), "utf8"),
  );
  const routes = Object.fromEntries(
    config.rewrites.map((item) => [item.source, item.destination]),
  );
  assert.equal(routes["/evaluate"], "/evaluate/index.html");
  assert.equal(routes["/evaluate-v1"], "/evaluate-v1/index.html");
  assert.equal(config.functions["api/ratings.js"].maxDuration, 15);
  assert.equal(config.functions["api/ratings-v2.js"].maxDuration, 15);
});

test("public v2 assets contain no model identity or private unblinding", () => {
  const publicText = [
    html,
    JSON.stringify(protocol),
    JSON.stringify(pack),
  ].join("\n");
  for (const marker of [
    "nonnegative_linear",
    "monotonic_network",
    "channel_gated_embedding",
    "frozen_hybrid",
    "BEGIN PRIVATE KEY",
  ]) {
    assert.equal(publicText.includes(marker), false);
  }
  assert.equal(protocol.production_recommendation_changed, false);
  assert.equal(pack.promotion_allowed, false);
});
