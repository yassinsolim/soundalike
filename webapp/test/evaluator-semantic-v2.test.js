import assert from "node:assert/strict";
import { createHash, webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const html = readFileSync(
  new URL("../evaluate-semantic-v2/index.html", import.meta.url),
  "utf8",
);
const v2Html = readFileSync(
  new URL("../evaluate-v2/index.html", import.meta.url),
  "utf8",
);
const semanticV1Html = readFileSync(
  new URL("../evaluate-semantic-v1/index.html", import.meta.url),
  "utf8",
);
const script = html.match(/<script>\s*([\s\S]*?)\s*<\/script>/)?.[1];
const protocol = JSON.parse(
  readFileSync(
    new URL("../evaluate-semantic-v2/protocol-semantic-v2.json", import.meta.url),
    "utf8",
  ),
);
const pack = JSON.parse(
  readFileSync(
    new URL("../evaluate-semantic-v2/semantic-pack.json", import.meta.url),
    "utf8",
  ),
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
      href: "http://localhost:8788/evaluate-semantic-v2/",
      origin: "http://localhost:8788",
      pathname: "/evaluate-semantic-v2/",
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

test("validates the exact locked semantic documents and rejects tampering", async () => {
  const sandbox = context();
  const trustedProtocol = realmClone(sandbox, protocol);
  const trustedPack = realmClone(sandbox, pack);
  assert.equal(
    await sandbox.__semanticV2Test.validateStudy(trustedProtocol, trustedPack),
    true,
  );

  const wrongProtocol = realmClone(sandbox, protocol);
  wrongProtocol.schema_version = 17;
  await assert.rejects(
    sandbox.__semanticV2Test.validateStudy(wrongProtocol, trustedPack),
    /Protocol/,
  );

  const tamperedPack = realmClone(sandbox, pack);
  tamperedPack.seeds[0].lists[0].ranking[0].track_id += 1;
  await assert.rejects(
    sandbox.__semanticV2Test.validateStudy(trustedProtocol, tamperedPack),
    /pack/,
  );
});

test("uses stable opaque list orders that vary by session", async () => {
  const sandbox = context();
  const state = sandbox.__semanticV2Test.emptyState();
  sandbox.__semanticV2Test.setStudy(protocol, pack, state);
  const lists = pack.seeds[0].lists;
  const first = await sandbox.__semanticV2Test.sessionOrder(
    lists,
    "list-order:test",
    `semantic-session-${"1".repeat(24)}`,
  );
  const again = await sandbox.__semanticV2Test.sessionOrder(
    lists,
    "list-order:test",
    `semantic-session-${"1".repeat(24)}`,
  );
  assert.deepEqual(
    first.map((item) => item.list_id),
    again.map((item) => item.list_id),
  );
  const observed = new Set();
  for (let value = 1; value <= 8; value += 1) {
    const order = await sandbox.__semanticV2Test.sessionOrder(
      lists,
      "list-order:test",
      `semantic-session-${String(value).repeat(24)}`,
    );
    observed.add(order.map((item) => item.list_id).join(","));
  }
  assert.equal(observed.size, 2);
  assert.deepEqual(
    pack.seeds.map((seed) => seed.priority_rank),
    Array.from({ length: 20 }, (_, index) => index + 1),
  );
  assert.deepEqual(
    pack.seeds.slice(0, 3).map((seed) => seed.scene),
    ["dance", "house", "hiphop"],
  );
  assert.equal(html.includes("orderedSeeds=[...pack.seeds]"), true);
  assert.equal(html.includes("sessionOrder(pack.seeds"), false);
});

test("autosaves, exports and imports only strict signed semantic state", async () => {
  const sandbox = context();
  const state = sandbox.__semanticV2Test.emptyState();
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
  sandbox.__semanticV2Test.setStudy(protocol, pack, state);
  assert.equal(sandbox.__semanticV2Test.validState(state), true);
  sandbox.__semanticV2Test.save();
  assert.equal(sandbox.__semanticV2Test.restoreAutosave().session_id, state.session_id);

  const exported = await sandbox.__semanticV2Test.buildExport();
  assert.equal(sandbox.__semanticV2Test.validExport(exported), true);
  const imported = await sandbox.__semanticV2Test.stateFromExport(exported);
  assert.equal(sandbox.__semanticV2Test.validState(imported), true);
  assert.equal(imported.session_id, state.session_id);

  exported.list_ratings[pack.seeds[0].lists[0].list_id].similarity =
    "not_similar";
  await assert.rejects(
    sandbox.__semanticV2Test.stateFromExport(exported),
    /HMAC/,
  );
});

test("keeps attribution out of anonymous player markup until rating", () => {
  const sandbox = context();
  const state = sandbox.__semanticV2Test.emptyState();
  sandbox.__semanticV2Test.setStudy(protocol, pack, state);
  const track = pack.tracks[pack.seeds[0].seed_track_id];
  const player = sandbox.__semanticV2Test.trackPlayerHtml(track, "Seed track");
  assert.equal(player.includes(track.title), false);
  assert.equal(player.includes(track.artist), false);
  assert.equal(player.includes(track.license.attribution), false);
  assert.equal(player.includes(`trackid=${track.track_id}&amp;format=mp31`), true);
  assert.equal(player.includes('preload="metadata"'), true);
  assert.equal(
    player.includes(`data-excerpt-start="${track.playback_excerpt.start_seconds}"`),
    true,
  );
  assert.equal(
    player.includes(`data-excerpt-end="${track.playback_excerpt.end_seconds}"`),
    true,
  );

  const attribution = sandbox.__semanticV2Test.trackAttributionHtml(track);
  assert.equal(attribution.includes(track.title), true);
  assert.equal(attribution.includes(track.artist), true);
  assert.equal(attribution.includes(track.license.url), true);
});

test("resets and stops players at committed excerpt boundaries", () => {
  const sandbox = context();
  const listeners = new Map();
  const audio = {
    currentTime: 0,
    dataset: { excerptStart: "5", excerptEnd: "25" },
    readyState: 0,
    pauseCalls: 0,
    addEventListener(name, listener) {
      listeners.set(name, listener);
    },
    pause() {
      this.pauseCalls += 1;
    },
  };
  sandbox.document.querySelectorAll = () => [audio];
  sandbox.__semanticV2Test.bindExcerptPlayers();
  listeners.get("play")();
  assert.equal(audio.currentTime, 5);
  audio.currentTime = 25;
  listeners.get("timeupdate")();
  assert.equal(audio.pauseCalls, 1);
  assert.equal(audio.currentTime, 5);
  audio.currentTime = 2;
  listeners.get("seeking")();
  assert.equal(audio.pauseCalls, 2);
  assert.equal(audio.currentTime, 5);
});

test("preserves byte-locked v2 assets and isolates routes and state", () => {
  const hashes = {
    "index.html": "b245ba0cbdc1be2821e5a7722b946c3e4330b508d848bdc74c593ff68fb628c6",
    "pilot-pack.json":
      "d23d66768f15fd5e37e01ad2a8905d181b4ff278c85674386edcd7dc50b267d3",
    "protocol-v2.json":
      "a88108894e3875159a9ae5b3fae61b01522e9c22647d9ff32748d53d0a5c981c",
  };
  for (const [name, expected] of Object.entries(hashes)) {
    const bytes = readFileSync(new URL(`../evaluate-v2/${name}`, import.meta.url));
    assert.equal(createHash("sha256").update(bytes).digest("hex"), expected);
  }
  assert.equal(v2Html.includes("soundalike-fulltrack-v2"), true);
  assert.equal(html.includes("soundalike-semantic-v2"), true);
  assert.equal(semanticV1Html.includes("soundalike-semantic-v1"), true);
  assert.equal(html.includes("soundalike-fulltrack-v2"), false);
  assert.equal(html.includes("/api/ratings-semantic-v2"), true);
  assert.equal(semanticV1Html.includes("/api/ratings-semantic-v1"), true);
  assert.equal(v2Html.includes('fetch("/api/ratings-v2"'), true);

  const config = JSON.parse(
    readFileSync(new URL("../vercel.json", import.meta.url), "utf8"),
  );
  const routes = Object.fromEntries(
    config.rewrites.map((item) => [item.source, item.destination]),
  );
  assert.equal(routes["/evaluate"], "/evaluate/index.html");
  assert.equal(
    routes["/evaluate-semantic-v2"],
    "/evaluate-semantic-v2/index.html",
  );
  assert.equal(
    routes["/evaluate-semantic-v1"],
    "/evaluate-semantic-v1/index.html",
  );
  assert.equal(routes["/evaluate-v2"], "/evaluate-v2/index.html");
  assert.equal(routes["/evaluate-v1"], "/evaluate-v1/index.html");
  assert.equal(config.functions["api/ratings.js"].maxDuration, 60);
  assert.equal(config.functions["api/ratings-v2.js"].maxDuration, 15);
  assert.equal(config.functions["api/ratings-semantic-v1.js"].maxDuration, 15);
  assert.equal(config.functions["api/ratings-semantic-v2.js"].maxDuration, 15);
});

test("public semantic assets contain no method identity or private unblinding", () => {
  const publicText = [
    html,
    JSON.stringify(protocol),
    JSON.stringify(pack),
  ].join("\n");
  for (const marker of [
    "fulltrack_audio_control_v1",
    "semantic_fulltrack_v1",
    "BEGIN PRIVATE KEY",
  ]) {
    assert.equal(publicText.includes(marker), false);
  }
  assert.equal(protocol.production_recommendation_changed, false);
  assert.equal(pack.promotion_allowed, false);
});
