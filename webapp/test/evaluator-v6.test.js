import assert from "node:assert/strict";
import { webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const html = readFileSync(new URL("../evaluate/index.html", import.meta.url), "utf8");
const script = html.match(/<script>\s*([\s\S]*?)\s*<\/script>/)?.[1];
const protocol = JSON.parse(
  readFileSync(new URL("../evaluate/protocol-v6.json", import.meta.url), "utf8"),
);
const pack = JSON.parse(
  readFileSync(
    new URL("../evaluate/active-pack-v6.json", import.meta.url),
    "utf8",
  ),
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

function collectIds(value) {
  const ids = new Set();
  if (Array.isArray(value)) {
    value.forEach((item) => collectIds(item).forEach((id) => ids.add(id)));
  } else if (value && typeof value === "object") {
    for (const [key, item] of Object.entries(value)) {
      if (
        (key === "track_id" || key === "seed_track_id") &&
        Number.isInteger(item)
      ) {
        ids.add(item);
      } else {
        collectIds(item).forEach((id) => ids.add(id));
      }
    }
  }
  return ids;
}

function collectArtistNames(value) {
  const names = new Set();
  if (Array.isArray(value)) {
    value.forEach((item) =>
      collectArtistNames(item).forEach((name) => names.add(name)),
    );
  } else if (value && typeof value === "object") {
    for (const [key, item] of Object.entries(value)) {
      if ((key === "artist" || key === "artist_name") && typeof item === "string") {
        names.add(item.trim().toLocaleLowerCase("en-US"));
      } else {
        collectArtistNames(item).forEach((name) => names.add(name));
      }
    }
  }
  return names;
}

test("validates only the exact active V6 pack and protocol", async () => {
  const sandbox = context();
  const trustedProtocol = clone(sandbox, protocol);
  const trustedPack = clone(sandbox, pack);
  assert.equal(
    await sandbox.__v6Test.docHash(trustedProtocol),
    protocol.content_sha256,
  );
  assert.equal(await sandbox.__v6Test.docHash(trustedPack), pack.content_sha256);
  assert.equal(
    await sandbox.__v6Test.validateStudy(trustedProtocol, trustedPack),
    true,
  );
  const tampered = clone(sandbox, pack);
  tampered.independent_holdout = true;
  await assert.rejects(
    sandbox.__v6Test.validateStudy(clone(sandbox, protocol), tampered),
    /pack/,
  );
});

test("marks V6 as model-improvement evidence rather than a promotion holdout", () => {
  assert.equal(pack.development_evidence, true);
  assert.equal(pack.independent_holdout, false);
  assert.equal(pack.promotion_allowed, false);
  assert.equal(pack.evidence_role, "development_model_improvement");
  assert.equal(protocol.development_evidence, true);
  assert.equal(protocol.independent_holdout, false);
  assert.equal(protocol.evidence_role, "development_model_improvement");
  assert.match(html, /development\/model-improvement evidence/);
  assert.match(html, /not an independent promotion holdout/);
});

test("contains 16 unique comparisons, two anchors, and 80 artists", () => {
  const sandbox = context();
  const signatures = pack.tasks.map((task) =>
    sandbox.__v6Test.taskSignature(clone(sandbox, task)),
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
  assert.equal(Object.keys(pack.tracks).length, 80);
  assert.equal(
    new Set(
      Object.values(pack.tracks).map(
        (track) => track.source_identity.artist_id,
      ),
    ).size,
    80,
  );
});

test("active V6 excludes every earlier exposed track and artist", () => {
  const earlierSameCorpus = [
    "../evaluate-v2/pilot-pack.json",
    "../evaluate-semantic-v1/semantic-pack.json",
    "../evaluate-semantic-v2/semantic-pack.json",
    "../evaluate-pacing-v3/pacing-pack.json",
    "../evaluate-v4/active-pack.json",
    "../evaluate-v5/active-pack.json",
  ].map((path) => JSON.parse(readFileSync(new URL(path, import.meta.url), "utf8")));
  const earlierTrackIds = new Set();
  const earlierArtistIds = new Set();
  for (const earlier of earlierSameCorpus) {
    collectIds(earlier).forEach((id) => earlierTrackIds.add(id));
    for (const track of Object.values(earlier.tracks || {})) {
      const artistId = track?.source_identity?.artist_id;
      if (Number.isInteger(artistId)) earlierArtistIds.add(artistId);
    }
  }
  const currentTrackIds = new Set(Object.keys(pack.tracks).map(Number));
  const currentArtistIds = new Set(
    Object.values(pack.tracks).map(
      (track) => track.source_identity.artist_id,
    ),
  );
  assert.deepEqual(
    [...currentTrackIds].filter((id) => earlierTrackIds.has(id)),
    [],
  );
  assert.deepEqual(
    [...currentArtistIds].filter((id) => earlierArtistIds.has(id)),
    [],
  );

  const v1 = JSON.parse(
    readFileSync(new URL("../evaluate-v1/served-lists.json", import.meta.url)),
  );
  const earlierNames = collectArtistNames(v1);
  assert.deepEqual(
    Object.values(pack.tracks)
      .map((track) => track.artist.trim().toLocaleLowerCase("en-US"))
      .filter((name) => earlierNames.has(name)),
    [],
  );
  assert.equal(
    Object.keys(pack.provenance.prior_exposure_pack_sha256s).length,
    7,
  );
  assert.equal(pack.provenance.includes_v5_exposure, true);
  assert.equal(
    pack.provenance.excludes_all_prior_exposed_tracks_and_artists,
    true,
  );
});

test("builds and imports strict complete-ranking partial exports", async () => {
  const sandbox = context();
  const state = sandbox.__v6Test.emptyState();
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

  sandbox.__v6Test.setStudy(protocol, pack, state);
  assert.equal(sandbox.__v6Test.validState(state), true);
  const exported = await sandbox.__v6Test.buildExport();
  assert.equal(sandbox.__v6Test.validExport(exported), true);
  const imported = await sandbox.__v6Test.importExport(exported);
  assert.equal(sandbox.__v6Test.validState(imported), true);
  assert.deepEqual(
    Array.from(imported.task_ratings[task.task_id].ranked_choice_ids),
    task.candidates.map((choice) => choice.choice_id),
  );
});

test("rejects non-string anonymous identifiers without regex coercion", async () => {
  const sandbox = context();
  const state = sandbox.__v6Test.emptyState();
  state.anonymous_rater_id = [`anon-v6-${"1".repeat(24)}`];
  assert.equal(sandbox.__v6Test.validState(state), false);

  const trusted = sandbox.__v6Test.emptyState();
  sandbox.__v6Test.setStudy(protocol, pack, trusted);
  const task = pack.tasks[0];
  trusted.started_at = "2026-07-30T00:00:00.000Z";
  trusted.last_activity_at = "2026-07-30T00:00:01.000Z";
  trusted.task_ratings[task.task_id] = clone(sandbox, {
    outcome: "rated",
    ranked_choice_ids: task.candidates.map((choice) => choice.choice_id),
    worst_primary_reason: "tempo_pacing",
    skip_reason: null,
    completed_at: trusted.last_activity_at,
    interaction_ms: 1000,
  });
  const exported = await sandbox.__v6Test.buildExport();
  exported.session_id = [`v6-session-${"2".repeat(24)}`];
  assert.equal(sandbox.__v6Test.validExport(exported), false);
});

test("uses explicit four-slot ranking labels and preserves worst-item reason", () => {
  for (const label of [
    "1 — Most similar",
    "2 — Next most similar",
    "3 — Second least similar",
    "4 — Least similar",
  ]) {
    assert.equal(html.includes(label), true);
  }
  assert.match(html, /Primary reason the fourth-ranked candidate misses:/);
  assert.match(html, /<label for="skip-reason"><strong>Skip reason:/);
  assert.deepEqual(protocol.ranking_slots, [
    "most_similar",
    "next_most_similar",
    "second_least_similar",
    "least_similar",
  ]);
});

test("loads V6 assets from the canonical active evaluator route", () => {
  assert.equal(html.includes('fetch("/evaluate/protocol-v6.json"'), true);
  assert.equal(html.includes('fetch("/evaluate/active-pack-v6.json"'), true);
  assert.equal(html.includes('fetch("/api/ratings-v6"'), true);
  assert.equal(html.includes('study:"development-v6-ranking"'), true);
  assert.equal(
    html.match(/<meta http-equiv="Content-Security-Policy"[^>]+>/)?.[0]
      .includes("frame-ancestors"),
    false,
  );
});

test("hides attribution until completion and preserves excerpt bounds", () => {
  const sandbox = context();
  const track = pack.tracks[String(pack.tasks[0].seed_track_id)];
  const hidden = sandbox.__v6Test.trackHtml(clone(sandbox, track), "Seed", false);
  const revealed = sandbox.__v6Test.trackHtml(
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
  assert.ok(
    track.audio.excerpt.end_seconds - track.audio.excerpt.start_seconds <= 20,
  );
});

test("public V6 assets contain no method mapping or private key", () => {
  const text = [html, JSON.stringify(protocol), JSON.stringify(pack)].join("\n");
  for (const marker of [
    "method_bindings",
    "method_orders",
    "method_rankings",
    "candidate_selection_sources",
    "blinding_key",
    '"control"',
    '"challenger"',
  ]) {
    assert.equal(text.includes(marker), false);
  }
  assert.equal(text.includes("textarea"), false);
});
