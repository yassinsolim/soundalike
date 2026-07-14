import assert from "node:assert/strict";
import { webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const html = readFileSync(
  new URL("../evaluate/index.html", import.meta.url),
  "utf8",
);
const script = html.match(/<script>\s*([\s\S]*?)\s*<\/script>/)?.[1];
const protocol = JSON.parse(
  readFileSync(new URL("../evaluate/protocol.json", import.meta.url), "utf8"),
);
const lists = JSON.parse(
  readFileSync(new URL("../evaluate/served-lists.json", import.meta.url), "utf8"),
);
const V16_PROTOCOL =
  "c94ce615c68cde595b4e48ac5010297d76bedbed52948b10d315a39286117727";
const V16_LISTS =
  "809b98ae4314b396ffb33f7349fee72c94e1a80a33d84b1661ab83166a52b9e9";

function evaluatorContext() {
  const values = new Map();
  const elements = new Map();
  const element = () => ({
    checked: false,
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
      return {
        click() {},
        replaceChildren() {},
        set textContent(_value) {},
        get innerHTML() {
          return "";
        },
      };
    },
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, element());
      return elements.get(id);
    },
  };
  const localStorage = {
    getItem(key) {
      return values.has(key) ? values.get(key) : null;
    },
    setItem(key, value) {
      values.set(key, String(value));
    },
    removeItem(key) {
      values.delete(key);
    },
  };
  const context = vm.createContext({
    Blob,
    Map,
    Number,
    Promise,
    Set,
    String,
    TextEncoder,
    URL,
    URLSearchParams,
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
    setTimeout,
  });
  vm.runInContext(
    `${script}
globalThis.__evaluatorTest = {
  restoreAutosave,
  setStudy(p, l) { protocol = p; pack = l; },
};`,
    context,
  );
  context.__evaluatorTest.setStudy(protocol, lists);
  return { context, localStorage, values };
}

function oldAutosave() {
  const ids = lists.seeds[0].results.slice(0, 5).map((item) => item.result_id);
  return {
    schema_version: 16,
    source_kind: "human_listener",
    provider: "hosted_client_only_evaluator",
    anonymous_rater_id: `anon-${"1".repeat(24)}`,
    session_id: `session-${"2".repeat(24)}`,
    protocol_sha256: V16_PROTOCOL,
    served_lists_sha256: V16_LISTS,
    local_session_key: "a".repeat(64),
    started_at: "2026-07-14T00:00:00.000Z",
    last_activity_at: "2026-07-14T00:00:05.000Z",
    current_seed: 3,
    result_ratings: Object.fromEntries(
      ids.map((id, index) => [
        id,
        {
          similarity: "very_similar",
          score_0_10: 8,
          junk_or_version: false,
          rated_at: `2026-07-14T00:00:0${index + 1}.000Z`,
          interaction_ms: 1000,
        },
      ]),
    ),
    list_ratings: {},
    lastInteractionAt: 123,
  };
}

function installOld(localStorage, saved) {
  const key = `soundalike-human-v16:${V16_PROTOCOL}:${saved.anonymous_rater_id}:${saved.session_id}`;
  localStorage.setItem(`soundalike-human-v16-current:${V16_PROTOCOL}`, key);
  localStorage.setItem(key, JSON.stringify(saved));
  return key;
}

test("trusted v16 autosave migrates all five ratings without deleting its fallback", () => {
  const { context, localStorage, values } = evaluatorContext();
  const old = oldAutosave();
  const oldKey = installOld(localStorage, old);

  const migrated = context.__evaluatorTest.restoreAutosave();

  assert.equal(migrated.schema_version, 17);
  assert.equal(migrated.provider, "hosted_private_submission_evaluator");
  assert.equal(migrated.protocol_sha256, protocol.content_sha256);
  assert.equal(migrated.served_lists_sha256, lists.content_sha256);
  assert.deepEqual(
    JSON.parse(JSON.stringify(migrated.result_ratings)),
    old.result_ratings,
  );
  assert.equal(Object.keys(migrated.result_ratings).length, 5);
  assert.equal(migrated.current_seed, old.current_seed);
  assert.equal(migrated.local_session_key, old.local_session_key);
  assert.deepEqual(
    JSON.parse(JSON.stringify(migrated.migration)),
    {
      from_schema_version: 16,
      from_provider: "hosted_client_only_evaluator",
      from_protocol_sha256: V16_PROTOCOL,
      from_served_lists_sha256: V16_LISTS,
      migrated_at: migrated.migration.migrated_at,
    },
  );
  assert.equal(values.has(oldKey), true);
});

test("v16 autosave migration rejects wrong hashes and invented IDs", () => {
  for (const mutate of [
    (saved) => {
      saved.protocol_sha256 = "0".repeat(64);
    },
    (saved) => {
      saved.result_ratings[`T14-${"f".repeat(24)}`] =
        saved.result_ratings[Object.keys(saved.result_ratings)[0]];
    },
  ]) {
    const { context, localStorage } = evaluatorContext();
    const old = oldAutosave();
    mutate(old);
    installOld(localStorage, old);
    assert.equal(context.__evaluatorTest.restoreAutosave(), null);
  }
});
