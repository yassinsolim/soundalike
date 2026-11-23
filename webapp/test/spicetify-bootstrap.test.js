import assert from "node:assert/strict";
import {
  createHash,
  createPublicKey,
  generateKeyPairSync,
  sign,
  verify,
  webcrypto,
} from "node:crypto";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import vm from "node:vm";

const bootstrapPath = new URL(
  "../../integrations/spicetify/bootstrap.js",
  import.meta.url,
);
const bootstrapSource = fs.readFileSync(bootstrapPath, "utf8");
const marketplaceManifest = JSON.parse(fs.readFileSync(
  new URL("../../manifest.json", import.meta.url),
  "utf8",
));
const stableManifest = JSON.parse(fs.readFileSync(
  new URL("../../integrations/spicetify/releases/stable-2.json", import.meta.url),
  "utf8",
));
// The feed of bootstraps pinned before 2026-11. Its signing key was lost, so
// it can never change again: those installs keep the runtime it names.
const legacyManifest = JSON.parse(fs.readFileSync(
  new URL("../../integrations/spicetify/releases/stable.json", import.meta.url),
  "utf8",
));
const legacyPublicSpki =
  "MCowBQYDK2VwAyEAFkN5Ka3jDavJYiPeH2itZv7+2Brg4UkhhRjVP15pJWk=";
const legacyStorageKey = "soundalike:marketplace-runtime-lkg:v1";
const storageKey = "soundalike:marketplace-runtime-lkg:v2";
const productionPublicSpki = bootstrapSource.match(
  /PUBLIC_KEY_SPKI_BASE64\s*=\s*\n?\s*"([^"]+)"/,
)?.[1];
const signingKey = generateKeyPairSync("ed25519");
const publicSpki = signingKey.publicKey
  .export({ type: "spki", format: "der" })
  .toString("base64");
assert.ok(productionPublicSpki);
assert.notEqual(productionPublicSpki, legacyPublicSpki);
const testSource = bootstrapSource.replace(productionPublicSpki, publicSpki);
assert.notEqual(testSource, bootstrapSource);
const manifestUrl =
  "https://raw.githubusercontent.com/yassinsolim/soundalike/main/integrations/spicetify/releases/stable-2.json";
const runtimeBase =
  "https://raw.githubusercontent.com/yassinsolim/soundalike/";
const runtimeCdnBase =
  "https://cdn.jsdelivr.net/gh/yassinsolim/soundalike@";
const repoRoot = fileURLToPath(new URL("../..", import.meta.url));
const signerPath = fileURLToPath(new URL(
  "../../integrations/spicetify/tools/sign-release.cjs",
  import.meta.url,
));
const canonicalRepository = "https://github.com/yassinsolim/soundalike.git";

// Published runtimes are pinned to commits that can live only in pull request
// refs, which ordinary clones and CI checkouts do not fetch.
function showPinned(commit, file) {
  const spec = `${commit}:${file}`;
  try {
    return execFileSync("git", ["show", spec], {
      cwd: repoRoot,
      stdio: ["ignore", "pipe", "ignore"],
    });
  } catch {
    execFileSync(
      "git",
      ["fetch", "--quiet", "--no-tags", canonicalRepository, commit],
      { cwd: repoRoot, stdio: "ignore" },
    );
    return execFileSync("git", ["show", spec], { cwd: repoRoot });
  }
}

const bundledRuntime = Buffer.from(showPinned(
  "52ee71dfea4503fd1619762613b0d795815bc3e8",
  "integrations/spicetify/soundalike.js",
));

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map(
      (key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`,
    ).join(",")}}`;
  }
  return JSON.stringify(value);
}

function candidate({
  body = Buffer.from("verified runtime"),
  commit = "a".repeat(40),
  sequence = 2,
  version = "2.0.0",
} = {}) {
  const hash = createHash("sha256").update(body).digest();
  return {
    sequence,
    version,
    url: `${runtimeBase}${commit}/integrations/spicetify/soundalike.js`,
    sha256: hash.toString("hex"),
    sri: `sha256-${hash.toString("base64")}`,
  };
}

function signedManifest(runtime) {
  const payload = {
    schema: 1,
    channel: "stable",
    sequence: runtime.sequence,
    runtime: {
      version: runtime.version,
      url: runtime.url,
      sha256: runtime.sha256,
      sri: runtime.sri,
    },
  };
  return JSON.stringify({
    payload,
    signature: sign(
      null,
      Buffer.from(canonicalJson(payload)),
      signingKey.privateKey,
    ).toString("base64"),
  });
}

function runtimeFields(runtime) {
  return {
    version: runtime.version,
    url: runtime.url,
    sha256: runtime.sha256,
    sri: runtime.sri,
  };
}

function mockResponse(body, url, status = 200) {
  const bytes = Buffer.isBuffer(body) ? body : Buffer.from(body);
  return {
    ok: status >= 200 && status < 300,
    status,
    url,
    async text() {
      return bytes.toString("utf8");
    },
    async arrayBuffer() {
      return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    },
  };
}

function loadBootstrap({
  manifest,
  bodies = new Map(),
  storage = new Map(),
  crypto = webcrypto,
  source = testSource,
  fetchOverride,
  scriptError = false,
  legacyPromise,
} = {}) {
  const scripts = [];
  const requests = [];
  const logs = [];
  const documentRoot = {
    appendChild(script) {
      scripts.push(script);
      queueMicrotask(() => {
        if (scriptError) script.onerror();
        else script.onload();
      });
    },
  };
  const context = {
    AbortController,
    TextEncoder,
    Uint8Array,
    atob,
    btoa,
    clearTimeout,
    console: {
      error(...values) { logs.push(["error", ...values]); },
      warn(...values) { logs.push(["warn", ...values]); },
    },
    crypto,
    ...(legacyPromise ? {
      __soundalikeBootstrapPromise: legacyPromise,
      __soundalikeBootstrapPromiseV2: legacyPromise,
    } : {}),
    document: {
      head: documentRoot,
      createElement(tag) {
        assert.equal(tag, "script");
        return { remove() {} };
      },
    },
    fetch(url, options) {
      requests.push(url);
      if (fetchOverride) return fetchOverride(url, options);
      if (url === manifestUrl) return Promise.resolve(mockResponse(manifest, url));
      if (bodies.has(url)) return Promise.resolve(mockResponse(bodies.get(url), url));
      if (url.includes("/52ee71dfea4503fd1619762613b0d795815bc3e8/")) {
        return Promise.resolve(mockResponse(bundledRuntime, url));
      }
      throw new Error(`Unexpected URL ${url}`);
    },
    localStorage: {
      getItem(key) { return storage.get(key) ?? null; },
      setItem(key, value) { storage.set(key, value); },
    },
    setTimeout,
  };
  context.window = context;
  vm.createContext(context);
  vm.runInContext(source, context);
  return {
    context,
    logs,
    requests,
    scripts,
    storage,
    async done() {
      await context.__soundalikeBootstrapPromiseV3;
      await new Promise((resolve) => setTimeout(resolve, 0));
    },
  };
}

test("loads a valid signed immutable update, verifies its hash, and persists it", async () => {
  const update = candidate();
  const bodies = new Map([[update.url, Buffer.from("verified runtime")]]);
  const app = loadBootstrap({ manifest: signedManifest(update), bodies });

  await app.done();

  assert.deepEqual(app.requests, [manifestUrl, update.url]);
  assert.equal(app.scripts.length, 1);
  assert.equal(app.scripts[0].src, executableRuntimeUrl(update.url));
  assert.equal(app.scripts[0].integrity, update.sri);
  assert.equal(app.scripts[0].crossOrigin, "anonymous");
  assert.deepEqual(
    JSON.parse(app.storage.get(storageKey)),
    { sequence: 2, runtime: runtimeFields(update) },
  );
});

test("rejects a bad signature or a tampered signed payload", async () => {
  const update = candidate();
  const invalidSignature = JSON.parse(signedManifest(update));
  invalidSignature.signature = "A".repeat(invalidSignature.signature.length);
  const app = loadBootstrap({ manifest: JSON.stringify(invalidSignature) });

  await app.done();

  assert.equal(app.requests.length, 2);
  assert.match(app.logs[0].slice(1).join(" "), /rejected/);
  assert.match(app.scripts[0].src, /52ee71dfea4503fd1619762613b0d795815bc3e8/);
});

test("rejects a tampered payload even when its original signature is present", async () => {
  const update = candidate();
  const tampered = JSON.parse(signedManifest(update));
  tampered.payload.runtime.version = "9.9.9";
  const app = loadBootstrap({ manifest: JSON.stringify(tampered) });

  await app.done();

  assert.deepEqual(app.requests, [manifestUrl, expectBundledRuntimeUrl()]);
  assert.match(app.logs[0].slice(1).join(" "), /rejected/);
});

test("rejects a correctly signed runtime from a non-allowlisted origin", async () => {
  const update = candidate();
  update.url = "https://example.test/soundalike.js";
  const app = loadBootstrap({ manifest: signedManifest(update) });

  await app.done();

  assert.deepEqual(app.requests, [manifestUrl, expectBundledRuntimeUrl()]);
  assert.match(app.logs[0].slice(1).join(" "), /rejected/);
});

test("rejects a signed semantic downgrade even with a newer sequence", async () => {
  const lkg = candidate({
    body: Buffer.from("last known good"),
    commit: "b".repeat(40),
    sequence: 3,
    version: "3.0.0",
  });
  const downgrade = candidate({
    body: Buffer.from("downgrade"),
    commit: "c".repeat(40),
    sequence: 4,
    version: "2.0.0",
  });
  const storage = new Map([[storageKey, JSON.stringify({
    sequence: lkg.sequence,
    runtime: runtimeFields(lkg),
  })]]);
  const app = loadBootstrap({
    manifest: signedManifest(downgrade),
    storage,
    bodies: new Map([[lkg.url, Buffer.from("last known good")]]),
  });

  await app.done();

  assert.deepEqual(app.requests, [manifestUrl, lkg.url]);
  assert.equal(app.scripts[0].src, executableRuntimeUrl(lkg.url));
});

test("rejects a runtime hash mismatch before it creates a script", async () => {
  const update = candidate({ body: Buffer.from("expected") });
  const app = loadBootstrap({
    manifest: signedManifest(update),
    bodies: new Map([[update.url, Buffer.from("tampered")]]),
  });

  await app.done();

  assert.equal(app.requests[1], update.url);
  assert.equal(app.scripts.length, 1);
  assert.match(app.scripts[0].src, /52ee71dfea4503fd1619762613b0d795815bc3e8/);
});

test("times out a manifest request and loads the immutable fallback", async () => {
  const app = loadBootstrap({
    manifest: "",
    source: testSource.replace("const MANIFEST_TIMEOUT_MS = 5000;", "const MANIFEST_TIMEOUT_MS = 5;"),
    fetchOverride(url, options) {
      if (url.includes("/52ee71dfea4503fd1619762613b0d795815bc3e8/")) {
        return Promise.resolve(mockResponse(bundledRuntime, url));
      }
      return new Promise((_resolve, reject) => {
        options.signal.addEventListener("abort", () => reject(new Error("aborted")));
      });
    },
  });

  await app.done();

  assert.equal(app.scripts.length, 1);
  assert.match(app.scripts[0].src, /52ee71dfea4503fd1619762613b0d795815bc3e8/);
});

test("times out a runtime request and rejects that update before script injection", async () => {
  const update = candidate();
  const app = loadBootstrap({
    manifest: signedManifest(update),
    source: testSource.replace("const RUNTIME_TIMEOUT_MS = 10000;", "const RUNTIME_TIMEOUT_MS = 5;"),
    fetchOverride(url, options) {
      if (url === manifestUrl) {
        return Promise.resolve(mockResponse(signedManifest(update), url));
      }
      if (url === update.url) {
        return new Promise((_resolve, reject) => {
          options.signal.addEventListener("abort", () => reject(new Error("aborted")));
        });
      }
      return Promise.resolve(mockResponse(bundledRuntime, url));
    },
  });

  await app.done();

  assert.deepEqual(app.requests, [manifestUrl, update.url, expectBundledRuntimeUrl()]);
  assert.equal(app.scripts.length, 1);
  assert.match(app.scripts[0].src, /52ee71dfea4503fd1619762613b0d795815bc3e8/);
});

test("rejects automatic updates without WebCrypto but permits SRI-protected fallback", async () => {
  const app = loadBootstrap({
    manifest: signedManifest(candidate()),
    crypto: {},
  });

  await app.done();

  assert.deepEqual(app.requests, []);
  assert.equal(app.scripts.length, 1);
  assert.match(app.scripts[0].src, /52ee71dfea4503fd1619762613b0d795815bc3e8/);
});

test("uses a persisted LKG when the feed is unavailable", async () => {
  const lkg = candidate({
    body: Buffer.from("persisted runtime"),
    commit: "d".repeat(40),
    sequence: 5,
    version: "5.0.0",
  });
  const storage = new Map([[storageKey, JSON.stringify({
    sequence: lkg.sequence,
    runtime: runtimeFields(lkg),
  })]]);
  const app = loadBootstrap({
    manifest: "",
    storage,
    bodies: new Map([[lkg.url, Buffer.from("persisted runtime")]]),
    fetchOverride(url) {
      if (url === manifestUrl) return Promise.reject(new Error("offline"));
      return Promise.resolve(mockResponse(Buffer.from("persisted runtime"), url));
    },
  });

  await app.done();

  assert.equal(app.scripts[0].src, executableRuntimeUrl(lkg.url));
});

test("shares one global loader promise when bootstrap is evaluated repeatedly", async () => {
  const update = candidate();
  const app = loadBootstrap({
    manifest: signedManifest(update),
    bodies: new Map([[update.url, Buffer.from("verified runtime")]]),
  });
  vm.runInContext(testSource, app.context);

  await app.done();

  assert.deepEqual(app.requests, [manifestUrl, update.url]);
  assert.equal(app.scripts.length, 1);
});

test("loads even when an older bootstrap's promise already exists", async () => {
  const update = candidate();
  const app = loadBootstrap({
    manifest: signedManifest(update),
    bodies: new Map([[update.url, Buffer.from("verified runtime")]]),
    legacyPromise: Promise.resolve(),
  });

  await app.done();

  assert.deepEqual(app.requests, [manifestUrl, update.url]);
  assert.equal(app.scripts.length, 1);
  assert.equal(app.scripts[0].src, executableRuntimeUrl(update.url));
});

function expectBundledRuntimeUrl() {
  return `${runtimeBase}52ee71dfea4503fd1619762613b0d795815bc3e8/integrations/spicetify/soundalike.js`;
}

function executableRuntimeUrl(runtimeUrl) {
  const commit = new URL(runtimeUrl).pathname.split("/")[3];
  return `${runtimeCdnBase}${commit}/integrations/spicetify/soundalike.js`;
}

test("executes verified runtime bytes from an immutable JavaScript CDN URL", async () => {
  const update = candidate();
  const app = loadBootstrap({
    manifest: signedManifest(update),
    bodies: new Map([[update.url, Buffer.from("verified runtime")]]),
  });

  await app.done();

  assert.deepEqual(app.requests, [manifestUrl, update.url]);
  assert.equal(app.scripts[0].src, executableRuntimeUrl(update.url));
  assert.match(app.scripts[0].src, /^https:\/\/cdn\.jsdelivr\.net\/gh\//);
  assert.equal(app.scripts[0].integrity, update.sri);
});

test("Marketplace permanently pins the bootstrap implementation commit", () => {
  assert.equal(marketplaceManifest.main, "integrations/spicetify/bootstrap.js");
  assert.equal(
    marketplaceManifest.branch,
    "38ca29ca9ec760dc40e58a567ca6aaff632ae306",
  );
  assert.match(marketplaceManifest.branch, /^[a-f0-9]{40}$/);
});

test("state saved by the bootstrap of the lost key cannot block the new feed", async () => {
  // That bootstrap saved its last runtime under the old storage name. A high
  // sequence and version there must not read as a downgrade here.
  const old = candidate({
    body: Buffer.from("old runtime"),
    commit: "e".repeat(40),
    sequence: 99,
    version: "9.0.0",
  });
  const saved = JSON.stringify({ sequence: old.sequence, runtime: runtimeFields(old) });
  const update = candidate();
  const app = loadBootstrap({
    manifest: signedManifest(update),
    bodies: new Map([[update.url, Buffer.from("verified runtime")]]),
    storage: new Map([[legacyStorageKey, saved]]),
  });

  await app.done();

  assert.deepEqual(app.requests, [manifestUrl, update.url]);
  assert.equal(app.logs.length, 0);
  assert.equal(app.scripts[0].src, executableRuntimeUrl(update.url));
  assert.equal(app.storage.get(legacyStorageKey), saved);
  assert.deepEqual(
    JSON.parse(app.storage.get(storageKey)),
    { sequence: 2, runtime: runtimeFields(update) },
  );
});

test("the bootstrap reads only the feed of its own key", () => {
  assert.equal(bootstrapSource.match(/releases\/[\w.-]+\.json/g).join(), "releases/stable-2.json");
  assert.ok(bootstrapSource.includes(`"${storageKey}"`));
  assert.ok(!bootstrapSource.includes(legacyStorageKey));
  assert.ok(!bootstrapSource.includes(legacyPublicSpki));
});

function feedVerifies(feed, spki) {
  return verify(
    null,
    Buffer.from(canonicalJson(feed.payload)),
    createPublicKey({ key: Buffer.from(spki, "base64"), format: "der", type: "spki" }),
    Buffer.from(feed.signature, "base64"),
  );
}

function assertFeedNamesItsRuntime(feed) {
  assert.equal(feed.payload.schema, 1);
  assert.equal(feed.payload.channel, "stable");
  const runtime = Buffer.from(showPinned(
    new URL(feed.payload.runtime.url).pathname.split("/")[3],
    "integrations/spicetify/soundalike.js",
  ));
  const digest = createHash("sha256").update(runtime).digest();
  assert.equal(feed.payload.runtime.sha256, digest.toString("hex"));
  assert.equal(feed.payload.runtime.sri, `sha256-${digest.toString("base64")}`);
  assert.match(
    runtime.toString("utf8"),
    new RegExp(
      `const RUNTIME_SEMANTIC_VERSION = ` +
        `"${feed.payload.runtime.version.replaceAll(".", "\\.")}";`,
    ),
  );
}

test("committed stable feed verifies against the production key and runtime", () => {
  assert.equal(feedVerifies(stableManifest, productionPublicSpki), true);
  assert.equal(feedVerifies(stableManifest, legacyPublicSpki), false);
  assertFeedNamesItsRuntime(stableManifest);
  // One line of sequences across both feeds, so no number is ever reused.
  assert.ok(stableManifest.payload.sequence > legacyManifest.payload.sequence);
});

test("the feed of the lost key stays exactly as it was last signed", () => {
  assert.equal(feedVerifies(legacyManifest, legacyPublicSpki), true);
  assert.equal(feedVerifies(legacyManifest, productionPublicSpki), false);
  assertFeedNamesItsRuntime(legacyManifest);
  assert.equal(legacyManifest.payload.sequence, 20);
  assert.equal(legacyManifest.payload.runtime.version, "2.13.0");
  assert.equal(
    legacyManifest.signature,
    "AyG6ITTB8FRhieIsW95nfbnhB2iwGglkAc+tJZgPrMatHAAO3sFTHDJlfQAh0ZWUafh3HSSmys4HFJlpfn+KBQ==",
  );
});

test("release signer enforces matching runtime and monotonic versions", () => {
  // Sign inside a throwaway repository so the test does not depend on the
  // published release or on local changes to the runtime.
  const scratch = fs.mkdtempSync(path.join(os.tmpdir(), "soundalike-signer-"));
  const keyDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "soundalike-signing-"));
  const git = (...args) => execFileSync("git", [
    "-c", "user.name=Signer Test",
    "-c", "user.email=signer-test@example.invalid",
    "-c", "commit.gpgsign=false",
    "-c", "core.autocrlf=false",
    ...args,
  ], { cwd: scratch, stdio: "pipe" }).toString().trim();
  const runtimePath = "integrations/spicetify/soundalike.js";
  const runtimeFile = path.join(scratch, ...runtimePath.split("/"));
  const output = path.join(scratch, "integrations", "spicetify", "releases", "stable.json");
  const keyFile = path.join(keyDirectory, "private.pem");
  const failure = (pattern) => (error) => {
    assert.match(String(error.stderr), pattern);
    return true;
  };
  try {
    fs.mkdirSync(path.join(scratch, "integrations", "spicetify", "tools"), { recursive: true });
    fs.mkdirSync(path.dirname(output), { recursive: true });
    fs.copyFileSync(
      signerPath,
      path.join(scratch, "integrations", "spicetify", "tools", "sign-release.cjs"),
    );
    const runtime = 'const RUNTIME_SEMANTIC_VERSION = "3.0.0";\n';
    fs.writeFileSync(runtimeFile, runtime);
    git("init", "--quiet");
    git("add", ".");
    git("commit", "--quiet", "-m", "release");
    const commit = git("rev-parse", "HEAD");
    fs.writeFileSync(keyFile, signingKey.privateKey.export({ type: "pkcs8", format: "pem" }));
    fs.writeFileSync(output, JSON.stringify({
      payload: { sequence: 4, runtime: { version: "2.9.0" } },
    }));
    const signRelease = ({ version = "3.0.0", sequence = 5 } = {}) => execFileSync(
      process.execPath,
      [
        "integrations/spicetify/tools/sign-release.cjs",
        "--runtime-file", runtimePath,
        "--runtime-url", `${runtimeBase}${commit}/${runtimePath}`,
        "--version", version,
        "--sequence", String(sequence),
        "--out", output,
        "--key", keyFile,
      ],
      { cwd: scratch, stdio: "pipe" },
    );

    signRelease();
    const signed = JSON.parse(fs.readFileSync(output, "utf8"));
    const digest = createHash("sha256").update(runtime).digest();
    assert.equal(signed.payload.sequence, 5);
    assert.equal(signed.payload.runtime.version, "3.0.0");
    assert.equal(signed.payload.runtime.sha256, digest.toString("hex"));
    assert.equal(
      verify(
        null,
        Buffer.from(canonicalJson(signed.payload)),
        signingKey.publicKey,
        Buffer.from(signed.signature, "base64"),
      ),
      true,
    );

    assert.throws(() => signRelease(), failure(/increase monotonically/));
    assert.throws(
      () => signRelease({ version: "3.0.1", sequence: 6 }),
      failure(/does not match the runtime source/),
    );
    fs.writeFileSync(runtimeFile, `${runtime}// local edit\n`);
    assert.throws(
      () => signRelease({ sequence: 6 }),
      failure(/does not match the immutable commit/),
    );
  } finally {
    fs.rmSync(scratch, { recursive: true, force: true });
    fs.rmSync(keyDirectory, { recursive: true, force: true });
  }
});
