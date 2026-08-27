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
  new URL("../../integrations/spicetify/releases/stable.json", import.meta.url),
  "utf8",
));
const productionPublicSpki = bootstrapSource.match(
  /PUBLIC_KEY_SPKI_BASE64\s*=\s*\n?\s*"([^"]+)"/,
)?.[1];
const signingKey = generateKeyPairSync("ed25519");
const publicSpki = signingKey.publicKey
  .export({ type: "spki", format: "der" })
  .toString("base64");
const testSource = bootstrapSource.replace(
  "MCowBQYDK2VwAyEAFkN5Ka3jDavJYiPeH2itZv7+2Brg4UkhhRjVP15pJWk=",
  publicSpki,
);
const manifestUrl =
  "https://raw.githubusercontent.com/yassinsolim/soundalike/main/integrations/spicetify/releases/stable.json";
const runtimeBase =
  "https://raw.githubusercontent.com/yassinsolim/soundalike/";
const runtimeCdnBase =
  "https://cdn.jsdelivr.net/gh/yassinsolim/soundalike@";
const bundledRuntime = Buffer.from(execFileSync(
  "git",
  ["show", "52ee71dfea4503fd1619762613b0d795815bc3e8:integrations/spicetify/soundalike.js"],
  { cwd: new URL("../..", import.meta.url) },
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
    ...(legacyPromise ? { __soundalikeBootstrapPromise: legacyPromise } : {}),
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
      await context.__soundalikeBootstrapPromiseV2;
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
    JSON.parse(app.storage.get("soundalike:marketplace-runtime-lkg:v1")),
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
  const storage = new Map([["soundalike:marketplace-runtime-lkg:v1", JSON.stringify({
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
  const storage = new Map([["soundalike:marketplace-runtime-lkg:v1", JSON.stringify({
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

test("loads even when the broken legacy bootstrap promise already exists", async () => {
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

test("committed stable feed verifies against the production key and runtime", () => {
  assert.ok(productionPublicSpki);
  const publicKey = createPublicKey({
    key: Buffer.from(productionPublicSpki, "base64"),
    format: "der",
    type: "spki",
  });
  assert.equal(
    verify(
      null,
      Buffer.from(canonicalJson(stableManifest.payload)),
      publicKey,
      Buffer.from(stableManifest.signature, "base64"),
    ),
    true,
  );
  const runtime = Buffer.from(execFileSync(
    "git",
    [
      "show",
      `${new URL(stableManifest.payload.runtime.url).pathname.split("/")[3]}` +
        ":integrations/spicetify/soundalike.js",
    ],
    { cwd: new URL("../..", import.meta.url) },
  ));
  const digest = createHash("sha256").update(runtime).digest();
  assert.equal(stableManifest.payload.runtime.sha256, digest.toString("hex"));
  assert.equal(
    stableManifest.payload.runtime.sri,
    `sha256-${digest.toString("base64")}`,
  );
  assert.match(
    runtime.toString("utf8"),
    new RegExp(
      `const RUNTIME_SEMANTIC_VERSION = ` +
        `"${stableManifest.payload.runtime.version.replaceAll(".", "\\.")}";`,
    ),
  );
});

test("release signer enforces matching runtime and monotonic versions", () => {
  const root = fileURLToPath(new URL("../..", import.meta.url));
  const releaseDirectory = fs.mkdtempSync(path.join(
    root,
    "integrations",
    "spicetify",
    "releases",
    "signer-test-",
  ));
  const keyDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "soundalike-signing-"));
  const output = path.join(releaseDirectory, "stable.json");
  const keyFile = path.join(keyDirectory, "private.pem");
  fs.writeFileSync(
    keyFile,
    signingKey.privateKey.export({ type: "pkcs8", format: "pem" }),
  );
  fs.writeFileSync(output, JSON.stringify({
    payload: {
      sequence: stableManifest.payload.sequence - 1,
      runtime: { version: "2.1.0" },
    },
  }));
  const command = [
    "integrations/spicetify/tools/sign-release.cjs",
    "--runtime-file",
    "integrations/spicetify/soundalike.js",
    "--runtime-url",
    stableManifest.payload.runtime.url,
    "--version",
    stableManifest.payload.runtime.version,
    "--sequence",
    String(stableManifest.payload.sequence),
    "--out",
    output,
    "--key",
    keyFile,
  ];
  try {
    execFileSync(process.execPath, command, { cwd: root });
    const signed = JSON.parse(fs.readFileSync(output, "utf8"));
    assert.equal(signed.payload.sequence, stableManifest.payload.sequence);
    assert.throws(
      () => execFileSync(process.execPath, command, {
        cwd: root,
        stdio: "pipe",
      }),
      /Command failed/,
    );
  } finally {
    fs.rmSync(releaseDirectory, { recursive: true, force: true });
    fs.rmSync(keyDirectory, { recursive: true, force: true });
  }
});
