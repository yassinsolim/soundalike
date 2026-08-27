// Soundalike Marketplace bootstrap. This file is deliberately small and pinned
// by Marketplace to an immutable commit. It only executes signed runtimes.
(function soundalikeMarketplaceBootstrap() {
  "use strict";

  const GLOBAL_PROMISE = "__soundalikeBootstrapPromise";
  if (window[GLOBAL_PROMISE]) return;

  const REPOSITORY = "yassinsolim/soundalike";
  const RAW_ORIGIN = "https://raw.githubusercontent.com";
  const MANIFEST_URL =
    `${RAW_ORIGIN}/${REPOSITORY}/main/integrations/spicetify/releases/stable.json`;
  const MANIFEST_TIMEOUT_MS = 5000;
  const RUNTIME_TIMEOUT_MS = 10000;
  const STORAGE_KEY = "soundalike:marketplace-runtime-lkg:v1";
  const PUBLIC_KEY_SPKI_BASE64 =
    "MCowBQYDK2VwAyEAFkN5Ka3jDavJYiPeH2itZv7+2Brg4UkhhRjVP15pJWk=";
  // This is intentionally an already-published commit, never this bootstrap's
  // commit. It remains available if an update or crypto verification fails.
  const BUNDLED_RUNTIME = {
    version: "1.0.0",
    sequence: 1,
    url: `${RAW_ORIGIN}/${REPOSITORY}/52ee71dfea4503fd1619762613b0d795815bc3e8/integrations/spicetify/soundalike.js`,
    sha256: "684008ee0f627573642991d50a91180e00c4a4e9ff8839d0dfd9690ab19022b3",
    sri: "sha256-aEAI7g9idXNkKZHVCpEYDgDEpOn/iDnQ39lpCrGQIrM=",
  };
  const RUNTIME_URL_PATTERN = new RegExp(
    `^${RAW_ORIGIN.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}/` +
      `${REPOSITORY.replace("/", "\\/")}/[a-f0-9]{40}/` +
      "integrations/spicetify/soundalike\\.js$",
  );

  function bytesFromBase64(value) {
    if (typeof value !== "string" || !/^[A-Za-z0-9+/]+={0,2}$/.test(value)) {
      throw new Error("Invalid base64.");
    }
    const binary = atob(value);
    return Uint8Array.from(binary, (character) => character.charCodeAt(0));
  }

  function canonicalJson(value) {
    if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
    if (value && typeof value === "object") {
      return `{${Object.keys(value).sort().map(
        (key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`,
      ).join(",")}}`;
    }
    return JSON.stringify(value);
  }

  function compareVersions(left, right) {
    const parse = (value) => {
      if (typeof value !== "string" || !/^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/.test(value)) {
        throw new Error("Runtime version must be a semantic version.");
      }
      return value.split(".").map(Number);
    };
    const leftParts = parse(left);
    const rightParts = parse(right);
    for (let index = 0; index < 3; index += 1) {
      if (leftParts[index] !== rightParts[index]) {
        return leftParts[index] > rightParts[index] ? 1 : -1;
      }
    }
    return 0;
  }

  function validRuntime(runtime) {
    if (!runtime || typeof runtime !== "object") throw new Error("Missing runtime.");
    compareVersions(runtime.version, "0.0.0");
    if (!RUNTIME_URL_PATTERN.test(runtime.url || "")) {
      throw new Error("Runtime URL is not an immutable allowlisted URL.");
    }
    if (!/^[a-f0-9]{64}$/.test(runtime.sha256 || "")) {
      throw new Error("Runtime SHA-256 is invalid.");
    }
    const expectedSri = `sha256-${bytesToBase64(hexToBytes(runtime.sha256))}`;
    if (runtime.sri !== expectedSri) throw new Error("Runtime SRI does not match SHA-256.");
    return runtime;
  }

  function hexToBytes(value) {
    return Uint8Array.from(value.match(/.{2}/g), (pair) => parseInt(pair, 16));
  }

  function bytesToBase64(bytes) {
    let binary = "";
    for (const value of bytes) binary += String.fromCharCode(value);
    return btoa(binary);
  }

  function fetchWithTimeout(url, options, timeoutMs) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    return fetch(url, { ...options, signal: controller.signal })
      .finally(() => clearTimeout(timeout));
  }

  async function verifiedManifest() {
    if (!globalThis.crypto?.subtle) throw new Error("WebCrypto is unavailable.");
    const response = await fetchWithTimeout(
      MANIFEST_URL,
      { cache: "no-store", redirect: "error" },
      MANIFEST_TIMEOUT_MS,
    );
    if (!response.ok) throw new Error(`Manifest request failed (${response.status}).`);
    if (response.url && response.url !== MANIFEST_URL) {
      throw new Error("Manifest URL redirected.");
    }
    const manifest = JSON.parse(await response.text());
    if (!manifest || typeof manifest !== "object" ||
      typeof manifest.signature !== "string" || !manifest.payload) {
      throw new Error("Manifest is malformed.");
    }
    const key = await crypto.subtle.importKey(
      "spki",
      bytesFromBase64(PUBLIC_KEY_SPKI_BASE64),
      { name: "Ed25519" },
      false,
      ["verify"],
    );
    const verified = await crypto.subtle.verify(
      { name: "Ed25519" },
      key,
      bytesFromBase64(manifest.signature),
      new TextEncoder().encode(canonicalJson(manifest.payload)),
    );
    if (!verified) throw new Error("Manifest signature verification failed.");
    const payload = manifest.payload;
    if (payload.schema !== 1 || payload.channel !== "stable" ||
      !Number.isSafeInteger(payload.sequence) || payload.sequence < 1) {
      throw new Error("Manifest payload is invalid.");
    }
    return { sequence: payload.sequence, ...validRuntime(payload.runtime) };
  }

  function readLkg() {
    try {
      const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
      if (!parsed || !Number.isSafeInteger(parsed.sequence) || parsed.sequence < 1) return null;
      return { sequence: parsed.sequence, ...validRuntime(parsed.runtime) };
    } catch (_) {
      return null;
    }
  }

  function saveLkg(candidate) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        sequence: candidate.sequence,
        runtime: {
          version: candidate.version,
          url: candidate.url,
          sha256: candidate.sha256,
          sri: candidate.sri,
        },
      }));
    } catch (_) {
      // Storage is an availability optimization, never a trust boundary.
    }
  }

  async function loadVerifiedRuntime(candidate) {
    validRuntime(candidate);
    if (!globalThis.crypto?.subtle) {
      return injectRuntimeScript(candidate);
    }
    const response = await fetchWithTimeout(
      candidate.url,
      { cache: "no-store", redirect: "error" },
      RUNTIME_TIMEOUT_MS,
    );
    if (!response.ok || (response.url && response.url !== candidate.url)) {
      throw new Error("Runtime request failed or redirected.");
    }
    const body = new Uint8Array(await response.arrayBuffer());
    const digest = await crypto.subtle.digest("SHA-256", body);
    const actual = Array.from(new Uint8Array(digest), (byte) =>
      byte.toString(16).padStart(2, "0"),
    ).join("");
    if (actual !== candidate.sha256) throw new Error("Runtime hash verification failed.");

    return injectRuntimeScript(candidate);
  }

  async function injectRuntimeScript(candidate) {
    await new Promise((resolve, reject) => {
      const script = document.createElement("script");
      const timeout = setTimeout(() => {
        script.remove?.();
        reject(new Error("Runtime script timed out."));
      }, RUNTIME_TIMEOUT_MS);
      script.src = candidate.url;
      script.integrity = candidate.sri;
      script.crossOrigin = "anonymous";
      script.onload = () => {
        clearTimeout(timeout);
        resolve();
      };
      script.onerror = () => {
        clearTimeout(timeout);
        reject(new Error("Runtime script SRI verification failed."));
      };
      (document.head || document.documentElement || document.body).appendChild(script);
    });
  }

  async function start() {
    const lkg = readLkg();
    try {
      const candidate = await verifiedManifest();
      if (lkg && (candidate.sequence <= lkg.sequence ||
        compareVersions(candidate.version, lkg.version) < 0)) {
        throw new Error("Refusing a signed runtime downgrade.");
      }
      await loadVerifiedRuntime(candidate);
      saveLkg(candidate);
      return;
    } catch (error) {
      console.warn("[soundalike] Marketplace update rejected; using last known good runtime.", error);
    }
    for (const candidate of [lkg, BUNDLED_RUNTIME]) {
      if (!candidate) continue;
      try {
        await loadVerifiedRuntime(candidate);
        return;
      } catch (error) {
        console.warn("[soundalike] Last known good runtime failed.", error);
      }
    }
    throw new Error("No verified Soundalike runtime could be loaded.");
  }

  window[GLOBAL_PROMISE] = start().catch((error) => {
    console.error("[soundalike] Marketplace bootstrap failed.", error);
  });
})();
