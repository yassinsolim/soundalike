#!/usr/bin/env node
"use strict";

const { createHash, createPrivateKey, sign } = require("node:crypto");
const { existsSync, readFileSync, writeFileSync } = require("node:fs");
const { execFileSync } = require("node:child_process");
const { isAbsolute, relative, resolve } = require("node:path");

function argument(name) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function required(name) {
  const value = argument(name);
  if (!value || value.startsWith("--")) throw new Error(`Missing ${name}.`);
  return value;
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

function inside(child, parent) {
  const path = relative(parent, child);
  return path && !path.startsWith("..") && !isAbsolute(path);
}

function compareVersions(left, right) {
  const parse = (value) => value.split(".").map(Number);
  const leftParts = parse(left);
  const rightParts = parse(right);
  for (let index = 0; index < 3; index += 1) {
    if (leftParts[index] !== rightParts[index]) {
      return leftParts[index] > rightParts[index] ? 1 : -1;
    }
  }
  return 0;
}

function main() {
  const repo = resolve(__dirname, "../../..");
  const runtimeFile = resolve(required("--runtime-file"));
  const output = resolve(required("--out"));
  const runtimeUrl = required("--runtime-url");
  const version = required("--version");
  const sequence = Number(required("--sequence"));
  const keyFile = argument("--key") || process.env.SOUNDALIKE_RELEASE_SIGNING_KEY_FILE;

  if (!inside(runtimeFile, repo) || !inside(output, repo)) {
    throw new Error("Runtime and output paths must be inside this checkout.");
  }
  if (keyFile && (!isAbsolute(keyFile) || inside(resolve(keyFile), repo))) {
    throw new Error("The signing key file must be absolute and outside this checkout.");
  }
  const keyPem = process.env.SOUNDALIKE_RELEASE_SIGNING_KEY_PEM ||
    (keyFile && readFileSync(resolve(keyFile), "utf8"));
  if (!keyPem) throw new Error("Set a protected signing-key environment value or file.");
  if (!/^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/.test(version)) {
    throw new Error("Version must be MAJOR.MINOR.PATCH.");
  }
  if (!Number.isSafeInteger(sequence) || sequence < 1) {
    throw new Error("Sequence must be a positive integer.");
  }
  if (!/^https:\/\/raw\.githubusercontent\.com\/yassinsolim\/soundalike\/[a-f0-9]{40}\/integrations\/spicetify\/soundalike\.js$/.test(runtimeUrl)) {
    throw new Error("Runtime URL must be an immutable allowlisted Soundalike URL.");
  }

  const immutableCommit = new URL(runtimeUrl).pathname.split("/")[3];
  const runtimePath = relative(repo, runtimeFile).replaceAll("\\", "/");
  if (runtimePath !== "integrations/spicetify/soundalike.js") {
    throw new Error("Runtime file must be integrations/spicetify/soundalike.js.");
  }
  const publishedRuntime = execFileSync(
    "git",
    ["show", `${immutableCommit}:${runtimePath}`],
    { cwd: repo },
  );
  try {
    execFileSync(
      "git",
      ["diff", "--quiet", immutableCommit, "--", runtimePath],
      { cwd: repo },
    );
  } catch {
    throw new Error("The checked-out runtime does not match the immutable commit.");
  }
  if (!publishedRuntime.toString("utf8").includes(
    `const RUNTIME_SEMANTIC_VERSION = "${version}";`,
  )) {
    throw new Error("The requested version does not match the runtime source.");
  }
  if (existsSync(output)) {
    const previous = JSON.parse(readFileSync(output, "utf8"));
    const previousSequence = previous?.payload?.sequence;
    const previousVersion = previous?.payload?.runtime?.version;
    if (!Number.isSafeInteger(previousSequence) || typeof previousVersion !== "string") {
      throw new Error("The existing release feed is malformed.");
    }
    if (sequence <= previousSequence || compareVersions(version, previousVersion) <= 0) {
      throw new Error("Release sequence and version must both increase monotonically.");
    }
  }
  const hash = createHash("sha256").update(publishedRuntime).digest();
  const payload = {
    schema: 1,
    channel: "stable",
    sequence,
    runtime: {
      version,
      url: runtimeUrl,
      sha256: hash.toString("hex"),
      sri: `sha256-${hash.toString("base64")}`,
    },
  };
  const privateKey = createPrivateKey(keyPem);
  if (privateKey.asymmetricKeyType !== "ed25519") {
    throw new Error("The release signing key must be Ed25519.");
  }
  const signature = sign(null, Buffer.from(canonicalJson(payload)), privateKey);
  writeFileSync(output, `${JSON.stringify({
    payload,
    signature: signature.toString("base64"),
  }, null, 2)}\n`);
  console.log(`Signed stable manifest sequence ${sequence} for ${version}.`);
}

try {
  main();
} catch (error) {
  console.error(`Release signing failed: ${error.message}`);
  process.exitCode = 1;
}
