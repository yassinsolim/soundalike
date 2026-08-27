# Marketplace runtime release

Marketplace pins `bootstrap.js` to an immutable commit. The bootstrap accepts
only the stable manifest at its fixed allowlisted URL, verifies its Ed25519
signature and monotonic sequence, then loads only an immutable runtime URL
whose SHA-256 and SRI agree. Do not change Marketplace's manifest for ordinary
runtime releases.

Runtime bytes are fetched and hashed from their signed immutable GitHub Raw
URL. They are executed from the corresponding immutable jsDelivr URL because
GitHub Raw serves JavaScript as `text/plain` with `nosniff`, which Chromium
correctly refuses to execute. Browser-enforced SRI binds the CDN response to
the already verified SHA-256.

## Release a runtime

1. Update `RUNTIME_SEMANTIC_VERSION` in `soundalike.js` using a new
   `MAJOR.MINOR.PATCH` version. Commit the runtime and bootstrap implementation.
   Record that commit's full 40-character SHA as `RUNTIME_COMMIT`.
2. Keep the Ed25519 private signing key outside the checkout, with restrictive
   filesystem permissions. Set `SOUNDALIKE_RELEASE_SIGNING_KEY_FILE` to its
   absolute path (or provide it only through protected CI secret environment
   configuration). Never copy the private key into this repository, an issue,
   logs, or a release artifact.
3. Generate the next signed, monotonic stable feed. The helper hashes the exact
   immutable Git object named by the runtime URL, not line-ending-transformed
   checkout content:

   ```powershell
   node integrations/spicetify/tools/sign-release.cjs `
     --runtime-file integrations/spicetify/soundalike.js `
     --runtime-url "https://raw.githubusercontent.com/yassinsolim/soundalike/$env:RUNTIME_COMMIT/integrations/spicetify/soundalike.js" `
     --version 2.0.0 `
     --sequence 2 `
     --out integrations/spicetify/releases/stable.json
   ```

4. Review the feed diff, run `node --test` from `webapp`, and commit only the
   public manifest and signature. Publish that commit normally. The bootstrap
   will discover it from the signed feed.

## Bootstrap changes

Changing `bootstrap.js` requires a deliberate Marketplace manifest update.
Commit the bootstrap first. In a separate follow-up commit, point
`manifest.json` at the first commit's immutable SHA. This avoids a
self-referential bootstrap pin and gives Marketplace a permanent bootstrap
artifact. Existing users need one final Marketplace reinstall to receive a new
bootstrap; signed runtime releases thereafter are automatic.
