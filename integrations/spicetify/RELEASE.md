# Marketplace runtime release

Marketplace pins `bootstrap.js` to an immutable commit. The bootstrap accepts
only the stable feed at its fixed allowlisted URL, verifies its Ed25519
signature and monotonic sequence, then loads only an immutable runtime URL
whose SHA-256 and SRI agree. Do not change Marketplace's manifest for ordinary
runtime releases.

## Feeds and keys

| Feed | Key id | Read by | State |
| --- | --- | --- | --- |
| `releases/stable-2.json` | `2026-11` | bootstraps pinned from 2026-11-23 on | current; sequences continue from 21 |
| `releases/stable.json` | first key | bootstraps pinned before 2026-11-23 | frozen at sequence 20, runtime 2.13.0 |

The first signing key was lost in 2026-11, and no copy of it exists. Nothing
can be signed for `releases/stable.json` again, so that file must stay exactly
as it is: installs made before 2026-11-23 verify it on every start and keep
running runtime 2.13.0. Changing or removing it would not update them. It would
only send them to the runtime they last saved, or on a fresh profile to the
1.0.0 runtime bundled in their bootstrap. They move to current runtimes by
reinstalling Soundalike from Marketplace once, which replaces their bootstrap.

Key `2026-11` is an Ed25519 key. Its public half is `PUBLIC_KEY_SPKI_BASE64` in
`bootstrap.js`; the SHA-256 of that SPKI DER is
`88c0382ef14b5e931459ea3e8427b2b4fb68ef6eb4f050d62c2fdb284ca6b25d`. The private
half is expected on the maintainer's release machine, outside every checkout,
at `.soundalike/release-keys/soundalike-marketplace-release-ed25519-private-2026-11.pem`
under the user profile, readable by that user only, with an offline backup.
Losing it again means another new key, feed and bootstrap, and another
reinstall for every user.

The bootstrap of key `2026-11` saves its last good runtime under its own
storage name, so a sequence saved by an older bootstrap cannot make the new
feed look like a downgrade. Sequences still continue across feeds (20, then
21) so that no number is ever used twice.

Both bootstraps accept a feed only when its sequence is higher than the one
they saved. On every start with an unchanged feed they therefore log
`Marketplace update rejected; using last known good runtime` with `Refusing a
signed runtime downgrade` to the console and load the saved runtime, which is
the one the feed names. That warning is expected and is not an error.

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
   filesystem permissions (see Feeds and keys for where it lives). Set
   `SOUNDALIKE_RELEASE_SIGNING_KEY_FILE` to its absolute path (or provide it
   only through protected CI secret environment configuration). Never copy the
   private key into this repository, an issue, logs, or a release artifact.
3. Generate the next signed, monotonic stable feed. The helper hashes the exact
   immutable Git object named by the runtime URL, not line-ending-transformed
   checkout content:

   ```powershell
   node integrations/spicetify/tools/sign-release.cjs `
     --runtime-file integrations/spicetify/soundalike.js `
     --runtime-url "https://raw.githubusercontent.com/yassinsolim/soundalike/$env:RUNTIME_COMMIT/integrations/spicetify/soundalike.js" `
     --version 2.16.0 `
     --sequence 22 `
     --out integrations/spicetify/releases/stable-2.json
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

A bootstrap with a new key also needs a new feed file and URL, a new storage
name and a new global promise name, as the `2026-11` bootstrap has. The feed
of the previous key stays untouched for the installs that still read it.

## Marketplace card

Only `main` resolves against the pinned `branch`. `readme` and `preview` are
absolute `main` URLs, so the Marketplace card and README page follow the
current docs without moving the bootstrap pin. Marketplace reads
`manifest.json` from the default branch, so card text changes reach new
visitors on merge; an installed copy keeps its stored card until reinstalled.
