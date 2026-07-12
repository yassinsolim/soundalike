# Inspector Feedback — Iteration 11

## Verdict: FAIL (goal) · Evaluator readiness: PASS

Independent inspection of commit `ef5fcaa` ("fix(eval): [B] resolve fresh
previews for blind ratings") against the immutable `goal.md` and
`inspector-feedback-10.md`. Nothing the Builder claimed was trusted: I recomputed
every v10/v11 content hash and ranking-order hash, re-verified the Ed25519
erratum seal with `ssh-keygen -Y verify`, re-derived the coverage denominators
from the frozen pack, **live-sampled the production endpoint and the Deezer
source API**, drove the loopback evaluator **end-to-end in Chrome** (load →
render → blinded A/B → shared-result dedup → real playback → no-preview 404 →
extra-param rejection), inspected the live network request headers, and ran the
full test suite.

**Two separate questions, two answers:**

1. **Evaluator readiness — PASS.** Iteration 11 directly and honestly fixes the
   sole readiness caveat from iter-10 (preview coverage was 25% of results / 37%
   of seeds). It now resolves **stable Deezer IDs for 480/480 results and 60/60
   seeds** and resolves preview URLs **fresh on demand** (so coverage cannot
   silently rot mid-session as signed URLs would). Independently re-verified live
   coverage: **457/480 results (95.2%)**, **59/60 seeds (98.3%)**, **558/600
   ranked positions (93.0%)** — above the 90% target. The one truly unavailable
   seed (Deezer `646715112`) genuinely has no preview at the Deezer source
   itself; it is handled with a visible message + fallback links. The evaluator
   is now genuinely ready for ≥3 human raters.

2. **Recommender quality (AC#3) — NOT demonstrated → FAIL.** Zero human ratings
   exist (`ratings_count_at_freeze=0`, no `sonic_human` report). No model, index,
   embedding, ranked order, or deployed version changed. The Builder explicitly
   does **not** claim AC#3 and fails closed. This is the correct honest state and
   the sole blocking reason for the goal FAIL.

The goal remains FAIL because the deciding human-aligned evidence still does not
exist — not because anything is dishonest or broken. Verified evaluation
infrastructure is not the same as a verified recommender.

---

## What I independently reproduced (nothing trusted)

**Content hashes & trusted pins — ALL MATCH.** Recomputed `content_sha256` for
`protocol-v11.json` (`24ab0485…`), `served-lists-v11.json` (`e22979e8…`),
`audio-access-erratum-v11.json` (`c40a6935…`) and `human-eval-preview-audit-v11.
json` (`7afd8907…`); every stored hash matches, and the v11 protocol/lists/
erratum equal the `TRUSTED_V11_*` constants pinned in `human_eval_v11.py`. The v10
predecessor files self-verify and equal `TRUSTED_V10_PROTOCOL/LISTS`.

**Old/new semantic list & order parity — VERIFIED by recomputation.**
`ranking_order_hash(v10) == ranking_order_hash(v11) == 39428533a7e33543…`
(recomputed independently). A deep structural diff over all 60 seeds shows the
v11 pack differs from v10 **only** by: added `deezer_track_id` per row; removed
`preview_url`/`preview_available`; added top-level `audio_access`,
`source_catalog_index`, `predecessor_served_lists_sha256`. Every `seed_id`,
`result_id`, `list_id`, position, scene, title and artist is identical
(0 non-metadata field diffs across 540 rows). The erratum's `old_*` hashes match
the real v10 files and its `new_*` hashes match the real v11 files; the protocol
is hash-bound to the lists and to the erratum.

**Stable Deezer ID mapping — VERIFIED.** `deezer_track_id == track_id` for all
540 rows (the ID is a copy of the already-frozen catalog ID, not a fresh
lookup), so IDs cannot drift. The aggregator additionally enforces
`deezer_track_id == old track_id` per row, preventing an ID swap under a valid
opaque order.

**Preview coverage — RE-DERIVED + LIVE-SAMPLED.** Denominators recomputed from
the frozen pack: 480 unique results, 60 seeds, 600 ranked positions, 533 unique
Deezer IDs — all match the audit. Audit arithmetic is internally consistent
(457+23+0=480; 59+1+0=60; 558+42+0=600; 0.93). Live-sampled 6 audit-"available"
IDs on the production endpoint → all HTTP 200 with `cdnt-preview.dzcdn.net`
previews; the audit's single no-preview seed `646715112` → **404 at both the
production endpoint and the Deezer source API** (confirmed genuinely
preview-less). Notably, 6 of the audit's freeze-time "no_preview" **result** IDs
now return previews at the Deezer source — i.e. the 457/480 snapshot is a
conservative floor and the fresh-on-demand design captures whatever is available
at rating time. This is expected Deezer preview volatility, not fabrication.

**Signed erratum — VERIFIED "Good".** `ssh-keygen -Y verify` returns
`Good "soundalike-human-eval-audio-erratum"` for the detached signature over the
erratum. All four `erratum-signature-metadata.json` byte hashes match; the
`erratum-allowed-signers` file equals `TRUSTED_ERRATUM_ALLOWED_SIGNERS_FILE`.
*Nuance (unchanged from prior seals):* the erratum key is generated ephemerally
at freeze and its public half committed alongside — this proves freeze integrity
and provenance, not independent third-party authenticity; the aggregator layers
the collector Ed25519 approval on top for the human evidence gate.

**Blinding / no leakage — GOOD.** The v11 lists carry only opaque
`^[LT]-[a-f0-9]{20}$` IDs (1200 checked, 0 malformed); a token scan finds no
method/role/score identity (`method` occurs only in the count key
`results_per_method`; `score` only inside a track title/artist string). No
private method-role or signer keys are committed anywhere in the v11 directory;
no `dzcdn.net`/`hdnea`/`PRIVATE KEY` strings are committed.

**HMAC / aggregator compatibility — VERIFIED end-to-end.** The new
`_verify_audio_access_erratum` gate (reached only when the key's bound v10 list
hash ≠ the served list hash) pins the v11 erratum/protocol/lists/key hashes,
re-verifies the original v10 `state.sig` against hard-coded predecessor file
hashes, compares every old/new title/artist/track-ID/opaque-ID/position, and
re-verifies the erratum signature. The end-to-end test
`test_new_pack_remains_aggregator_compatible_when_private_key_is_present`
**passed (not skipped)** — a real HMAC-signed, collector-Ed25519-approved export
is accepted against the v11 pack with `valid_export_count==1`. Anti-proxy
(`human_listener` / `standalone_local_evaluator`), collector approval, and
constant-time HMAC checks are unchanged.

**CSP / referrer / fetch privacy — VERIFIED live in Chrome.** Page CSP is
`default-src 'none'; base-uri 'none'; form-action 'none'; frame-src 'none';
script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; media-src
https://*.dzcdn.net; connect-src 'self' https://soundalike.yassin.app`, plus
`<meta name="referrer" content="no-referrer">`. The live `/api/preview?id=…`
request carried **only** the numeric ID, an **empty `referer`**, no cookies/
credentials (`credentials:'omit'`), and `cache:'no-store'`. Autosave `state`
holds ratings/IDs only — no preview URL is persisted (verified in code and via
the audit's `local_storage_contained_signed_preview_url:false`). An extra
parameter (`id=1&rating=9`) is rejected **400 bad id**, so a rating value can
never be smuggled to the endpoint. Only a benign a11y console hint; no CSP
violations, no analytics, no third-party requests.

**Local/prod endpoint security — VERIFIED.** The loopback server validates the
`Host` header (non-loopback host → 421), accepts a single positive numeric `id`
only, sets `no-store`/`no-referrer`/`nosniff`/`DENY`/`frame-ancestors 'none'`/
`same-origin` CORP, validates the Deezer HTTPS `dzcdn.net` origin, and fails
closed. The production `preview.py` was likewise hardened (strict single-numeric-
id parse, `no-store`, `Referrer-Policy: no-referrer`, `nosniff`, `cover` removed,
`dzcdn.net` origin check).

**Live Chrome playback — VERIFIED.** Loaded the bundled locked study →
"Locked files validated. Autosave is active." → Seed 1/60 rendered with two
blinded lists, MIREX 3-class + optional 0–10 + junk/version per result, whole-
list coherence + top-3-unrelated, and the shared-result dedup banner. Clicked
"Load legal 30s preview": audio element played (scrubber 3.6s of a 29.989s clip),
status "Fresh preview loaded. The signed CDN URL is held only in page memory.",
media served from `cdnt-preview.dzcdn.net` (HTTP 206). The known no-preview ID
returned 404 → visible fallback path.

**Quality gate — PASS.** `.\.venv\Scripts\python.exe -m pytest tests\ -q` →
**520 passed (~23 s)** (iter-10 was 511; +9 v11 tests: parity, stable-ID
coverage, erratum signature/blinding, network privacy, loopback host/param
rejection, production-handler id validation, audit-count, committed-audit
assertion, aggregator compatibility).

**Commit hygiene — CLEAN.** The commit body is exactly `fix(eval): [B] resolve
fresh previews for blind ratings`; no `Assisted-by`/`Co-authored-by`/`Generated-
by` trailer. Working tree is clean apart from the orchestrator's `status.json`
flip.

**Production recommender — UNCHANGED.** No recommender/model/index/embedding file
is touched by `ef5fcaa` (only the `webapp/api/preview.py` audio endpoint). Live
site returns HTTP 200; deployed recommender version retained.

---

## Is the evaluator ready for ≥3 human raters? — YES (readiness PASS)

The iter-10 blocker (sparse, recognition-biased "sonic" ratings from ~25% preview
coverage) is resolved and independently confirmed: 95% of results, 98% of seeds,
and 93% of ranked positions resolve live, with fresh-on-demand resolution +
one automatic retry so coverage does not degrade across a 90–150 minute session.
The protocol and on-page instructions now require ≥3 raters (ideally 5), and the
aggregator computes a chance-corrected pairwise kappa, so one-rater bias is
guarded procedurally.

**Residual (non-blocking) readiness notes:**
- 1/60 seeds (`646715112`) has no Deezer preview at source; that seed's sonic
  rating must use a fallback or be skipped.
- Preview availability is transient; the exact 457 will drift run-to-run (by
  design), so treat the audit as a floor, not a fixed guarantee.
- The hardened `preview.py` is **staged, not deployed**: the live endpoint still
  returns `cover` with `Cache-Control: public, max-age=600` and no
  `Referrer-Policy`. The deployed endpoint still functions (CORS `*`) and the
  evaluator only ever sends the ID, so this does not block collection — but the
  extra privacy hardening only takes effect once deployed.

---

## Acceptance Criteria Check

- [x] **AC#1 — Frozen suite / scenes / actual lists.** MET and preserved. The v10
  60-seed / 13-scene blinded pack is byte-for-byte intact; v11 adds stable Deezer
  IDs with proven order parity.
- [x] **AC#2 — ≥3 materially different approaches + negatives.** MET across the
  program; unchanged this iteration.
- [ ] **AC#3 — ≥20% gain / no scene <−10% / ≥80% coherent top-5 / no junk.**
  **NOT MET (deciding).** Zero human ratings; no `sonic_human` report. Not
  claimed. Sole blocking reason for FAIL.
- [x] **AC#4 — External validation, not same-artist-only.** MET (MTAT human-sonic
  calibration from iter-10, calibration-only, no win claimed).
- [~] **AC#5 — Wired into desktop + hosted, live-verified.** N/A for the
  recommender (nothing shipped). Open item carried from iter-10: the version/
  preview-endpoint changes touch production source but are staged, not deployed.
- [x] **AC#6 — Resources measured, no silent fallback.** MET; the evaluator fails
  closed and surfaces every no-preview/failure visibly.
- [x] **AC#7 — Regression tests + full suite + docs.** MET: 520 tests green;
  README + CASE_STUDY §13 honestly document the erratum, coverage, limitations,
  reproduction, and that AC#3 remains unclaimed.

---

## Quality Gate
- Command: `.\.venv\Scripts\python.exe -m pytest tests\ -q`
- Result: **PASS** — independently reproduced **520 passed** (~23 s).
- Note: the gate confirms code correctness only; it does not adjudicate AC#3.

---

## Issues Found (ranked)
1. **[BLOCKER — expected] AC#3 has no human evidence.** `ratings_count_at_freeze=
   0`; no `sonic_human` report. Correctly not claimed. This is the only reason for
   the goal FAIL.
2. **[READINESS — minor] Hardened `preview.py` is staged, not deployed.** The live
   endpoint still returns `cover` and caches with `max-age=600`. Deploy it (and
   live-verify per AC#5) or keep clearly labelled as staged.
3. **[LIMITATION] One seed has no source preview; preview availability is
   volatile.** Expected and handled by fallbacks + fresh-on-demand; document so
   raters know to skip/fallback rather than guess.
4. **[PROCESS] One-rater bias.** Need ≥3 (ideally 5) raters with overlapping full
   sessions; report inter-rater kappa alongside the deltas.

## Credit where due
This is a disciplined, honest readiness fix. Instead of re-freezing a larger set
of expiring signed URLs, the Builder added a separately signed metadata-only
erratum that provably preserves every ranked identity and order, resolves audio
fresh on demand with strong privacy (only a numeric ID leaves the page), and
proved 95%/98%/93% live coverage — all independently reproduced, with a browser-
verified playback path, a defensively over-engineered aggregator binding, 520
green tests, no leaked keys, and no attribution trailer. The recommender is
untouched and AC#3 is left honestly unclaimed. The evaluator is now ready to earn
AC#3; the remaining gap is exactly what it should be — **actual human ratings do
not yet exist.**

---

## What Must Be Fixed (to reach PASS)
1. **Collect real human ratings.** Have the user + **≥3 raters** (ideally 5)
   complete the 60-seed study; run the collector-signed exports through
   `human_aggregate_v10` to produce a `sonic_human` report.
2. **Meet AC#3 on that human evidence** with a clear margin: ≥20% nDCG@5 gain,
   ≥80% coherent top-5, CI excluding zero, no scene regressing >10%, and no junk/
   duplicate/version items — only then open a fresh FINAL / propose deployment.
3. **Report inter-rater kappa** alongside the deltas to bound one-rater bias.
4. **Decide the staged endpoint/filter status:** deploy and live-verify (AC#5),
   or keep explicitly labelled staged-not-deployed.
