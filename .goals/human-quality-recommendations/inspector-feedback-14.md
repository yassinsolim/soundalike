# Inspector Feedback — Iteration 14

## Verdict: FAIL (goal) · Disambiguation fix: PASS · Commit hygiene: PASS (regression fixed) · Evaluator: READY TO RATE

Independent inspection of commit `f276651` ("fix(recs): [B] disambiguate
homonymous artists") against the immutable `goal.md` and `inspector-feedback-13.md`.
Nothing the Builder claimed was trusted: I recomputed every committed JSON
`content_sha256`, re-derived the served-lists and semantic-order hashes,
re-verified the Ed25519 state signature with `ssh-keygen`, re-checked the full
v13→v14 supersession chain, re-hashed the local (gitignored) identity + CLAP
assets, recomputed all six collision-audit counts from the 10.7 MB local audit,
reconstructed the actual before/after top-5 lists for both changed seeds,
swept all 60 seeds for cross-scene collapse and false merges/abstentions, read
the guard source to confirm no artist-specific branch, ran the full suite, and
scanned the diff for secrets.

**One sentence:** This is a clean, fully-signed, correctly-scoped fix that
removes the v13 "Nothing" drum&bass collision with a *generic* audio+style
identity guard (no name/benchmark special-case), re-issues the blinded pack as
signed v14 with a coherent supersession record and zero ratings, ships nothing
to production, drops the previously-flagged attribution trailer — but the goal
is still **FAIL** because the deciding human evidence (AC#3) does not yet exist.

---

## What I independently reproduced (nothing trusted)

**Commit hygiene — FIXED (was a 3× regression).** `git log -1 --format=%B
f276651` is subject-only: **no `Assisted-by:` / attribution trailer**. The
iteration-9/12/13 violation is resolved.

**Nothing root cause — VERIFIED and correctly diagnosed.** Seed DEV-SONIC-037
"You Wind Me Up — Nothing" (shoegaze, catalog Deezer artist **388063**). The
Last.fm graph mapped `normalize_text("nothing") → legacy graph artist_id
**12063**`, whose top neighbours are drum&bass/dubstep (ASC, Source Direct,
Submorphics, Cookie Monsta…). The source MusicBrainz MBID
`f4cd6526-…` ("US dark ambient producer Jason William Walton") was **discarded**
by the v13 build, so the source identity was merged into the Deezer name
cluster. The v14 guard recomputes a source-profile confidence of **0.6069 <
0.62** (low because the query's shoegaze audio does not agree with the
drum&bass neighbour centroids) and **abstains to exact production**. I confirmed
the crux: `"nothing"` has a **single** catalog Deezer artist id (`[388063]`),
so this was a graph-vs-catalog collision, not a catalog multi-ID homonym — which
is exactly why an audio-consistency guard (not an ID-merge) is the right fix.

**All six collision-audit counts — RE-DERIVED from the local audit.** Loaded
the gitignored `artist-identity-audit-v14.json` (file sha `845f10a8…` matches
the committed pointer; 10,672,189 B matches). `all_key_details` holds 18,315
normalized-name keys = 4,256 `unknown` + 13,921 `unique` + 138 `homonym`.

| collision_count | committed | my recompute |
|---|--:|--:|
| keys_with_multiple_deezer_artist_ids | 138 | **138** (= homonym key_type) |
| keys_with_multiple_source_mbids | 163 | **163** |
| keys_with_case_punctuation_or_transliteration_variants | 6254 | **6254** (raw_spelling_variants>1) |
| keys_with_multimodal_audio | 7643 | 7643 (audit field); naïve within-cos<0.70 = 7554* |
| homonym_keys_audio_unresolvable | 49 | **49** |
| unknown_id_keys | 4256 | **4256** (= unknown key_type) |

*My looser one-line proxy for "multimodal" gives 7554; the committed 7643
matches the local audit's own `keys_with_multimodal_audio` field exactly (the
production count also applies the `MIN_TRACKS_FOR_MULTIMODAL=4` gate). Not a
discrepancy in the artifact.

**Stable-ID / MBID handling — VERIFIED.** `rows_with_stable_primary_deezer_id
176,454` + `rows_without 96,399` = **272,853** (catalog rows). `stable_id_fraction
0.6467`. `source_mbid_direct_links: 0` with the honest policy string — "no
unverified Deezer-MBID link is claimed"; MBIDs are retained at normalized-name
level only. `track_ids_tobytes_sha256 a20632fc…` equals the pinned catalog
identity from v13/production.

**Every content hash & cross-binding — RECOMPUTED, all match (19/19).** Using
the module's own `content_hash` (sort_keys, compact separators, `content_sha256`
popped): the 5 committed artifacts, `protocol-v14`, `state`, and
`served-lists-v14` all self-hash exactly; `served_lists_sha256 5e7d852e…` =
`content_hash(served-lists)`; `semantic_order_sha256 7ecc7a45…` = recomputed
`semantic_order_hash(served-lists)`; state↔protocol, identity_audit,
diagnostics, and compact-asset bindings all close; `evaluator_sha256 b9cf6b12…`
= file hash of `benchmarks/human_eval_v14.html`.

**v14 signed supersession — VERIFIED.** `ssh-keygen -Y verify` →
**Good "soundalike-human-eval-v14" signature** over `state.json`; the
signature-metadata `state_sha256 72d987d5…` equals the raw file hash of
state.json. `supersedes_v13` reproduces exactly against the committed v13 files:
old protocol `35c106be…`, old lists `8c09b31e…`, old state `4f8af084…`, old
semantic `405a56b6…`, v13 diagnostics `2b76bf30…` — all recomputed and match.
`ratings_discarded 0`, `ratings_migrated 0`, `ratings_count_at_freeze 0`,
`RANKINGS_LOCKED`, reason "artist-identity collision correction". No private
key material is committed (only `.pub`, `allowed_signers`, `.sig`); the private
method-key and collector signer are gitignored and confirmed untracked.

**Generic guard, no artist/benchmark rule — VERIFIED in code + tests.**
`compute_source_profile_confidence = 0.55·audio_mean + 0.45·style_mean` over the
top-5 graph neighbours; abstain iff `source_conf < GUARD_SOURCE_PROFILE_MIN
(0.62)` **or** `query_resolve_conf < GUARD_CENTROID_MIN (0.60)`. The constants
are module-level and documented "not tuned to any artist"; grep finds **no**
hard-coded `"nothing"`, `DEV-SONIC-037`, `crowbar`, or scene string in the
ranking path (only docstring examples). `safety` flags:
`no_artist_specific_ranking_condition`, `no_benchmark_or_scene_boost`,
`cross_source_unlinked_ids_abstain` = true; `same_name_graph_clusters_mixed` =
false. A dedicated test — `test_nothing_regression_generic_evidence_no_name_branch`
— asserts the fix carries no name branch.

**Candidate-cluster isolation — VERIFIED.** `IdentityAsset.disambiguate` gates
on `GUARD_CENTROID_MIN 0.60` + `GUARD_CENTROID_MARGIN 0.05`; per-slot candidate
IDs require cosine ≥ `GUARD_CANDIDATE_MIN 0.60`; unlinked cross-source IDs
abstain rather than merge.

**Actual before/after lists — RECONSTRUCTED.** DEV-SONIC-037 v14: **both**
served lists are now the shoegaze production list (Sunny Day Real Estate /
Crowbar / Deftones / Bryan's Magic Tears / Catherine Wheel) — i.e. the
challenger abstained; the drum&bass collapse is gone. DEV-SONIC-057 (Miles
Davis "So What", jazz): the ID-ambiguous "Green Dolphin Street — Bill Evans"
(Deezer artist ids `[2060, 12404182]`) was dropped from the challenger and
back-filled with **Cannonball Adderley** (Kind-of-Blue personnel — highly
coherent). `semantic_diff`: 58/60 exact, 2 changed, 7 positions, mean overlap
0.978; the two changed seeds and 6 removed / 1 new unique result IDs reconcile
exactly.

**Other homonyms swept — no false merges or false abstentions.** I reconstructed
all 60 seeds. The v13-flagged legitimate homonyms are correctly **retained**:
Justice (French electronic) for MGMT, Galaxie 500 for my bloody valentine,
Chris Brown for Taio Cruz, Kirby for Rina Sawayama/Kali Uchis. The guard changed
behaviour on **exactly one** seed (037) and per-slot-filtered one (057);
`false_abstention_on_unambiguous_artist_count = 0`. No drum&bass artist from the
old collapse now appears in any inappropriate seed.

**Residual Crowbar — confirmed and honestly disclosed.** "Empty Room — Crowbar"
(sludge/heavy) sits at position 2 of the shoegaze production baseline the guard
abstained *to*; it is untouched by the guard and explicitly flagged in the
quality-security artifact ("heavy/sludge adjacent rather than shoegaze; retained
for blinded raters"). Nothing is blackgaze/heavy-adjacent, so it is borderline,
and it is a **baseline** trait, not a v14 regression. (Same category: Darkthrone
at #1 of the MBV baseline, and a "Paul Chambers"/"Paul Chamber" near-duplicate in
the Miles Davis baseline — all pre-existing baseline noise the human A/B exists to
surface.)

**Identity asset resources — VERIFIED / plausible.** Local
`artist-identity-v14.npz` re-hashed → `d47078a7…` (8,270,124 B) matches. Load
142.9 ms, incremental RSS 107 MB. Query latency **improved**: v14 mean 1754 ms
vs v13 2240 ms (−21.7%), p95 −15.8%. Prospective RSS 1.41 GB ≪ 48 GB dev RAM;
`vercel_fit_claimed: false`, guard is development-only, not wired. Production
`deepvibe_index_v5.npz` re-hashed `f3ed57af…` — **unchanged**.

**Preview coverage — VERIFIED.** Ranked positions **600/600** resolvable
(fraction 1.0), 472/472 unique results available, **59/60** seeds (the lone
`646715112` is the same carried no-preview seed), 1 new id (`3105354`, probed
`available`/200) and 6 removed ids. A concurrent throttled full-probe
(193 positions) is correctly **rejected** as coverage evidence with no ranking
change.

**Evaluator compatibility — VERIFIED, additive.** `human_aggregate_v10` adds
`_verify_v14_state` and extends the schema allowlist to `{10,13,14}`; the 10/11
and 13 paths are untouched. The v14 verifier binds content hashes, RANKINGS_LOCKED,
zero ratings across protocol/lists/state, served/semantic/private-key bindings,
the supersedes_v13 provenance (non-zero changed-seed count, zero
discarded/migrated), and an `ssh-keygen` state-signature check — a **stronger**
gate than v13, behind the existing ≥3-independent-signed-rater requirement.

**Quality gate — PASS (reproduced).** `.\.venv\Scripts\python.exe -m pytest
tests\ -q` → **729 passed in 37.5 s** (matches the claimed 729; +195 over v13's
534, from the three new v14 test modules with extensive real *and* synthetic
homonym coverage). Build / `compileall` / `pip_audit` are Builder-reported and
were **not** independently re-run here (long/network); the declared test gate
reproduces.

**Security — CLEAN.** The only "PRIVATE KEY" strings in the diff are negative
test assertions and a scanner pattern list; committed `.pub`/`.sig` are public.
No tokens/keys.

**Production unchanged — VERIFIED.** Changed set = `.gitattributes`, the v14
ML/eval modules + their tests, `human_aggregate_v10` (additive), README,
CASE_STUDY, the v14 evaluator HTML, and the v14 `.goals` protocol/artifacts.
**No** webapp, API, production recommender, `deepvibe_index_v5.npz`, index
manifest, or deployment asset touched. All report flags
`production_changed/deployed/ac3_claimed/commercial_final_opened = false`;
`production_version_retained: 2026.07.11-dual-sonic64`.

---

## Acceptance Criteria Check

- [x] **AC#1 — Frozen suite / scenes / actual lists.** MET; v14 re-issues a
  signed 60-seed / 13-scene pack recording real lists; v10/v11/v13 packs preserved.
- [x] **AC#2 — ≥3 materially different approaches + negatives.** MET (carried);
  three predeclared CLAP variants, `pure_clap` honestly rejected; v14 adds a
  generic identity guard on top.
- [ ] **AC#3 — ≥20% gain / no scene <−10% / ≥80% coherent top-5 / no junk.**
  **NOT MET (dispositive).** `ratings_count_at_freeze 0`, no `sonic_human`
  report, `ac3_claimed false`. Sole substantive blocker.
- [x] **AC#4 — External validation, not same-artist-only.** MET; MusicBrainz
  MBID / Deezer-affinity / Last.fm used as proxy safety and now for source
  identity, explicitly non-deciding (`proxy_evidence_is_deciding false`).
- [~] **AC#5 — Wired into desktop + hosted, live-verified.** N/A this iteration
  (deployment prohibited; nothing wired).
- [x] **AC#6 — Resources measured, no silent fallback.** MET; identity asset
  size/load/RSS/latency measured; Vercel fit explicitly not claimed.
- [x] **AC#7 — Regression tests + full suite + docs.** MET; 729 green; README +
  CASE_STUDY §15 document the collision root cause, the generic guard, the
  all-key audit, and reproduction.

---

## Quality Gate
- Command: `.\.venv\Scripts\python.exe -m pytest tests\ -q`
- Result: **PASS** — independently reproduced **729 passed** (~37.5 s).
- Note: the gate confirms code/report correctness only; it does not adjudicate AC#3.

---

## Issues Found (ranked)

1. **[BLOCKER — expected] AC#3 has no human evidence.** `ratings_count == 0`;
   no `sonic_human` report. Correctly not claimed. The only substantive reason
   the goal is not yet achievable.
2. **[RIGOR — transparency caveat] Guard threshold sits close to the fixed seed.**
   `GUARD_SOURCE_PROFILE_MIN = 0.62` vs Nothing's `0.6069`. The mechanism is
   genuinely generic (a real shoegaze-vs-drum&bass audio mismatch drives the low
   score) and a test forbids a name branch, but the committed diagnostics do not
   publish the per-seed `source_profile_confidence` distribution for the ~36
   challenger-active seeds. Publish that distribution so the threshold's
   margin/generalisation is auditable (mirrors the v13 geometry-gate note).
3. **[MINOR — conservative side-effect] Per-slot filter can drop a coherent
   result on ID ambiguity.** DEV-SONIC-057 lost a scene-appropriate "Bill Evans"
   purely because its Deezer artist id was ambiguous; back-fill (Cannonball
   Adderley) is equally coherent, so net-neutral/positive — worth noting for raters.
4. **[MINOR — A/B signal] Full abstention makes both DEV-SONIC-037 lists
   identical**, so that seed yields no discriminating A/B signal (expected
   consequence of abstaining to production).
5. **[MINOR — definitional] `normalized_catalog_keys_audited 18,257` (catalog
   keys) vs `all_key_details` 18,315 normalized names** (includes ~58 source-only
   names). Consistent, but the two counts could be labelled to avoid confusion.

## Credit where due
The root-cause diagnosis is precise, the fix is a properly generic guard rather
than a special-case, every hash/signature/binding/asset I checked is internally
consistent and reproducible to the byte, the supersession discards zero ratings,
the aggregator gains a stronger (not weaker) v14 binding, production is untouched,
and the previously-flagged attribution trailer is gone. The homonym test suite
(real + synthetic + transliterated + abstention) is thorough.

---

## Evaluator readiness & minimum next action

**READY TO RATE.** The v14 pack is frozen, hash-bound, state-signed
(`ssh-keygen` Good), `RANKINGS_LOCKED`, ratings 0, preview coverage 600/600
positions, the identity collision removed, and the aggregator verifies and
consumes schema-14 studies behind a ≥3-independent-signed-rater gate. The
supersession cleanly retires the v13 pack with zero discarded ratings.

**Minimum next action to move toward PASS:**
1. Recruit ≥3 (ideally 5) independent raters; run the frozen v14 blinded A/B;
   collect collector-Ed25519-signed exports; aggregate into a `sonic_human` report.
2. Test AC#3 on that human evidence with margin (≥20% primary-score gain, ≥80%
   coherent top-5, CI excluding zero, no scene regressing >10%, no junk/version).
3. Optionally publish the per-seed source-profile-confidence distribution to make
   the 0.62 guard threshold fully auditable.

## What Must Be Fixed (to reach PASS)
- Collect and aggregate **real** signed human ratings and satisfy **AC#3** on
  them; until then the goal remains **FAIL** by design. Nothing else in this
  iteration blocks progress.
