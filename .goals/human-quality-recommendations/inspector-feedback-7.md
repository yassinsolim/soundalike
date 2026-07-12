# Inspector Feedback — Iteration 7

## Verdict: FAIL

Independent inspection of commit `56a06dd` ("feat(recs): [B] validate
graph-first hybrid ranking") against the immutable `goal.md` and
`inspector-feedback-6.md`. Every Builder claim was treated as untrusted and
re-derived from the signed `protocol-v8-development` artifacts, the v8 CV/direct
JSON, the compact graph/index assets, the policy source code, the test suite,
and independent recomputation of the content-hash chain and the Ed25519 seal.

**Bottom line.** This is the most disciplined iteration to date and it did
essentially everything iteration 6 asked for: it corrected the signed
provenance erratum, proved source independence, replaced the over-parameterized
11-feature scorer with a **pre-registered low-capacity 3-parameter** graph-first
policy, added **nested 5-fold + scene-held-out** cross-validation that caught
the exact FINAL failure mode *without* opening any FINAL, shrank the graph to a
**compact ~11 MB** asset (killing the iter-6 +921 MB RSS risk), and left
production untouched. But the **DEV preconditions failed** — aggregate DEV
composite gain **+13.2% (< 20%)**, scene-held-out floor breached in **4 scenes**
(only **4/17** folds pass), direct sonic review **13/20 (< 16/20)** — so the
Builder correctly refused to create or open a FINAL. With `final_open_count=0`,
`all_preconditions_passed=false`, and nothing shipped, the immutable goal's
deciding criterion (AC#3) is **not met**. The Builder self-reports this
honestly; the verdict is FAIL because the goal is not achieved, not because the
work is dishonest.

---

## What I independently reproduced (nothing trusted)

**Provenance correction + signatures — real and verifiable.**
- `development-protocol.json` contains a `corrected_provenance` block that
  records the erratum: frozen v7 declared evaluation source "ListenBrainz
  session-based similar recordings" but the actual per-record primary source is
  **"Deezer related artists" (100/100 records)**; ListenBrainz secondary on 38.
  This directly resolves iter-6 issue #4.
- The Ed25519 detached seal **verifies `Good`** (`ssh-keygen -Y verify`,
  namespace `soundalike-protocol`) when fed byte-exact via stdin redirection.
- Every hash in `signature-metadata.json` matches on disk: `state.json`
  (`d4a42afe…`), `signer.pub` (`7c320cd8…`), `allowed_signers` (`87fb55a9…`),
  `state.sig` (`b39756b5…`), `development-protocol.json` (`9cfce22c…`).
- All **6 declared input-asset SHA-256** match on disk (v7 benchmark
  `b2fbf249…`, item2vec-full `59a24b29…`, iter-6 graph `10cc0e84…`, the new
  compact full graph `59a50473…`, style `d8ef2158…`, MusicBrainz tag cache
  `aeb182d7…`).
- `goal.md`, `protocol-v5/6/7` were **not** modified in this commit.

**Low-capacity policy — exactly 3 params, no hidden per-scene/manual boosts.**
- `CatalogPolicy` in `src/soundalike/ml/catalog_policy.py` exposes exactly
  `audio_weight`, `style_weight`, `style_guard_min` (`numeric_parameter_count=3`).
  `score = G + audio_weight·A + style_weight·S`, with `G = 0.7·w + 0.3/log2(rank+1)`
  and `A = mean(sonic01, clap01, 1/(1+vibe))` using **fixed baked constants**
  (0.7/0.3, equal audio blend) that are not tuned degrees of freedom.
- I read the full ranker: it is **scene-agnostic** — there is no scene lookup,
  no per-scene branch, no manual boost, no genre override anywhere in the
  scoring, candidate, dedup, or guard paths. Dedup enforces one-per-artist +
  junk/seed-title/same-artist filters (matches the goal's dedup requirement).
- The `top3_guard` (`S ≥ style_guard_min`) is a **no-op** at the selected
  `style_guard_min = 0.0` (all S ≥ 0), so it introduces no hidden preference.
- Selection space is a small **fixed grid** (audio ∈ {.1,.2,.3} × style ∈
  {0,.2,.35} × guard ∈ {0,.15,.25}); selected `(0.3, 0.35, 0.0)`. The direct
  lists were generated with this same selected policy.

**Nested / scene-held-out CV — recomputed and honest.**
- Primary metric `composite = 0.80·nDCG@10 + 0.20·style_coherence@3`.
- Selected policy aggregate DEV: baseline composite `0.19461` → challenger
  `0.22035` = **+13.2%** (< 20% gate). nDCG gained a healthy `0.03477 → 0.04882`
  (+40%), but the composite's `0.20·style` term (baseline style ≈ 0.83, near
  ceiling, barely moving) **structurally dilutes** the nDCG gain to +13% — a
  metric-design fact that makes the 20% *composite* gate hard to reach.
- Scene-held-out (17 scenes), per-fold gate = {≥20% composite ∧ every scene
  ≥ −10% ∧ recall@1000 improves ∧ mrr/recall non-regression}. I tabulated
  every fold: **only 4/17 pass** (pop +30.1%, folk-country +26.6%, rock +23.8%,
  shoegaze +20.6%). **Four scenes breach the −10% floor**: reggae-dub-ska
  **−36.1%**, classical **−25.3%**, african **−13.6%**, other **−10.7%**.
  `all_reported_gates_pass=false` — correct.
- Nested inner-fold composite gains span only ~6–18.6% (all < 20%). CV data is
  **all previously-opened** v7 (100 records; v7 FINAL was burned in iter-6) plus
  v6 pairs (190); `no_unopened_final_labels_compared=true`,
  `final_open_count=0`, `fresh_final_created=false`. No fresh FINAL was touched.

**Direct 13/20 — recomputed from the ranked lists, not trusted.**
- `catalog-direct-judgments-v8.json`: **13 pass / 7 fail** (fails: direct-08,
  09, 11, 16, 17, 18, 20), matching `catalog-direct-validation-v8.json`
  (`effective_passes=13`, `required=16`, `gate_met=false`).
- The judgments are **content-hash-bound** to the actual lists: I recomputed the
  canonical content hash of `catalog-direct-lists-v8.json` (`1f0fcb07…`) and it
  matches the file's self-declared `content_sha256` **and** both the judgments'
  and validation's `lists_sha256`. (The raw-file hash differs only because the
  bound hash is canonical-JSON, verified in code: `_content_hash` strips
  `content_sha256` and dumps sorted/compact.)
- Each of the 20 records carries **two real 5-song ranked lists** (`catalog_policy`
  challenger + `current_production_dual_sonic` baseline) with title/artist and
  G/A/S rationales; `target_labels_used=false`. The failing reasons are honest
  and sonic (city-pop drift, Latin/French-R&B intrusions, generic-pop openers).

**Compact graph + resources — measured, fits, no silent fallback.**
- The runtime graph is now the compact `catalog-artist-graph-full-v8.npz`:
  18,258 artists × 96 neighbors, `float16` audio/weights + `int16` indices,
  ~11.4 MB uncompressed arrays — an order of magnitude below the iter-6 graph
  and **without** the +921 MB RSS. `graph_contract.only_full=true`,
  `masked_variants_present=[]` (masks are diagnostic-only, never shipped).
- Peak RSS **1.493 GB** < internal 1.611 GB cap; platform gate under a
  conservative **2 GB Hobby** ceiling with 654 MB headroom;
  `zero_errors`/`zero_fallbacks`/`deterministic_output=true`. The resident-target
  miss is **non-blocking and honest** ("core production index alone exceeds
  resident target"). `resource_gate_pass=true`.
- Caveat: the 2 GB limit is a *conservative assumption* — Vercel tier could not
  be credential-verified (`project_tier=unknown`). Fine for a non-ship
  preflight; must be confirmed against the real tier before serving.

**Tests / scans — independently reproduced.**
- I ran the quality gate myself: **459 passed in 19.25 s** (matches
  `catalog-quality-security-v8.json`). `pip_audit` 0 known vulns; secret scan 0
  private keys; build clean; `immutable_goal_modified=false`,
  `protocol_v7_modified=false`, v8 signature verified.

**Production unchanged — live-confirmed.**
- Live `GET /api/stats` returns `version 2026.07.11-dual-sonic64`,
  `library_size 272853` — the pre-existing production, unchanged.
- The commit touches **no** `webapp/`, `integrations/`, deploy, `pyproject`, or
  `requirements` files. No production/webapp module imports any `catalog_*` dev
  module (grep-verified). `iteration7_deployed=false`,
  `deployment_attempted=false`, `production_unchanged=true`. No silent shipping.

---

## Acceptance Criteria Check

- [x] **AC#1 — Frozen suite / scenes / held-out / sources / provenance.**
  MET and improved. `protocol-v8-development` verifies (5 file hashes + 6 input
  hashes + Ed25519 seal `Good`); the iter-6 provenance erratum is corrected in a
  fresh signed artifact; scene-held-out spans 17 scenes; 20-seed direct held-out
  covers the required difficult scenes; actual ranked lists recorded.
- [x] **AC#2 — ≥3 materially different approaches + negatives.** MET and
  extended. Adds a genuinely new mechanism: unmasked full Last.fm-360K artist
  graph + MusicBrainz style overlap under a pre-registered 3-param graph-first
  policy, cross-validated by nested + scene-held-out CV. Prior negatives
  (audio-only, collaborative, masked-graph 11-feature scorer) documented in
  CASE_STUDY.
- [ ] **AC#3 — ≥20% clear gain / no scene < −10% / ≥80% top-5 / no junk.**
  **NOT MET — DEV preconditions fail; no FINAL evidence exists.** Aggregate DEV
  composite **+13.2% (< 20%)**; scene-held-out floor breached in **4 scenes**
  (reggae −36%, classical −25%, african −14%, other −11%; only 4/17 folds pass);
  direct sonic **13/20 (65%) < 80%**. `final_open_count=0` — there is no
  ranked-FINAL improvement to claim at all. Honestly reported.
- [x] **AC#4 — External validation; not same-artist-only.** MET as supporting,
  with an important disclosure. `catalog-source-independence-v8.json` proves the
  training graph is **ID-isolated** from the Deezer labels (different dataset,
  operator, API, ID namespace; `id_isolation.passed=true`), which legitimizes
  unmasked direct edges. BUT the **Music4All↔Deezer learned-neighborhood overlap
  is 82.4%** (Last.fm↔Deezer 36.8%) — so Deezer-nDCG must stay **supporting**,
  not the deciding axis (see analysis).
- [x] **AC#5 — Wired into canonical + hosted, live-verified.** Correctly handled
  for a FAIL: production unchanged, live site confirmed working on the prior
  version, no repo push / no manifest change / no asset upload. No failed method
  shipped.
- [x] **AC#6 — Resources measured, fits limits, no silent fallback.** MET and
  improved. Compact ~11 MB graph, peak RSS 1.49 GB within a conservative 2 GB
  ceiling, zero fallbacks, deterministic. Resolves iter-6's +921 MB deploy-risk.
  Confirm the real Vercel tier before serving.
- [x] **AC#7 — Regression tests + full suite + docs.** MET, with a repeat
  commit-hygiene defect. 459 tests pass; build/audit/secret-scan clean;
  README + CASE_STUDY document the graph-first preflight honestly ("nested folds
  6.0–18.6%, scene-held-out 4/17, direct 13/20, no protocol-v8 FINAL created").
  **Defect:** commit `56a06dd` again carries a prohibited
  `Assisted-by: Claude:Sonnet-4.6` attribution trailer.

---

## Quality Gate
- Command: `.\.venv\Scripts\python.exe -m pytest tests\ -q`
- Result: **PASS** — independently reproduced **459 passed in 19.25 s**
  (build + pip_audit + secret scan clean per `catalog-quality-security-v8.json`).
- Note: the gate confirms code correctness only; it does not adjudicate AC#3,
  which fails on the DEV preconditions (composite +13.2%, 4/17 scenes, 13/20).

---

## Issues Found (ranked)
1. **[BLOCKER] AC#3 unmet; goal not achieved.** No FINAL opened; DEV composite
   gain +13.2% (< 20%), scene-held-out floor breached in 4 scenes (only 4/17
   pass), direct sonic 13/20 (< 16/20). Honestly reported; nothing shipped.
2. **[STRUCTURAL] A *global* graph-first policy cannot clear a per-scene floor.**
   The graph helps pop/rock/shoegaze/folk/jazz/hip-hop/electronic/r&b but
   regresses african/classical/reggae/other, because it deviates from baseline
   in **every** scene, including the ones it hurts. This is why a confidence-
   gated fallback (below) is the right next move.
3. **[AXIS] Deciding metric still drifts toward taste-affinity.** 82.4%
   Music4All↔Deezer neighborhood overlap means Deezer-graded nDCG rewards artist
   relatedness; the goal's deciding axis is **sonic** (direct 13/20). The
   composite's 0.20·style term also caps achievable composite gain (a +40% nDCG
   gain became +13% composite).
4. **[INTEGRITY — REPEAT] Prohibited attribution trailer** on commit `56a06dd`
   (`Assisted-by: Claude:Sonnet-4.6`) — identical to iter-6 defect #5, a direct
   violation of the goal's commit convention. Must stop.
5. **[DEPLOY-CONFIRM] 2 GB ceiling is assumed, not verified** (`project_tier=
   unknown`). Confirm the real hosted limit before proposing the graph for
   serving.

## Credit where due
Did the entire iter-6 to-do list, honestly and verifiably: corrected the signed
provenance erratum, proved source independence (ID-isolation), replaced the
over-parameterized scorer with a pre-registered **3-param** policy, built nested
**+ scene-held-out** CV that exposed the exact heterogeneous failure mode
*before* any FINAL, made the graph **compact** (killing the +921 MB risk), kept
single-open discipline (`final_open_count=0`), left production untouched, and
reported an honest FAIL with 459 green tests and a verified seal. The remaining
gap is scientific (a global policy is the wrong shape, and the deciding axis must
be sonic), not integrity — apart from the repeat attribution trailer.

---

## Answers to the orchestrator's three questions

### 1. Is a globally predeclared confidence-gated fallback to production the right next direction?
**Yes — and it is better-motivated than the global graph-first policy that just
failed.** Two independent reasons:
- **It structurally protects the per-scene floor.** Where the gate *abstains*
  and falls back to production `dual_sonic`, challenger ≡ baseline for those
  seeds, so per-scene relative change ≈ 0 ≥ −10%. The policy only deviates where
  it is confident — and the scene-held-out results show graph-first is strong in
  exactly the scenes where independent co-listening evidence is dense. A gate
  can keep the winning scenes' gains while zeroing the losing scenes'
  regressions (reggae/classical/african/other).
- **It is doubly motivated: it should also raise the deciding *sonic* score.**
  Production `dual_sonic` is a pure-sonic method; falling back to it on
  low-confidence seeds repairs precisely the recurring direct-review failures
  where the graph drifts sonically (city-pop direct-16, Latin direct-17/18,
  and the iter-5/6 hyperpop/digicore misses). So the same gate that fixes the
  nDCG floor should also push direct review past 16/20.

The trigger should be **independent Music4All + Last.fm agreement AND
audio/scene (style) consistency** — i.e., only override production when both
collaborative sources concur *and* the candidate is stylistically close to the
seed. That is a legitimate, ID-isolated confidence signal.

### 2. Can such a gate stay ≤3 parameters and be cross-validated without leakage?
**Parameter budget: feasible but tight.** A minimal admissible form spends the
3-param budget on the gate itself — e.g. (i) a Music4All∧Last.fm agreement/
strength threshold τ, (ii) a style-consistency threshold σ, (iii) one audio
tie-break weight — with the intra-mode blends (G's 0.7/0.3, A's equal mean)
frozen as constants. You cannot *also* keep free `audio_weight`+`style_weight`
and add gate thresholds without exceeding 3; the design must fold the current
weights into fixed constants or drop them.

**Leakage: yes in the strict ID sense, but with a hard caveat.** `id_isolation`
passes (no shared Deezer IDs enter training), so the same `catalog_cv.py` nested
5-fold + scene-held-out harness can select τ/σ on opened DEV without opening
FINAL. **However**, the 82.4% Music4All↔Deezer learned-neighborhood overlap
means a gate that fires on Music4All/Last.fm agreement is strongly correlated
with the Deezer label structure. Therefore CV-ing the gate against **Deezer-
graded nDCG would optimize the *supporting* axis** and inflate the apparent gain.
The gate must be **selected and gated on a SONIC gold** (blind soundalike sets,
junk-deduped) with Deezer/ListenBrainz kept explicitly as supporting evidence.
The composite's 0.20·style term should also be revisited — as measured, it
dilutes real nDCG movement so much that a ≥20% *composite* gate may be
near-unreachable even when the sonic axis genuinely improves.

### 3. What exact DEV acceptance should be required before consuming FINAL?
All of the following, on opened/DEV data, **before** creating a single fresh
unopened FINAL (and re-sealed into the protocol with frozen ≤3 params):
1. **Pre-register the gated fallback** with frozen thresholds/weights (≤3 numeric
   params total), the abstention rule, and the deciding SONIC gold — signed.
2. **Nested 5-fold CV:** selected gated policy ≥ **20%** relative gain vs
   production on the deciding (sonic) primary, with the CV mean CI excluding
   zero. *(This iteration: +13.2%, fails.)*
3. **Scene-held-out CV:** **every** held-out scene ≥ **−10%** relative **and**
   aggregate ≥ 20%. The fallback design is what should finally make this
   reachable. *(This iteration: 4 scenes < −10%, 4/17 folds pass, fails.)*
4. **Direct SONIC review ≥ 16/20** on DEV-adjacent held-out seeds, blind, with
   dedup (duplicates/karaoke/slowed/remix/seed-mashups excluded), specifically
   repairing city-pop, Latin, hyperpop/digicore, and the Pixies→trip-hop class.
   *(This iteration: 13/20, fails.)*
5. **Axis hygiene:** deciding gold is sonic; Deezer-nDCG held as supporting; the
   82% Music4All↔Deezer overlap disclosed; ID-isolation proof retained.
6. **Resource gate against the REAL hosted tier** (verify the Vercel plan; do not
   ship on the conservative-assumption path), zero silent fallbacks.
7. **Only then open ONE fresh unopened FINAL**; claim the ≥20% gain solely on the
   axis the gate addresses; **no** post-open ablation selection; single-open
   discipline; re-seal.
8. If — and only if — FINAL clears, wire consistently into desktop + hosted
   numpy path with parity tests, live-verify ≥10 diverse seeds (including
   previously poor ones), then update manifest/README/CASE_STUDY.

---

## What Must Be Fixed (to reach PASS)
1. **Reshape the mechanism into the confidence-gated fallback** above; a global
   graph-first policy cannot clear the per-scene floor.
2. **Keep ≤3 frozen params** and select the gate thresholds on a **sonic** DEV
   gold via the existing nested + scene-held-out CV — never on Deezer-nDCG.
3. **Clear all DEV acceptance gates (2–6 above) BEFORE creating any FINAL**;
   only then open ONE fresh FINAL and claim the gain on the sonic axis.
4. **Stop emitting the `Assisted-by:` attribution trailer** — it violates the
   commit convention for the second consecutive iteration.
5. **Confirm the real hosted memory tier** before proposing the graph for
   serving.
