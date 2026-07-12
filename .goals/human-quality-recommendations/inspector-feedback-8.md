# Inspector Feedback — Iteration 8

## Verdict: FAIL

Independent inspection of commit `b6747a7` ("feat(recs): [B] gate collaborative
retrieval by confidence") against the immutable `goal.md` and
`inspector-feedback-7.md`. Every Builder claim was treated as untrusted and
re-derived from the signed `protocol-v8-gated-development-r2/r3` artifacts, the
v8 gated CV/direct/outcome JSON, the policy source code, the test suite, and
independent recomputation of the content-hash chain and both Ed25519 seals.

**Bottom line.** This is another disciplined, honest iteration. The Builder did
exactly the iter-7 to-do list: it reshaped the mechanism into a **pre-registered
3-parameter confidence-gated fallback** to production `dual_sonic`, moved the
deciding gold onto a **curated sonic-editorial axis** (Deezer excluded from
selection), cross-validated on opened DEV via nested + scene-held-out CV, kept
single-open discipline (`final_open_count=0`), left production untouched, **and
finally dropped the prohibited attribution trailer**. But the **DEV
preconditions failed hard** — the deciding sonic primary *regressed* **−16.0%**
(challenger 0.007717 vs baseline 0.009184), the scene floor is breached, and the
hosted-tier resource gate is unverified — so the Builder correctly refused to
open any FINAL or deploy. With `all_preconditions_passed=false`,
`final_open_count=0`, and nothing shipped, the immutable goal's deciding
criterion (AC#3) is **not met**. The Builder reports this honestly; the verdict
is FAIL because the goal is not achieved, not because the work is dishonest.

The direct sonic review did improve to **16/20 vs production 15/20**, but that is
a **single-seed** delta (only the `shoegaze` seed flips) and cannot rescue AC#3
(see the analysis below).

---

## What I independently reproduced (nothing trusted)

**3-parameter gate — exactly as pre-registered.**
- `catalog-gated-policy-v8.json` and `catalog_policy.py` expose exactly
  `tau`, `sigma`, `audio_weight` (`numeric_parameter_count=3`); selected
  `(tau=0.35, sigma=0.30, audio_weight=0.05)`. Rank `= G + audio_weight·A`;
  graph blend `0.5·Last.fm_G + 0.5·Music4All_G`; **on abstention the ordering is
  the exact current production `dual_sonic`** (verified: all abstained seeds have
  identical challenger/production lists).
- The gate fires only when both sources cover the seed with **≥5 shared
  neighbors** and fixed agreement `≥ tau`, **and** style/audio consistency
  `≥ sigma` (`min(mean(A top5), mean(S top5), min(S top3))`).

**Sonic-only selection — verified.**
- `catalog-gated-sonic-dev-cv-v8.json`: `final_policy_selection_source =
  "all credible sonic DEV inner CV only"`, `deezer_used_for_selection=false`,
  MusicBrainz style used for the **sigma gate only**. Deciding gold curated to
  **65 credible category_a_sonic** editorial/participant/named-critic track-level
  records; v7 multipositive (100) kept **supporting-only** (taste-affinity,
  "never normalized into deciding records"). This is the axis hygiene iter-6/7
  asked for.

**Nested / scene CV — recomputed; the primary REGRESSED.**
- Nested 5-fold aggregate (61 records): baseline `sonic_primary` **0.009184** →
  challenger **0.007717** = **relative_gain −16.0%** (absolute −0.00147),
  paired-bootstrap ci95 **[−0.0044, 0.0]**, `p_positive=0.0`; improved **0**,
  worsened **1**, unchanged **60**. `nested_5fold_hard_gate=false`.
- Scene-held-out: aggregate identical −16.0%. **Only `metal` has a non-zero
  primary** (baseline 0.1867 → challenger 0.1569, −16.0%); **all other 13 scenes
  are 0.0 for both methods**. `every_scene_above_minus_10=false`,
  `scene_held_out_hard_gate=false`.
- **The whole primary is decided by one pair.** `recall_at_10 = 0.0164 = 1/61`
  for both methods: baseline retrieves exactly **one** of 61 gold counterparts in
  the top-10 (a metal pair); the challenger keeps it in the pool but pushes its
  rank down, so nDCG/MRR fall −16% while 60/61 records stay pinned at zero. This
  is single-positive pair retrieval at the noise floor (see Q1).

**Firing / abstention — Music4All coverage is the bottleneck.**
- 25/61 fired, 36/61 abstained. Abstention reasons: **`missing_independent_source`
  31**, `fewer_than_five_shared_neighbors` 4, `consistency_below_sigma` 1.
- `catalog_policy.py:332` hard-requires **both** `lastfm` **and** `music4all`
  coverage; Music4All maps only ~5% of the 272k catalog, so **31/61 DEV seeds and
  ~15/20 direct seeds abstain purely for missing Music4All** — before any quality
  judgment (see Q2).

**Direct 16/20 vs 15/20 — recomputed from the ranked lists.**
- `catalog-gated-direct-validation-v8.json`: challenger **16**, production **15**,
  `gate_met=true` (required 16). Per-seed reproduction: **15/20 seeds abstained**
  (challenger ≡ production); of the 5 fired, 4 already passed under production, and
  **exactly one (`shoegaze`) flips fail→pass**. The entire net gain over
  production is **one seed**. Remaining fails (hyperpop_digicore, gorillaz, rnb,
  art_pop) all abstained and are identical to production. Lists content-hash
  bound (`lists_sha256 bc6cd10f…`).

**Resources / tier — fits hardware; tier still unverified but now fails closed.**
- Peak RSS **1.494 GB**, graph **12.16 MB**, load 7.96 s, warm p95 265 ms,
  `fallback_count=0`, `deterministic=true`.
- `catalog-vercel-tier-evidence-v8.json`: `vercel whoami`/`project inspect` and
  two REST calls all fail without credentials (403); `project_tier=unknown`,
  `tier_verified=false`. This time the resource gate **fails closed** ("no
  Hobby/Pro assumption is used") — an improvement over iter-7's assumed 2 GB —
  but `verified_hosted_tier_resource_gate.passed=false` remains an unmet
  precondition.

**Provenance / seals / tests — independently verified.**
- Both Ed25519 seals verify **`Good`** byte-exact (`ssh-keygen -Y verify`,
  namespace `soundalike-protocol`): r2 key `zP6Xc4In…`, r3 key `9+EYGGvT…`.
  r2/r3 `state.json` hashes match declared (`e33e3e8f…`, `84769a21…`); the
  bound `catalog_policy.py` hash matches (`f59ab745…`). r3 state:
  `phase=DEVELOPMENT_LOCKED`, `final_open_count=0`, `fresh_final_blocked=true`,
  `deployment_blocked=true`. No `*final*` protocol directory exists.
- Quality gate independently reproduced: **479 passed in 19.06 s**; build clean,
  `pip_audit` 0 vulns, secret scan 0 private keys.
- Production **unchanged**: live `/api/stats` `version 2026.07.11-dual-sonic64`,
  `library 272853`; commit touches **no** `webapp/`, `integrations/`, deploy,
  `pyproject`, or `requirements`; no production module imports `catalog_*`.
  `goal.md` and `protocol-v5/6/7` unmodified. Junk/seed-title/one-per-artist
  dedup retained in the policy.
- **Attribution trailer FIXED.** Commit `b6747a7` body is clean
  (`feat(recs): [B] gate collaborative retrieval by confidence`) — no
  `Assisted-by:` line. Resolves the repeat iter-6/iter-7 integrity defect.

---

## Acceptance Criteria Check

- [x] **AC#1 — Frozen suite / scenes / held-out / sources / provenance.** MET.
  Signed r2/r3 protocols verify (hashes + both seals `Good`); curated
  sonic-editorial deciding gold; 20-seed difficult direct held-out with actual
  5-song ranked lists recorded and hash-bound.
- [x] **AC#2 — ≥3 materially different approaches + negatives.** MET and
  extended. Adds a genuinely new mechanism: a confidence-gated dual-source
  fallback that abstains to production. Prior negatives (audio-only,
  collaborative, masked-graph scorer, global graph-first policy) documented.
- [ ] **AC#3 — ≥20% clear gain / no scene < −10% / ≥80% top-5 / no junk.**
  **NOT MET.** The deciding sonic primary **regressed −16.0%** (ci95 excludes
  positive, `p_positive=0.0`); scene floor breached (metal −16%, 13 scenes flat
  at 0.0); `final_open_count=0` — there is no FINAL evidence at all. Direct
  review 16/20 clears the ≥80% sub-clause but is only **+1 seed** over
  production's 15/20 (~7%), not a clear ≥20% improvement. Honestly reported.
- [x] **AC#4 — External validation; not same-artist-only.** MET as supporting.
  Deezer excluded from selection (`deezer_used_for_selection=false`); overlaps
  disclosed (Music4All↔Deezer 82.4%, Last.fm↔Deezer 36.8%); id-isolation holds.
- [x] **AC#5 — Wired into canonical + hosted, live-verified.** Correctly N/A for
  a FAIL: production unchanged, live site confirmed on the prior version, no
  push / manifest change / asset upload. No failed method shipped.
- [~] **AC#6 — Resources measured, fits limits, no silent fallback.** PARTIAL.
  Resources measured and fit the hardware; zero fallbacks; deterministic. But the
  **hosted-tier precondition is unverified** (`project_tier=unknown`) and
  correctly **fails closed** — better than iter-7's assumption, still unmet.
- [x] **AC#7 — Regression tests + full suite + docs.** MET. 479 tests pass;
  build/audit/secret-scan clean; README + CASE_STUDY updated;
  **attribution trailer removed** (iter-6/7 defect resolved).

---

## Quality Gate
- Command: `.\.venv\Scripts\python.exe -m pytest tests\ -q`
- Result: **PASS** — independently reproduced **479 passed in 19.06 s**
  (build + `pip_audit` + secret scan clean per `catalog-gated-quality-security-v8.json`).
- Note: the gate confirms code correctness only; it does not adjudicate AC#3,
  which fails on the DEV preconditions (primary −16%, scene floor, no FINAL).

---

## Issues Found (ranked)
1. **[BLOCKER] AC#3 unmet; goal not achieved.** Deciding sonic primary regressed
   −16.0%; scene floor breached; no FINAL opened. Direct 16 vs 15 is a single
   seed, not a clear ≥20% gain. Nothing shipped. Honestly reported.
2. **[STRUCTURAL] Mandatory Music4All AND-term cripples the gate.** Requiring
   both sources (code line 332) forces **31/61** abstentions for
   `missing_independent_source` alone, and Music4All is the wrong anchor for a
   sonic gate (82.4% Deezer overlap). See Q2.
3. **[POWER] The deciding primary is underpowered by construction.** 60/61
   records score 0 for both methods; the −16% is one metal pair's rank shift.
   Single-positive pair retrieval cannot measure "ranked-list quality." See Q1/Q3.
4. **[DEPLOY-CONFIRM] Hosted tier still unverified** (`project_tier=unknown`).
   Now fails closed (good), but must be confirmed before proposing the graph for
   serving.

## Credit where due
Did the entire iter-7 to-do list, verifiably: reshaped into a pre-registered
**3-param confidence-gated fallback**, curated a **sonic-only** deciding gold with
Deezer excluded from selection, ran nested + scene-held-out CV on opened data,
kept single-open discipline (`final_open_count=0`), left production untouched,
measured resources and **failed the tier gate closed** instead of assuming, kept
459→**479 green tests**, verified both seals — and **finally removed the
prohibited attribution trailer**. The remaining gap is scientific (wrong gate
coverage + an underpowered deciding metric), not integrity.

---

## Answers to the orchestrator's three questions

### 1. Is exact single-pair top-10 retrieval on 61 pairs an adequately powered primary for "human-aligned ranked-list quality" now that direct-list coherence improved — without relaxing the goal post-hoc?

**No, it is not adequately powered — and saying so is NOT relaxing the goal; it
is the opposite.** The evidence is decisive: **60 of 61** records score exactly
0 on nDCG@10/MRR@10/Recall@10 for **both** baseline and challenger, so the
primary has effectively **one degree of freedom** (the single metal pair). A
metric where 98% of records cannot move regardless of method cannot detect
ranked-list quality; the −16% is entirely one pair's rank shift (~8→~11). This is
the identical noise-floor pathology inspectors flagged in iterations 3, 4, and 5.

Critically, the immutable goal **does not mandate single-pair retrieval** as the
primary. AC#1 requires "actual top-ranked recommendations, not only scalar
embedding metrics"; AC#3's deciding evidence "must include direct inspection of
actual ranked song lists and scene/style coherence"; and the scope boundary
explicitly forbids "Claiming perfection from proxy metrics … or coverage."
Single-positive pair retrieval is itself a proxy the goal already disfavors.
**Replacing it with a properly powered primary that grades actual served lists
honors the goal; it does not relax it.** What *would* be post-hoc relaxation is
lowering the +20% threshold, or declaring the +1-seed direct result (16 vs 15) a
"clear improvement" — neither is licensed.

So the improved direct coherence does **not** rescue AC#3: (a) on every powered
metric that actually exists this iteration the +20% is unmet (indeed negative),
and (b) 16 vs 15 is a single-seed ~7% move, not a clear ≥20% gain, and no FINAL
was opened. The correct conclusion is to build the powered primary the goal
already demands and clear +20% on it — not to treat the underpowered proxy's
failure as evidence the goal is too strict.

### 2. Is requiring Music4All + Last.fm agreement structurally wrong (5% coverage, 31 missing-source abstentions)? Can a low-capacity Last.fm-confidence + song-level audio/style gate be developed on opened data while retaining independent Deezer supporting validation?

**Yes — the mandatory dual-source AND-term is structurally wrong, and the code
and data prove it.** `catalog_policy.py:332` hard-requires **both** Last.fm and
Music4All coverage; Music4All maps only ~5% of the catalog, so it abstains on
**31/61** DEV seeds and **~15/20** direct seeds for `missing_independent_source`
alone — before any quality judgment. And Music4All is the wrong anchor for a
*sonic* gate: its learned neighborhood overlaps Deezer **82.4%** (Last.fm only
36.8%), so a Music4All-anchored gate is strongly correlated with the
taste-affinity label, not the sonic target. Requiring the low-coverage,
Deezer-correlated source as a mandatory conjunct maximizes abstention while
minimizing sonic signal — the worst of both. (On the 25 seeds it did fire, the
primary still did not improve.)

**Yes — a Last.fm-confidence + song-level audio/style gate is developable on
already-opened data.** Concretely: drop the mandatory Music4All conjunct; gate on
(i) Last.fm neighborhood confidence/strength τ (broader coverage, the more
Deezer-independent source at 36.8%), (ii) a **song-level** audio+style
consistency σ (per-track sonic cosine + MusicBrainz style), and (iii) one audio
tie-break weight — still **≤3 params**. Keep Music4All as an *optional*
corroborator that strengthens confidence where present but never gates coverage.
This preserves the abstain-to-production floor protection (abstain ⇒ challenger ≡
baseline ⇒ ≥ −10%) while letting the gate fire on the majority of seeds it
currently cannot reach. Deezer stays strictly supporting (excluded from
selection, as this iteration already correctly does), id-isolation retained,
overlaps disclosed — all CV-able on the opened v6/v7 + graph inputs via the
existing nested + scene-held-out harness without opening FINAL.
**Caveat:** a wider gate only matters if the deciding primary is powered enough
to register it (Q1/Q3) — otherwise it is still adjudicated by one metal pair.

### 3. Recommend a rigorous primary that evaluates actual top-ranked lists across ≥50 seeds and ≥12 scenes, includes direct scene/style coherence, is reproducible and independently grounded, and does not merely substitute a proxy.

Replace single-positive pair retrieval with a **list-level, multi-graded, blind
sonic primary**:

1. **Unit = the actual served top-K list** (K=5–10) for each of **≥50 seeds
   across the ≥12 required scenes** (AC#1 set), not a single hidden counterpart —
   directly measuring "human-aligned ranked-list quality" as worded.
2. **Graded, multi-positive sonic gold per seed.** Pre-register a set of
   track-level "sounds-alike" positives at **≥2 relevance grades**, from credible
   *independent sonic* sources (editorial "if you like X" / participant / named
   critic — the `category_a_sonic` set already curated), junk/karaoke/slowed/
   tribute/seed-mashup deduped. Score each list with **graded nDCG@K** so partial
   credit exists and lists are not pinned at zero.
3. **Direct scene/style coherence as a co-primary, not a footnote.** Fold the
   blind top-5 coherence review (no unrelated-scene item in positions 1–3) INTO
   the primary as a weighted, scored term, scaled from 20 to **≥50 seeds** — the
   goal already names this as deciding evidence.
4. **Independent grounding preserved.** Deciding gold sonic and independent of the
   training graph (Last.fm-360K/Music4All) **and** of Deezer; Deezer/ListenBrainz/
   MusicBrainz reported as **supporting** only, never used for selection; retain
   id-isolation; disclose the 82.4%/36.8% overlaps.
5. **Adequate power + honest inference.** With graded multi-positive lists over
   ≥50 seeds, per-seed nDCG is non-degenerate, so paired-bootstrap CIs become
   informative (the current `p_positive=0.0`, `ci_high=0.0` is a degeneracy
   symptom). Require CV mean CI to exclude zero **and** ≥20% relative gain **and**
   every scene ≥ −10% on this powered primary **before** opening one fresh FINAL.
6. **Reproducibility.** Freeze the seed list, per-seed graded gold, dedup rules,
   and scorer in a signed protocol (as already done) with input hashes + Ed25519
   seal, so the primary is regenerable end-to-end.

This is not a softer proxy: it makes the primary the *actual served list* judged
against independent sonic ground truth plus direct scene coherence — exactly what
AC#1/AC#3 require — and it is strictly harder to game than single-pair retrieval.

---

## What Must Be Fixed (to reach PASS)
1. **Adopt the powered list-level sonic primary above** (≥50 seeds, ≥12 scenes,
   graded multi-positive nDCG@K + direct scene/style coherence as a co-primary).
   Do not treat the underpowered single-pair proxy's failure as goal-too-strict.
2. **Drop the mandatory Music4All conjunct;** re-anchor the ≤3-param gate on
   Last.fm confidence τ + song-level audio/style σ + one audio tie-break, with
   Music4All optional. Keep Deezer supporting-only; retain id-isolation.
3. **Clear all DEV gates on the powered primary** (CI excludes zero, ≥20%
   aggregate, every scene ≥ −10%, direct coherence ≥80% with a clear margin over
   production) **before** opening ONE fresh FINAL; claim the gain only on the
   sonic axis; no post-open ablation selection.
4. **Confirm the real hosted memory tier** (fail-closed is correct; verification
   is still required) before proposing the graph for serving.
5. Keep the clean commit hygiene — the attribution trailer removal must stick.
