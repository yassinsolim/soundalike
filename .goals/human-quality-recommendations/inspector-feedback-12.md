# Inspector Feedback — Iteration 12

## Verdict: FAIL (goal) · Method quality: PASS · Commit hygiene: FAIL

Independent inspection of commit `d87034d` ("feat(ml): [B] cross-validate human
audio similarity") against the immutable `goal.md` and `inspector-feedback-11.md`.
Nothing the Builder claimed was trusted. I recomputed the benchmark and report
`content_sha256`, independently rebuilt the artist co-occurrence graph and its
component count, re-derived OOF coverage and per-fold purge disjointness from the
manifests, **re-ran every paired statistic (triad bootstrap, artist-cluster
bootstrap, sign-flip, exact McNemar) from the committed OOF vectors using the
module's own functions**, re-applied the win gate, re-verified all 25 checkpoint
hashes, ran the full test suite, scanned the diff for secrets, and confirmed the
exact changed-file set. I also independently computed the CLAP-vs-incumbent
statistics the task asked about.

**One sentence:** This is the most methodologically rigorous iteration in the
program — a genuine leakage-purged nested cross-fit that honestly **rejects** the
learned projection and ships nothing — but the goal is still **FAIL** because the
deciding human commercial-list evidence (AC#3) does not exist, and the commit
carries a **prohibited `Assisted-by:` attribution trailer** that violates the
stated commit convention.

---

## What I independently reproduced (nothing trusted)

**Reconciliation — VERIFIED.** The v12 benchmark reconciles exactly **307 accepted
constraints, 611 distinct clips, 193 distinct artists**; the module hard-fails if
these differ (`EXPECTED_CONSTRAINTS`, `EXPECTED_CLIPS`). Benchmark
`content_sha256` recomputes to the stored value; report `content_sha256` likewise
matches.

**Artist-component purged fold construction — VERIFIED, no leakage.** I
independently recomputed the raw artist graph: **1 connected component** (matches
the stored `raw_artist_graph_connected_components: 1`), which is exactly why plain
component GroupKFold is impossible. The v12 answer — Louvain communities of the
shared-artist/shared-clip triad graph packed into 5 folds balancing row count,
confidence mass, and the 3 fixed vote strata, then per-fold purging of every
non-test row sharing any clip/artist with test — is sound. I verified from the
committed fold manifests: **0/5 folds have any test↔eligible-train clip or artist
overlap**, and `test + eligible_train + purged == all rows` for every fold. OOF
coverage is **exactly once per source row** for all 307 (recomputed).

**Nested selection isolation — VERIFIED.** Family + hyperparameter selection uses
**inner-OOF confidence-weighted accuracy only**; the inner folds are built from
the outer training rows and the evaluator predicts only on inner-validation rows.
The unit test `test_nested_tuner_evaluator_never_sees_outer_rows` asserts the
evaluator is never handed a forbidden outer-test row, and I confirmed the
`run_nested_cv` control flow never selects on outer test. Per-fold nested family
choices vary (`fma_distilled_linear`×3, `mlp_triplet`, `linear_triplet`) — a
plausible signature of genuine per-fold inner selection, not a fixed pick.

**All OOF accuracies (n=307 each) — REPRODUCED from committed vectors:**

| method | correct | acc | conf-wtd |
|---|---:|---:|---:|
| **clap (fixed)** | 204 | **66.4%** | 74.3% |
| mlp_triplet | 199 | 64.8% | 70.8% |
| fma_distilled_linear | 197 | 64.2% | 70.7% |
| orthogonal_linear_triplet | 190 | 61.9% | 69.5% |
| linear_triplet | 187 | 60.9% | 69.4% |
| **nested_selected_learned** | **186** | **60.6%** | 67.5% |
| fma_supcon | 185 | 60.3% | 66.1% |
| vibe_dsp | 180 | 58.6% | 64.8% |
| artist_supcon (incumbent) | 176 | 57.3% | 62.8% |
| smooth_linear | 176 | 57.3% | 64.0% |

**Rejection gate — VERIFIED correct.** The predeclared promotion-eligible model is
`nested_selected_learned` (+3.26 pt over incumbent). Re-running the stats:
triad-bootstrap CI **[-1.30, +7.82]** (lower < 0), artist-cluster CI **[0.00,
+6.62]** (lower not > 0), sign-flip **p=0.213**, exact McNemar **p=0.212** (31
challenger-only vs 21 baseline-only), 3/5 folds positive. Gate result: **not
passed**, failing 3 of 4 conditions. This matches the committed `win_gate` exactly.
The rejection is the correct, honest outcome.

**Vote-strata / CIs / tests — VERIFIED.** The vote-strength strata (`<0.25`,
`[0.25,0.50)`, `>=0.50`) are fixed and predeclared; deltas by stratum (+1.09 low,
+3.31 medium, +5.32 high) reproduce. All four paired tests reproduce to the stored
values.

**Unchanged catalog / evaluator / production — VERIFIED.** The commit touches only
the new ML module, its test, README, CASE_STUDY, two evidence artifacts, and 25
compact fold checkpoints. **No** production recommender, catalog-embedding,
query-transform, Last.fm gate, v10/v11 signed protocol/list, human evaluator, or
webapp file is modified. Report flags all `False`:
`final_all_data_projection_trained`, `catalog_embeddings_changed`,
`commercial_evaluator_changed`, `commercial_final_opened`,
`production_ranking_changed`. Working tree is clean except the orchestrator's
`status.json`.

**Preview reaudit — VERIFIED as evidence-only.** `human-eval-preview-reaudit-v12.
json` re-audits the **unchanged** v11 pack (`served_lists_sha256` = the v11 hash);
its own `content_sha256` recomputes correctly. Fresh coverage improved to
**480/480 results, 59/60 seeds, 600/600 ranked positions (100%)**, one genuinely
preview-less seed (`646715112`) handled by fallback. It is explicitly not a new
pack or ratings. Note the observed live endpoint is still `CORS: *`,
`Cache-Control: public, max-age=600` — i.e. the hardened `preview.py` from iter-11
remains **staged, not deployed** — and `chrome_playback` is
`"pending manual verification"` (Builder did not re-drive the browser; honest).

**Quality gate — PASS.** `.\.venv\Scripts\python.exe -m pytest tests\ -q` →
**526 passed (~23 s)** (iter-11 was 520; +6 v12 tests). The v12 tests are
substantive: connected-graph OOF-once + purge disjointness, fold determinism +
strata boundaries, nested-tuner outer-row isolation, full statistic/gate pass+fail
conditions, checkpoint schema, and a **fail-closed** test that rebinds the
committed report hash, asserts `win_gate.passed is False` and all-unchanged flags,
verifies all 25 checkpoint hashes, and checks the reaudit coverage/hash.

**Resources — measured, within budget.** RTX 5080; feature prep + nested training
~34.7 s + 136.6 s; peak GPU **71.3 MB allocated / 92.3 MB reserved**; 25
checkpoints **1.86 MB**; fixed cache 2.39 MB; FMA cache 0.73 MB. Trivially fits the
hardware.

**Security — CLEAN.** No secrets in the diff; all `password`/`secret`/`token`/
`dzcdn.net` matches are benign docs describing OAuth-PKCE / no-secrets-in-git
hygiene.

---

## CLAP 66.4% vs incumbent 57.3% — the requested deep dive

**Was CLAP genuinely predeclared and OOF-evaluated?** **Yes.** CLAP is a hard-coded
member of `FIXED_NAMES`, evaluated as a **frozen** LAION-CLAP embedding under pure
cosine odd-one-out. Because it is never fitted on any fold, its "OOF" is trivially
the full 307 rows with **zero CV leakage** — it receives exactly one prediction per
source row like every other arm, and the coverage assertion passed.

**Is it a defensible development hypothesis (not a claimed winner)?** **Yes on both
counts.** I estimated its paired statistics from the committed OOF vectors using the
module's own functions:

- delta **+9.12 pt** (204 vs 176; 73 CLAP-only-correct vs 45 incumbent-only)
- triad-bootstrap CI **[+2.28, +15.96]**, P(positive)=0.995
- artist-cluster bootstrap CI **[+4.63, +13.70]**
- paired sign-flip **p=0.012**, exact McNemar **p=0.013**
- fold deltas [+13.3, +19.7, −5.5, −4.3, +26.2] → 3/5 positive

**If the predeclared win gate were applied to CLAP, it would PASS all four
conditions.** So the signal is real and robust, not noise — CLAP is a *strong*
hypothesis worth pursuing.

**But the Builder is right not to promote it, and does not.** README/CASE_STUDY
explicitly label CLAP (and the MLP/FMA-distilled families) "useful hypotheses, not
post-hoc winners," and name the exact hazard: selecting the best-of-10 outer
comparator reuses the outer folds and p-hacks the result. Three correct reasons
block promotion here:

1. **Multiplicity / winner's curse.** The gate was predeclared for the single
   promotion-eligible model, not for the maximum over 10 arms. CLAP's per-method
   p≈0.01 is uncorrected; picking the argmax after seeing outer scores inflates it.
2. **Wrong court.** Per `goal.md`, MTAT is *supporting* evidence; the **deciding**
   evidence must be the human commercial ranked-list protocol. An MTAT win cannot
   ship anything.
3. **Not derivable from the shipped catalog.** CLAP cannot be produced from the
   existing 272,853 artist-SupCon vectors; it needs raw audio.

**One risk the docs under-state:** LAION-CLAP's pretraining corpus plausibly
overlaps MagnaTagATune (a ubiquitous public benchmark). If so, CLAP's 66.4% is
partly a *seen-data* advantage that may not transfer. This is not CV leakage (the
human labels are independent and, on the commercial protocol, do not yet exist) —
but it is exactly why CLAP must be re-tested on the **independently frozen,
still-unrated** commercial protocol before any promotion. Please state this overlap
caveat explicitly if CLAP is elevated.

**Would a compact CLAP catalogue index preserve test isolation?** **Yes, if
predeclared and frozen before any ratings.** Because `ratings_count == 0`, the
window is open. Isolation holds provided:
- The CLAP arm's ranked lists for the 60 frozen seeds are generated and folded into
  a **re-frozen, re-signed** blinded pack **before** any human rating is collected;
- CLAP is a frozen pretrained embedding (no fitting on seeds/labels → structurally
  leakage-proof w.r.t. the human labels);
- lists keep **opaque IDs** so raters stay blind to method identity;
- the **same** retrieval/rank/dedup/junk/version pipeline is reused, swapping only
  the embedding;
- the MTAT result is cited only as the motivating hypothesis, never as the verdict.

**Practical feasibility (with safeguards):**
- **Audio/API/legal/caching:** CLAP needs raw audio. Reuse the already-available,
  legal **Deezer 30 s previews** (30 s is sufficient for CLAP); embed offline on the
  RTX 5080 and persist **only the derived vectors**, not the audio. Any catalog row
  without an available preview must be handled **explicitly** (incumbent fallback or
  exclusion) — no silent quality fallback (AC#6).
- **Index size / PCA:** native CLAP is 512-D → 272,853×512×f32 ≈ **559 MB**, too
  large for the hosted compact path. PCA (whitening fit on catalog, **not** the
  seeds) to ~64–128 D + float16 → **~35–70 MB**, matching the current index
  envelope. Measure retrieval-quality loss from compression.
- **Vercel memory:** the 512-D f32 index will not fit alongside runtime+numpy
  (~1 GB budget); the PCA-f16 index will. Cold-start load memory and latency must be
  measured per AC#6.

Net: building a compact CLAP index for the commercial protocol is a **legitimate,
isolation-preserving next development step** — but it is *development*, and does not
change this iteration's verdict.

---

## Acceptance Criteria Check

- [x] **AC#1 — Frozen suite / scenes / actual lists.** MET and preserved; v10/v11
  pack byte-identical, reaudited only.
- [x] **AC#2 — ≥3 materially different approaches + negatives.** MET and extended;
  v12 adds a leakage-purged nested cross-fit over 5 learned families + 4 fixed
  comparators with an honest documented rejection.
- [ ] **AC#3 — ≥20% gain / no scene <−10% / ≥80% coherent top-5 / no junk.**
  **NOT MET (deciding).** Zero human commercial-list ratings exist; no
  `sonic_human` report. The promotion candidate is rejected on MTAT and nothing is
  claimed. Sole substantive blocking reason.
- [x] **AC#4 — External validation, not same-artist-only.** MET; MTAT human
  odd-one-out is independent human-behaviour evidence, used as hypothesis not
  verdict.
- [~] **AC#5 — Wired into desktop + hosted, live-verified.** N/A for the recommender
  (nothing shipped). Carried open item: hardened `preview.py` still staged, not
  deployed.
- [x] **AC#6 — Resources measured, no silent fallback.** MET; GPU/time/index/cache
  sizes recorded and within budget.
- [x] **AC#7 — Regression tests + full suite + docs.** MET; 526 green, README +
  CASE_STUDY §All-triad honestly document method, negative result, CLAP-as-
  hypothesis, and reproduction.

---

## Quality Gate
- Command: `.\.venv\Scripts\python.exe -m pytest tests\ -q`
- Result: **PASS** — independently reproduced **526 passed** (~23 s).
- Note: the gate confirms code/report correctness only; it does not adjudicate AC#3.

---

## Issues Found (ranked)

1. **[BLOCKER — expected] AC#3 has no human evidence.** `ratings_count == 0`; no
   `sonic_human` report. Correctly not claimed. The only substantive reason the goal
   is not yet achievable.
2. **[COMMIT-HYGIENE VIOLATION — must fix, regression] Prohibited attribution
   trailer.** Commit `d87034d` body ends with `Assisted-by: Claude:Sonnet-4.6`.
   `goal.md` states: "Do not add attribution trailers or generated-by taglines."
   This regressed (same defect flagged at iteration 9). Future Builder commits must
   omit it.
3. **[RIGOR — refinement] CLAP↔MTAT pretraining-overlap caveat is under-stated.**
   If CLAP is elevated to a formal hypothesis, explicitly note possible MTAT
   exposure in CLAP pretraining and rely on the independently frozen commercial
   protocol as the clean test.
4. **[CARRIED — minor] Hardened `preview.py` staged, not deployed.** Live endpoint
   still `CORS: *`, `max-age=600`, no `Referrer-Policy`. Deploy + live-verify
   (AC#5) or keep clearly labelled staged.

## Credit where due
This is exemplary negative-result engineering: a raw artist graph that is one
connected component (making component GroupKFold impossible) is handled with a
deterministic multi-membership purged cross-fit, verified zero-leakage, with inner-
only selection, every arm covered OOF exactly once, four paired significance tests,
a predeclared gate, 25 hash-bound checkpoints, a fail-closed regression test, and
measured resources — all independently reproduced. The strongest exploratory number
(CLAP +9.1 pt, gate-passing) is explicitly refused as a post-hoc winner. Nothing was
shipped, and AC#3 is left honestly unclaimed. The only self-inflicted defect is the
attribution trailer.

---

## What Must Be Fixed (to reach PASS)
1. **Collect real human ratings.** Run the frozen 60-seed blinded study with **≥3
   raters** (ideally 5); aggregate the collector-signed exports into a `sonic_human`
   report.
2. **Meet AC#3 on that human evidence** with margin: ≥20% nDCG@5 (or primary-score)
   gain, ≥80% coherent top-5, CI excluding zero, no scene regressing >10%, no junk/
   duplicate/version items — only then open a fresh commercial FINAL / propose
   deployment.
3. **(Recommended, isolation-preserving)** Predeclare and **freeze a CLAP arm into
   the blinded pack before ratings** — PCA-compressed, preview-embedded, memory-
   measured, with the MTAT overlap caveat stated — so CLAP is tested cleanly rather
   than promoted on MTAT.
4. **Drop the attribution trailer** on all future `[B]` commits.
5. **Decide the staged endpoint status:** deploy + live-verify (AC#5) or keep
   explicitly labelled staged-not-deployed.
