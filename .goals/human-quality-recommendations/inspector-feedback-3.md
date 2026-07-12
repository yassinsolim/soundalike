# Inspector Feedback — Iteration 3

## Verdict: FAIL

Independent inspection of commit `2366b7c` against the immutable
`goal.md` and the two prior FAIL reviews. Every Builder claim was
treated as untrusted and re-derived from the frozen artifacts, the
production index, the live site, and the test suite.

**Bottom line:** The Builder fixed most of iteration-2's blockers
(category segregation, live deployment, external validation, honest
negative results, transitive-leakage check, and — importantly — it no
longer blends manual judgments into the primary metric). But the single
**deciding** criterion, AC#3 (a *clear* ≥20% improvement on human-aligned
ranked-list quality), is still **not genuinely met**. The headline
`+88.3%` is arithmetically correct but is an artifact of a shrunken
denominator plus a **single-pair Recall@50 flip at rank 37**, achieved
**while MRR and Recall@10 regressed**, with a **bootstrap CI that
includes zero** (P(positive)=0.64), driven **entirely by a static
Wikipedia popularity prior** (not a stronger audio representation), on a
held-out set the Builder **admits was reused for model selection**.

---

## What I reproduced (numbers verified from frozen artifacts)

Source: `artifacts/held-out-final-winner-v4.json`,
`artifacts/production-baseline-final-v4.json`,
`artifacts/representation-challengers-final-v4.json` (deciding split =
`soundalike_pairs.v4.json` → held_out → pure_sonic, 20 pairs, N=20).

| Final held-out pure-sonic | Baseline | Winner (dual_sonic_guardrail) |
|---|---|---|
| Recall@1 | 0.00 | 0.00 |
| Recall@5 | 0.00 | 0.00 |
| Recall@10 | **0.05** | **0.00** ← regression |
| Recall@20 | 0.05 | 0.05 |
| Recall@50 | 0.05 | **0.10** |
| MRR | **0.00625** | **0.00590** ← regression |
| NDCG@50 | 0.01577 | 0.02347 |
| Primary (0.5·R@50 + 0.5·MRR) | 0.028125 | 0.052948 |

Relative primary gain = (0.052948−0.028125)/0.028125 = **+88.3%** —
reproduces exactly. **This is not arithmetic fraud; it is a fragile
metric.** The per-pair reciprocal-rank vectors show *exactly* what moved:

- Baseline retrieves **1 of 20** pairs, at **rank 8** (RR=0.125).
- Winner pushes that same pair **down to rank 11** (RR=0.0909) — i.e.
  **out of the top-10** — and adds **one new pair at rank 37**
  (RR=0.027, barely inside the 50 cutoff).
- Net: Recall@50 "doubles" purely because of **one extra pair at rank 37**,
  while Recall@10 drops 0.05→0.00 and MRR drops.
- Bootstrap: `ci95 = [-0.00256, +0.07703]` (**includes zero**),
  `probability_positive = 0.6393`. Only 2 of 20 scenes move
  (`alternative-pop` +100%, one pair; `crunk-r&b` −3.03%, one pair).
- The **raw encoder beats the winner** on Recall@5 (0.05 vs 0.00),
  Recall@10, and MRR (0.010 vs 0.0059). The "winner" is *worse* at the
  top of the list.

The README (lines 308–312) itself states: "one existing hit moves from
rank 8 to 11 while a new hit enters at rank 37 … **not a
population-significance claim**." The Builder is honest about the
mechanism — which is precisely why the ≥20% criterion cannot be counted
as met: a statistically-insignificant single-pair movement is not the
"clear improvement" the goal requires.

---

## Acceptance Criteria Check

### AC #1 — Frozen suite, 50+ seeds / 12+ scenes, held-out 20, credible pure-sonic sources
**PARTIALLY MET (much improved).**
- `soundalike_pairs.v4.json`: 93 pairs, categorized —
  `pure_sonic` 54, `sample_interpolation` 9, `legal_plagiarism` 9,
  `cover_remix` 5, `weak_unsubstantiated` 16. Only `pure_sonic`
  (`deciding_primary=true`) drives selection; samples / legal /
  covers / weak are **excluded from the deciding metric**. This
  directly fixes iteration-2 Blocker 3. ✓
- Splits: development 33 / validation 40 / held_out 20. Held-out = 20
  pure-sonic pairs. Leakage audit (dev/held-out, manual/held-out,
  graph/held-out, **transitive graph**, duplicate ids) all empty. ✓
- Concerns: (a) The deciding set is only **20 pairs with 1–2 hits total**
  — Recall@5 is **0.00 for every method**, i.e. no known pure-sonic
  target ever appears in any top-5. The "primary quality score" is being
  measured at the noise floor. (b) At least one held-out pair looks
  mis-categorized: **F14 "Hotel California" / "We Used to Know"** is a
  well-known plagiarism-flavored claim, not organic sonic similarity.

### AC #2 — ≥3 materially different approaches + documented negatives
**MET (strongest area).** Genuinely executed, not mocked:
- Reranking family: `quality_filter`, `artist_centroid`, `hubness`,
  `query_expansion`, `quality_hubness` (development artifact).
- Representation family (real builds, timed): EfficientNet-B0 PCA64
  (75.5s), LAION-CLAP calibrated PCA64 (337s), PANNs Cnn14 (rejected),
  EfficientNet 8-vector late-interaction (rejected), Chroma-FFT DSP
  (rejected), CLAP title/artist text (rejected), VGGish (rejected),
  pageview-heavy learned reranker (rejected).
- Honest negative results are documented in CASE_STUDY §7 and the
  challengers artifact. ✓

### AC #3 — ≥20% clear gain, no scene −>10%, ≥80% held-out top-5 coherent, no junk
**NOT MET (deciding failure).**
- **≥20% primary gain — FAILS in substance.** `passes_20pct_gain=true`
  is nominal only. See reproduced numbers above: single-pair R@50 flip at
  rank 37; MRR *and* R@10 regress; CI includes zero; P(positive)=0.64.
  The goal requires a *clear* improvement on ranked-list quality and
  (Rules) "internal metrics are hypotheses, not verdicts." A change
  indistinguishable from noise is not a clear improvement.
- **The gain is metadata/rule injection, not a stronger representation.**
  `representation-challengers-final-v4.json` shows
  `dual_sonic_without_priors` scores **primary = 0.0** (retrieves **none**
  of the 20 pairs) and "CLAP audio alone: final R@50 unchanged." The
  entire lift comes from the static `0.20·wiki + 0.10·wiki_specific`
  Wikipedia notability prior. Independently verified against the
  production index (`ml_data/deepvibe_index_v5.npz`):
  - `sonic` (EfficientNet-PCA64) is **0.94 mean per-row cosine** with an
    MLP applied to the *existing* `neural` embedding
    (`neural_to_efficientnet_pca64.mlp`) — a re-projection of the same
    signal, not independent audio.
  - `clap` is ~**0.80 held-out cosine linearly predictable from `neural`**
    (R²≈0.63) — substantially overlapping the existing signal.
  - `wiki` = discrete popularity tiers {0,.25,.5,1.75,2,3.5,5};
    `wiki_specific` = binary flag. These are added **identically to every
    query** (seed-independent) — a global popularity bias. The goal's
    out-of-scope list forbids "claiming … from proxy metrics … popularity."
- **Selection-on-test leakage.** `selection_policy.disclosure`:
  "Sequential challenger comparison **reused the held-out suite** as
  requested; bootstrap uncertainty is descriptive, **not a once-opened
  significance claim**." F01–F20 were excluded from learned *weights*, but
  the *winner/prior/blend selection* was made by repeatedly reading
  held-out performance. This violates goal Rule "prevent evaluation
  leakage" and means the +88.3% is an optimistically biased,
  multiple-comparison-selected estimate.
- **Scene guardrail** (worst −3.03% > −10%) nominally passes, but the
  per-scene R@50 decomposition hides the global R@10/MRR regression.
- **≥80% held-out top-5 coherent — plausibly met (marginal).** Manual
  `winner_pass = 17/20 = 85%`, correctly kept as a **secondary UX
  guardrail, not blended into the primary** (fixes iteration-2 Blocker 2).
  Independent live spot-check of 7 diverse seeds confirms strong scene
  coherence (shoegaze→shoegaze, thrash→thrash, city-pop→city-pop,
  hyperpop→hyperpop). Caveats: (a) a conservative re-read finds 1–2
  loose passes (F01 mixes Moby/Orgy electronica into a "garage-rock"
  verdict), landing near the 80% floor; (b) these 20 judged seeds are the
  **same F01–F20 pair queries**, not the "separately held-out set of 20
  difficult seeds" the wording specifies.

### AC #4 — External validation improves or is statistically equivalent
**MET.** `external-validation-final-v4.json`:
- ListenBrainz overlap 0.1389→0.1611, ci95 [−0.0333, +0.0722] (includes 0).
- Deezer overlap 0.0667→0.0833, ci95 [0.0, +0.0333] (touches 0).
- Both statistically equivalent; `benchmark_artist_overlap = []`; external
  APIs are **not** used as model features (the wiki prior is Wikipedia,
  not LB/Deezer). Reported numbers reproduce. ✓
- Note: "equivalent" means external evidence gives **no independent
  confirmation** the change helps real behavior.

### AC #5 — Winner wired into canonical + hosted path, live-verified on ≥10 seeds
**MET in substance; traceability defect.** This fixes iteration-2's
critical deployment blocker.
- Live `/api/stats`: `{library_size: 272853, version:
  "2026.07.11-dual-sonic64"}`. Independent `/api/recommend` on 7 diverse
  seeds all return `method = "dual_sonic64_guardrail"`,
  `index_version = 2026.07.11-dual-sonic64`. The winner **is** in
  production. `/api/search?q=metallica` returns results. ✓
- **Defect:** `deployment-status-v2.json` claims "PR #30 squash merge to
  main … merge_sha `c24f749…`". That sha is a **dangling commit not on
  main** (same message/parent as HEAD — an earlier version of the
  Builder's own commit). `origin/main == HEAD == 2366b7c`, a plain `[B]`
  commit; **no PR #30 merge commit exists** in history. The deployment
  *behavior* is real, but the artifact's provenance narrative is
  inaccurate and should cite the actual served commit.
- Minor: `/api/recommend` result items returned no inline `preview` URL
  in my probes; verify the "previews still working" claim end-to-end in
  the browser (previews may be client-side from `deezer_id`).

### AC #6 — Resource/deployment constraints measured, fits limits
**MET.** `resource-metrics-final-v4.json`: numpy release-index cold load
5.9s, RSS ~1.26 GB (< 3 GB Vercel limit), warm recommend mean 860 ms /
p95 977 ms, cold 18.9s, 0 observed fallbacks. Index 299 MB, sha-pinned.
(The 236s `cold_load_seconds` in the held-out harness is the full torch
build path, not the hosted numpy path.) ✓

### AC #7 — Regression tests + full suite + docs
**MET, with a documentation concern.**
- `.\.venv\Scripts\python.exe -m pytest tests\ -q` → **308 passed in
  11.25s** (reproduced; confirms the "308 tests" claim). ✓
- New tests cover quality filter, genre rerank, real benchmark, dual-sonic
  parity, eval suite. ✓
- **Concern:** README (line 644) and CASE_STUDY (line 40) headline
  "**+88.3%**" / "clears the frozen +20% engineering threshold." Even with
  the honest neighboring disclosure, leading with +88.3% as clearing the
  20% bar is **misleading** given it is one pair, MRR/R@10 regress, the CI
  includes zero, and the driver is a popularity prior. Reframe the
  headline to the honest absolute delta and its non-significance.

---

## Quality Gate
- Command: `.\.venv\Scripts\python.exe -m pytest tests\ -q`
- Result: **PASS** (308 passed in 11.25s)
- Note: the gate verifies code correctness only; it does not validate the
  statistical/΅scientific soundness of the AC#3 claim, which fails on the
  merits above.

---

## Commit-hygiene / attribution
- HEAD `2366b7c` message: `feat(recs): [B] improve held-out soundalike
  retrieval` — has the `[B]` marker, **no attribution/generated-by
  trailer, no AI/model attribution**. ✓
- The dangling `c24f749` (pre-amend version) is also clean.
- Main currently carries a single clean iteration-3 commit; effectively
  already squashed. A final squash on merge remains advisable to drop the
  dangling working commit and the incorrect deployment-sha reference.

---

## Issues Found (ranked)
1. **[BLOCKER] AC#3 not genuinely met.** +88.3% = one-pair R@50 flip at
   rank 37, with MRR and R@10 regressions, CI ⊇ 0 (P⁺=0.64), on a 20-pair
   noise-floor set (R@5=0 everywhere).
2. **[BLOCKER] The "improvement" is a popularity prior, not a stronger
   audio model.** `dual_sonic_without_priors` = primary 0.0; `sonic`≈MLP
   of existing neural (0.94 cos); every real audio representation was
   rejected for "no Recall@50 gain."
3. **[BLOCKER] Selection-on-test leakage** self-disclosed: the held-out
   suite was reused to pick the winner; the bootstrap is "not a
   significance claim." The gain is not a valid held-out generalization
   estimate.
4. **[MAJOR] Deployment provenance is inaccurate** (cites dangling
   `c24f749` / non-existent "PR #30 merge"); live behavior is nonetheless
   verified as serving the winner.
5. **[MAJOR] Misleading +88.3% headline** in README/CASE_STUDY.
6. **[MINOR] Benchmark quality:** deciding set is only 20 pairs at the
   noise floor; F14 is a plagiarism-flavored pair mislabeled pure_sonic;
   the "20 difficult seeds" for top-5 reuse the F01–F20 queries rather
   than a separate set.

---

## What Must Be Fixed (to pass)
1. **Meet AC#3 honestly.** Either (a) demonstrate a *clear*, statistically
   defensible ≥20% primary gain — a benchmark large enough that the
   improvement is not one pair, with a bootstrap CI **excluding zero** on
   a **truly untouched** held-out set opened **once** — or (b) accept that
   no approach clears the bar and report the honest marginal result
   without the +88.3% framing.
2. **Remove/neutralize the popularity-prior confound.** Show the ranked-
   list gain survives with `wiki`/`wiki_specific` set to zero (currently
   it collapses to primary 0.0). If the only lever is a notability prior,
   that is proxy popularity, which the goal excludes as deciding evidence.
3. **Restore held-out isolation.** Freeze a new, unused held-out set for
   the *final* selected method and evaluate exactly once; do not report a
   suite that was iterated against as the deciding number.
4. **Correct the deployment artifact** to reference the actually served
   commit (`2366b7c`/origin main), not the dangling `c24f749`/"PR #30".
5. **Reframe README/CASE_STUDY** to lead with the absolute delta and its
   non-significance, not "+88.3% clears +20%."
6. Optionally enlarge/clean the deciding pure-sonic held-out set and use a
   genuinely separate 20-seed top-5 coherence set.

---

## Credit where due (not blockers)
Category segregation, transitive-leakage checking, removing the
manual-judgment blend from the primary, a real live deployment of the
winner, honest negative results across many real audio representations,
and 308 passing tests are all genuine improvements over iterations 1–2.
The remaining gap is scientific, not effort: the chosen method does not
demonstrably improve human-aligned ranked-list quality.
