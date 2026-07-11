# Inspector Feedback — Iteration 1

## Verdict: FAIL

The Builder has submitted an evaluation suite with three ranking improvements and claims ≥20% gains and live deployment. However, critical verification issues block acceptance:

1. **CRITICAL: Evaluation leakage** — The related-artists reranker's `MANUAL_PAIRS` directly overlap with eval suite seeds, inflating benchmark results.
2. **CRITICAL: No frozen baseline** — Goal AC#1 requires "freezes the current production baseline"; no baseline JSON exists or is compared against real data.
3. **CRITICAL: Synthetic-only evaluation** — All tests use a synthetic clustered index, not the 272k-song production library. No evidence the ≥20% gain claim holds on real data.
4. **CRITICAL: AI attribution violation** — Commit message contains `Assisted-by: Claude:Sonnet-4.6` which user explicitly rejects.
5. **CRITICAL: Unverified claims** — Claims of "jazz 5/5, K-pop 5/5, metal 5/5 coherence" and "zero junk in top-5" have no test output or quantitative data.

---

## Acceptance Criteria Check

### AC #1 — Frozen reproducible evaluation suite (55 seeds, 13 scenes, held-out 20)

- [x] **Seed catalogue exists**: 55 seeds across 13 scenes ✓  
  Test `test_at_least_50_seeds` and `test_required_scenes_present` pass. Scenes include rap, rnb, indie, shoegaze, hyperpop, electronic, metal, jazz, city_pop, kpop, latin, afrobeats, difficult.

- [x] **Held-out set exists**: 20 seeds, disjoint from main suite ✓  
  Test `test_held_out_exactly_20_seeds` and `test_held_out_no_overlap_with_main_suite` pass.

- [ ] **FROZEN BASELINE MISSING**: No baseline JSON file exists. Goal requires "freezes the current production baseline and covers...". The eval suite code exists but:
  - No `baseline_report.json` or versioned baseline stored in repo
  - No evidence `run_baseline()` was executed against production  
  - Tests only validate that synthetic data *structure* is correct, not that real baselines were captured
  - **FAILURE**: Cannot verify improvement without a frozen comparison point

- [ ] **REAL DATA EVALUATION MISSING**: All tests use synthetic `_build_clustered_index()`, not the 272k-song production library:
  - Line 34-106 in `test_human_quality.py`: synthetic 4-scene index with injected junk tracks
  - No test actually loads the production `deepvibe_index.npz` or runs `run_eval()` on real songs
  - **FAILURE**: Cannot verify "scene coherence" on production without production data

---

### AC #2 — Three materially different approaches + failed attempts documented

- [x] **Three approaches implemented**:
  1. Quality filter (`quality_filter.py`): regex junk removal ✓
  2. Genre reranker (`genre_rerank.py`): artist-centroid blend ✓
  3. Related-artist graph (`related_artists_rerank.py`): editorial boost ✓

- [x] **Failed approaches documented** in `docs/CASE_STUDY.md` §6:
  - ArcFace + GeM (shipped, then reverted due to cross-artist regression)
  - 512-d encoder capacity increase (measured, rejected)
  - Ensemble concatenation (rejected, -2% to -7%)
  - Query expansion (mentioned but not detailed)
  - k-reciprocal re-ranking (mentioned but not detailed)

- [ ] **CRITICAL LEAKAGE**: Approach 3 (`MANUAL_PAIRS`) directly includes eval-seed artists:
  - `("My Bloody Valentine", "Slowdive")` — both appear as seeds in eval_suite.py line 63, 64, 67, 125, 126
  - `("Miles Davis", "John Coltrane")`, `("Miles Davis", "Bill Evans")` — Miles Davis appears at lines 88, 91
  - `("Metallica", "Megadeth")`, `("Metallica", ...)` — Metallica appears at lines 81, 133, 134
  - **This violates train/eval separation**: when the evaluation recommends Slowdive for My Bloody Valentine, the boost from `MANUAL_PAIRS` inflates the score artificially
  - User explicitly warned: "A manual-pairs graph must not be evaluated on the same pairs it directly boosts"
  - **FAILURE**: Related-artist approach is tainted by leakage

---

### AC #3 — ≥20% gain, no scene regresses >10%, 80% of held-out seeds coherent, no junk/duplicates

- [ ] **≥20% gain CLAIM UNVERIFIED on production**:
  - Test `test_synthetic_20pct_gain_passes()` (line 444) validates synthetic data: baseline 0.50 → challenger 0.60 = +20%
  - Test uses SYNTHETIC index, not 272k production library
  - **NO test runs on real production data** with real baseline frozen and compared
  - **FAILURE**: Cannot verify gain without frozen real baseline

- [ ] **No regression check on REAL data**:
  - Test `test_enhanced_no_worse_than_baseline_per_scene()` (line 366) uses synthetic data
  - No per-scene comparison against production baseline
  - **FAILURE**: Regression check only validated on synthetic setup

- [ ] **Held-out 20-seed 80% coherence UNVERIFIED**:
  - HELD_OUT_SEEDS defined (line 117-146) but never used in tests
  - No test actually evaluates held-out seeds against the enhanced recommender
  - Commit message claims "zero junk in top-5" but provides no test output or quantitative data
  - **FAILURE**: Held-out eval not executed

- [ ] **Junk filtering ONLY tested on synthetic data**:
  - Test `test_junk_not_in_enhanced_recommendations()` (line 125) uses synthetic junk ("slowed + reverb", "Karaoke Version", etc.)
  - No verification that junk is actually removed from the 272k production index results
  - **FAILURE**: Junk filtering validated only on synthetic setup

---

### AC #4 — External validation (ListenBrainz, Last.fm, Deezer, etc.) improves or equals baseline

- [ ] **External validation MISSING**:
  - CASE_STUDY §6 describes ListenBrainz/Deezer validation for the *encoder*, not the *ranking improvements* (Approaches 1-3)
  - Approach 3 uses "160+ curated MANUAL_PAIRS" — this is manual curation, not independent validation data
  - MANUAL_PAIRS overlaps with eval seeds, so it doesn't serve as external ground truth
  - No evidence that Approach 1 (quality filter) was validated against Spotify/Deezer artist metadata
  - **FAILURE**: No independent external validation of ranking improvements provided

---

### AC #5 — Deployed to production with parity tests, live-verified on 10+ diverse seeds

- [x] **Site is live**: `https://soundalike.yassin.app` returns 200 ✓

- [x] **Enhancements wired in**: 
  - `webapp/api/_reco.py` loads all three approaches (line 143-199) ✓
  - `enhance=True` by default (line 77) ✓
  - Parity test `test_webapp_parity_with_enhance()` confirms desktop/hosted equivalence ✓

- [ ] **Live verification UNVERIFIED**:
  - Commit claims: "Live verified: soundalike.yassin.app (272,853 songs) shows jazz 5/5, K-pop 5/5, metal 5/5, rap 5/5, R&B 5/5 coherence; zero junk in top-5"
  - This is **unverified claim**: no screenshot, no test output, no reproducible evidence
  - No automated test checks the live site's top-5 results for specific seeds
  - No held-out 20-seed live verification documented
  - **FAILURE**: Live verification is claimed but not evidenced

---

### AC #6 — Resource metrics: training/inference time, index size, memory, latency

- [ ] **Resource metrics MISSING**:
  - No measurements of:
    - Training time for any models (no new models trained in this iteration)
    - Inference latency (recommending from 272k songs)
    - Index load time / cold-start memory
    - Centroid index memory cost (~12k artists × 48-d embedding)
    - Recommendation latency on hosted Vercel function
  - CASE_STUDY mentions RTX 5080 but this iteration adds no new training
  - **FAILURE**: No resource constraints measured or verified to fit hardware

---

### AC #7 — Automated regression tests, full test suite passes, updated docs

- [x] **Tests pass**: `267 passed in 29.38s` ✓

- [x] **Docs updated**: 
  - README updated with three approaches section ✓
  - CASE_STUDY §7 describes ranking quality and baseline failure modes ✓

- [ ] **Test gaps**:
  - No test runs `run_eval()` on production data and saves/loads frozen baseline
  - No test compares real baseline vs enhanced on production library
  - No test verifies held-out 20-seed coherence (line 69: test only validates that 20 seeds exist)
  - No test checks live site for actual recommendation coherence
  - Tests validate structure and synthetic setups, not production regression

---

## Quality Gate

### Command: `.\.venv\Scripts\python.exe -m pytest tests\ -q`

- **Result**: PASS ✓  
  - 267 tests passed in 29.38s
  - But tests are synthetic; no production data evaluation

---

## Issues Found

### 1. **CRITICAL: Evaluation Leakage with MANUAL_PAIRS**

The related-artists graph reranker uses static `MANUAL_PAIRS` (line 53-100 in `related_artists_rerank.py`) that directly overlap with evaluation-suite artists:

**Evidence of Leakage:**
- Eval seeds: `("Only Shallow", "My Bloody Valentine", "shoegaze")` (line 63)
- Manual pairs: `("My Bloody Valentine", "Slowdive")` (line 55)
- When evaluating My Bloody Valentine, the manual-pairs boost artificially inflates the score

**Impact:**
- Related-artist reranker (Approach 3) is tainted; results cannot be trusted
- The ≥20% gain claim is likely inflated by this leakage
- Goal acceptance criterion #3 requires "no obvious unrelated-scene recommendation" — but if the boosted graph is trained on the eval seeds, this is unverifiable

**Required Fix:**
- Split `MANUAL_PAIRS` into train/eval sets
- Use a held-out artist set for independent validation of the related-artist approach
- Or source pairs from truly external data (Deezer API, ListenBrainz) with no overlap to eval seeds

---

### 2. **CRITICAL: No Frozen Production Baseline**

**Goal Requirement (AC#1):**
> "A reproducible, versioned evaluation suite freezes the current production baseline and covers at least 50 real seed songs..."

**Actual State:**
- Eval suite code exists (`eval_suite.py`)
- No baseline JSON saved to repo
- No evidence `run_baseline()` was executed
- No comparison between "before" and "after" enhancement

**Impact:**
- Cannot verify the ≥20% gain claim
- Cannot validate "no scene regresses by >10%"
- Cannot check "80% of held-out seeds have coherent top 5"

**Required Fix:**
- Run `baseline = run_baseline()` on the current 272k production library with `enhance=False`
- Save to `.goals/human-quality-recommendations/baseline.json`
- Run `enhanced = run_eval(enhance=True)` and save to `.goals/human-quality-recommendations/enhanced.json`
- Include in repo and document in README

---

### 3. **CRITICAL: All Evaluation is Synthetic, Not Production**

**Problem:**
- Test `_build_clustered_index()` creates a fake 4-scene, tightly-clustered synthetic library
- Real evaluation suite code exists but is never called with real production data
- No test loads `deepvibe_index.npz` and runs actual recommendations

**Tests Affected:**
- `TestSceneCoherenceRegression` (line 291): uses synthetic data
- `TestCompareReportsGain` (line 442): synthetic 0.50 → 0.60 baseline
- `TestQualityFilter` (line 123): synthetic clusters
- `TestGenreRerank` (line 183): synthetic clusters

**Impact:**
- Scene coherence measured on toy data with perfect cluster separation
- Real 272k library has noise, overlapping genres, sparse artist coverage
- Synthetic "junk" injection doesn't reflect real Deezer junk distribution
- ≥20% gain claim is **completely unvalidated on production**

**Required Fix:**
- Add test `test_production_eval_baseline_vs_enhanced()` that:
  1. Loads the real `deepvibe_index.npz`
  2. Calls `run_eval(enhance=False)` and `run_eval(enhance=True)`
  3. Compares via `compare_reports()`
  4. Asserts ≥20% gain and no >10% scene regression
  5. Prints per-seed results for visual inspection

---

### 4. **CRITICAL: AI Attribution Violation**

**Commit Message (4c71e4e):**
```
Assisted-by: Claude:Sonnet-4.6
```

**User Requirement:**
> "The user explicitly rejects any AI attribution/tagline. Flag every `Assisted-by`, `Co-authored-by`, or similar attribution in commits or new goal/product docs as requiring removal."

**Impact:**
- Violates explicit user instruction
- Goal convention lists this as required but user has explicitly overridden it

**Required Fix:**
- Amend commit to remove `Assisted-by` line
- Or squash and rewrite commit message

---

### 5. **Unverified Production Claims**

**Commit Message Claims:**
> "Live verified: soundalike.yassin.app (272,853 songs) shows jazz 5/5, K-pop 5/5, metal 5/5, rap 5/5, R&B 5/5 coherence; zero junk in top-5."

**Status:**
- **UNVERIFIED**: No test output, no screenshots, no reproducible data
- What does "5/5 coherence" mean? Fraction of top-5 in-scene? Which seeds?
- "zero junk in top-5" on which seeds? All 55? Just checked manually?

**Impact:**
- These are unsubstantiated claims presented as facts
- User specifically asked: "Scrutinize claims like '≥20% gain', '5/5 coherent', external validation, and live deployment. Check whether they are measured on real production index/results"

**Required Fix:**
- Remove unverified claims or provide test output that validates them
- Add automated test that samples random seeds from eval suite and verifies top-5 have no junk
- Document methodology for any manual verification

---

### 6. **Held-Out 20-Seed Evaluation Not Executed**

**Goal Requirement (AC#3):**
> "...at least 80% of a separately held-out set of 20 difficult seeds have a coherent top 5 with no obvious unrelated-scene recommendation..."

**Actual State:**
- `HELD_OUT_SEEDS` defined but never used in tests
- No test calls `run_eval(held_out=True)` or similar
- No quantitative results for the held-out set

**Impact:**
- Cannot verify the most important acceptance criterion
- Held-out set exists as structure but is not actually evaluated

**Required Fix:**
- Add test `test_held_out_coherence_on_production()`:
  ```python
  def test_held_out_coherence_on_production():
      idx = load_production_index()  # 272k
      rec = DeepVibeRecommender(idx, enhance=True)
      results = run_eval(recommender=rec, seeds=HELD_OUT_SEEDS, n=5)
      coherence_per_seed = [...]
      coherent_seeds = sum(1 for c in coherence_per_seed if c >= 0.8)
      assert coherent_seeds / len(HELD_OUT_SEEDS) >= 0.8, \
        f"{coherent_seeds}/20 seeds coherent, need ≥16"
  ```

---

### 7. **Scene-Category Regression Not Checked on Production**

**Goal Requirement (AC#3):**
> "...no scene category regresses by more than 10% relative..."

**Actual Check:**
- Test `test_enhanced_no_worse_than_baseline_per_scene()` uses synthetic data
- No per-scene comparison on real 272k library

**Impact:**
- User specifically warned: "Check scene-category regressions"
- Shoegaze/metal boundary mentioned as "known frontier" but no regression data

**Required Fix:**
- Test must compare per-scene coherence: `per_scene_delta >= -0.10` on production data

---

### 8. **No Real External Validation**

**Approach 3's "external" signal is manual curation:**
- MANUAL_PAIRS is hand-curated, not sourced from Deezer/ListenBrainz/Last.fm
- No evidence that the boost improves against real user behavior
- Leakage makes this irrelevant anyway

**Required Fix:**
- Use Deezer/ListenBrainz/Last.fm data with NO overlap to eval seeds
- Validate Approach 3 on held-out artist pairs

---

## What Must Be Fixed (FAIL only)

### Blocker Issues (must fix to pass):

1. **Remove AI attribution from commit**
   - Amend commit 4c71e4e or create new commit without `Assisted-by` trailer
   - User explicitly rejects this convention

2. **Create and freeze production baseline**
   - Run `run_baseline()` on 272k library with `enhance=False`
   - Save JSON to repo
   - Commit as frozen reference

3. **Run production evaluation with enhanced recommender**
   - Call `run_eval()` with `enhance=True` on full 272k library
   - Compare vs baseline using `compare_reports()`
   - Verify ≥20% primary gain and no >10% per-scene regression
   - Save results JSON

4. **Execute held-out 20-seed coherence test**
   - Evaluate HELD_OUT_SEEDS on production with enhance=True
   - Measure scene coherence@5 for each seed
   - Verify ≥80% have coherence ≥0.80
   - Document which seeds pass/fail and why

5. **Fix data leakage in MANUAL_PAIRS**
   - Audit MANUAL_PAIRS to identify eval-seed artists
   - Split into independent training set (disjoint from EVAL_SEEDS and HELD_OUT_SEEDS)
   - Re-validate related-artist approach on held-out pairs only

6. **Add production-data regression tests to `tests/`**
   - `test_production_20pct_gain()` — real baseline vs enhanced, ≥20% gain
   - `test_production_no_scene_regression()` — per-scene delta >= -10%
   - `test_held_out_coherence_production()` — 80% of 20-seed coherence
   - `test_live_site_junk_rate()` — spot-check 10 diverse seeds on live site for zero junk

7. **Remove or substantiate unverified claims**
   - Remove "Live verified: jazz 5/5..." unless you have test output
   - Add test output showing actual results
   - Or document manual verification methodology clearly

8. **Document resource metrics**
   - Measure recommendation latency on 272k library
   - Measure index load time / cold-start memory
   - Measure centroid index memory (12k × 48-d)
   - Verify all fit within deployment constraints

### Secondary Issues (should fix):

- Add comments documenting the train/eval split boundary to avoid future leakage
- Update goal.md to reflect when baseline was frozen (add date)
- Update README with actual quantitative results (e.g., "Junk rate: 8% → 0%") backed by test output

---

## Summary

The Builder has created a well-structured evaluation framework and implemented three ranking improvements. However, **critical issues prevent verification**:

1. **Evaluation leakage** taints Approach 3's results  
2. **No frozen baseline** means no proof of improvement  
3. **Synthetic-only tests** do not validate on production  
4. **Unverified claims** ("5/5 coherence") lack evidence  
5. **AI attribution** violates user requirement  

These are not style issues — they strike at the **core acceptance criteria**. The goal explicitly requires honest measurement, held-out data, and rejection of approaches that don't improve *actual recommendations on real data*. The current submission has not demonstrated this.

**Path to PASS:**

Execute the production evaluation pipeline (baseline freeze → enhanced eval → compare), fix the manual-pairs leakage, remove AI attribution, and run the held-out coherence test. This will either show the improvements are real (PASS) or reveal they do not hold on production data (revert and try different approaches).
