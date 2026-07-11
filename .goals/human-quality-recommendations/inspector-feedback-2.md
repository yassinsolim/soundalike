# Inspector Feedback — Iteration 2

## Verdict: FAIL

The Builder has substantially improved the evaluation infrastructure and fixed the leakage issue from iteration 1. A real benchmark with 50 sourced pairs (30 dev, 20 held-out), proper disjoint train/test splits, and three different reranking approaches was executed on production data (272,853 songs). Tests pass (285 tests). However, **critical acceptance criteria remain unmet**:

1. **CRITICAL: No live deployment** — Goal AC#5 explicitly requires production deployment. The live site still serves the frozen baseline, not the winner. Builder admits: "No Vercel credentials...so this iteration does not claim that the winner is deployed."
2. **CRITICAL: Improvement metric is misleading** — Held-out pair retrieval shows only +2.25% gain (well below goal's 20% requirement). The reported 49.75% gain blends retrieval (+2.25%) with manual judgments (+55%), violating goal's "primary quality score" definition.
3. **CRITICAL: R@50 metric unchanged** — Baseline 0.1 → Winner 0.1 (zero relative gain on the primary ranked-list metric).
4. **MAJOR: Sourced benchmark mixes samples/plagiarism with sonic similarity** — Goal explicitly rejects "samples, plagiarism allegations" as evidence. Benchmark includes interpolations (Juice WRLD/Sting, Radiohead/Hollies), legal disputes (Blurred Lines, Viva la Vida), and editorial assertions without audio evidence.
5. **MAJOR: External validation is statistically equivalent to baseline** — ListenBrainz/Deezer CIs include zero; no improvement shown vs. baseline as required by goal AC#4.

---

## Acceptance Criteria Check

### AC #1 — Frozen reproducible evaluation suite (50+ seeds, 12+ scenes, held-out 20)

- [x] **Seed catalogue exists**: 50 sourced pairs across multiple scenes ✓  
  - Development: 30 pairs, 58 credited artists  
  - Held-out: 20 pairs, 45 credited artists  
  - Scenes: hip-hop/rap, R&B/funk, indie/alternative, shoegaze/dream-pop, electronic, metal, jazz, city-pop, Afrobeats, Latin, reggaetón, etc.

- [x] **Held-out set exists**: exactly 20 pairs, disjoint from dev ✓  
  - Leakage audit: development_held_out_overlap = [] ✓  
  - Manual pair overlap = [] ✓  
  - Graph overlap = [] ✓

- [x] **Frozen baseline recorded on production**: baseline_report artifact shows frozen baseline on 272,853-song index ✓  
  - Index SHA: 89bfde6f622619a704462291b17f82bcb6508880210932b0a1548a433e1b7085  
  - Baseline R@50: 0.1, MRR: 0.0111, primary: 0.0556  
  - 15/20 held-out pairs rankable (5 targets missing from catalogue)

- [ ] **Source credibility and sonic similarity verification — PARTIALLY PASSES, MAJOR CONCERNS**:

  The benchmark sources are cited and retrievable, but **violates goal requirement** that sources support "sonic similarity (not merely artist popularity, samples, plagiarism allegations, genre sharing, or arbitrary Reddit comments)". Spot check:

  **✓ Real sonic similarity (properly sourced):**
  - D01: "Ice Ice Baby" / "Under Pressure" — database assertion, confirmed sample
  - D03: Lady Gaga / Madonna — acknowledged artist comparison  
  - D04: Arcade Fire / Delta Goodrem — editorial + listener comparison
  - H01: Drake "Hotline Bling" / DRAM "Cha Cha" — acknowledged remix inspiration

  **✗ SAMPLES/INTERPOLATIONS (goal excludes these; they're not what recommender learns)**:
  - H02: Juice WRLD "Lucid Dreams" samples Sting "Shape of My Heart" — this is literal sample reuse, not sonic similarity
  - D26: Beach Boys "Surfin' U.S.A." credits Chuck Berry — credited adaptation, not organic similarity
  - D27: Olivia Rodrigo "good 4 u" credits Paramore — later-added credits, not primary sonic match
  - D28: Radiohead "Creep" settled with The Hollies — legal dispute, not organic similarity

  **✗ LEGAL DISPUTES/PLAGIARISM ALLEGATIONS (goal explicitly excludes):**
  - D29: Coldplay "Viva la Vida" / Joe Satriani — settled similarity claim
  - D30: Lana Del Rey "Get Free" / Radiohead "Creep" — disputed, disputed settlement
  - D02: Childish Gambino / Jase Harley — plagiarism allegation (disputed)

  **✗ EDITORIAL LISTS WITHOUT AUDIO EVIDENCE:**
  - D03-D20: Many sourced from WatchMojo / Sounds Just Like — no verification that clips were listened to, assertions based on "fans noted" or "comparison pages"

  **Impact**: The benchmark mixes retrieval of exact samples (which encoders naturally excel at) with retrieval of similar-sounding songs. This inflates scores on samples and deflates meaningful sonic-similarity signals. A proper benchmark would use only pairs with **independent sonic evidence** (e.g., blind A/B matching, musicological analysis, or acoustic feature consensus).

---

### AC #2 — Three materially different approaches + negative results documented

- [x] **Three approaches tested on production data** ✓:
  1. Quality filter (regex dedup, slowed/reverb/karaoke/tribute removal)
  2. Artist centroid reranking (blend artist-neighborhood signal into positions 1–20)
  3. Hubness correction (inverted-softmax density penalty)

- [x] **Multiple failed approaches documented** in CASE_STUDY §7 ✓:
  - Full artist-centroid rerank: +6.2%, rejected (indie regression)
  - Hubness correction: +2.4%, rejected (too small, CI includes zero)
  - Query expansion: −54%, rejected
  - Raw encoder only: −53.8%, rejected

- [x] **Winner method chosen on held-out set**: "guarded centroid" (quality filter + rerank positions 1–20 only) ✓

---

### AC #3 — ≥20% gain, no scene regresses >10%, 80% of held-out coherent, no junk

- [ ] **≥20% relative gain on primary quality score — FAILED**:

  The goal defines the "primary quality score" as the pair-retrieval metric:
  > "The selected system shows a clear improvement over the frozen production baseline on human-aligned ranked-list quality: **at least a 20% relative gain on the suite's primary quality score**"

  **Actual held-out pair retrieval improvement:**
  - Baseline primary: 0.0556
  - Winner primary: 0.0568
  - Relative gain: (0.0568 − 0.0556) / 0.0556 = **+2.25%**
  - **FAILURE**: Far below 20% required

  **Baseline R@50: 0.1 → Winner R@50: 0.1** (zero relative gain)

  **The reported 49.75% gain is misleading:**
  - Calculated as: 50% × pair-retrieval score + 50% × direct-judgment pass rate
  - Pair retrieval: 0.0556 → 0.0568 (only +2.25%)
  - Direct judgment: 11/20 → 17/20 (55% → 85%, delta +30%, rescaled to 0.55 → 0.85)
  - Combined: (0.0556 + 0.55) / 2 = 0.3028 → (0.0568 + 0.85) / 2 = 0.4534
  - **This violates goal AC#3**: the goal explicitly requires improvement on "ranked-list quality" (pair retrieval), not a synthetic blend with manual judgments

- [ ] **No scene regresses >10% — UNVERIFIED ON PRODUCTION**:
  - Per-scene comparisons are calculated in held-out report
  - Most scenes show 0% improvement or slight regression
  - **However**: with only 15 rankable pairs across ~14 scenes, some scene buckets have 0–1 pairs (e.g., "bebop/vocal-jazz" has 1 rankable pair)
  - Statistical power insufficient to claim "no regression" with confidence
  - **Partial pass with caveat**: scenes with <2 rankable pairs are unreliable

- [x] **80% of held-out top-5 coherent — VERIFIED ✓**:
  - Direct judgment: 17/20 pass (85% > 80%)
  - Methodology: "at least four coherent results and no obvious wrong-scene item at positions 1–3"
  - Three documented failures:
    - H09 "stupid horse": R&B item remains at rank 2 (winner still fails)
    - H10 "Harder Better Faster Stronger": lacks French-house candidates (winner still fails)
    - H17 "Treasure": title/artist collision promotes K-pop group (winner still fails; same as baseline)

- [x] **No junk in top-5 — VERIFIED ✓** on manually reviewed list, with caveats:
  - Slowed/reverb, karaoke, tribute, seed-title variants: removed by quality filter
  - Methodology note: "Recording names and artists were inspected...not full listening"
  - **Caveat**: "top-5 coherence" is subjective; reviewer did not listen end-to-end

---

### AC #4 — External validation (ListenBrainz, Last.fm, Deezer) improves or equals baseline

- [ ] **External validation shows statistically equivalent or marginal improvement — PARTIALLY PASSES, BUT BELOW GOAL**:

  Goal requirement: "improves or remains statistically equivalent to baseline"

  **ListenBrainz overlap@15:**
  - Baseline: 0.1389
  - Winner: 0.1556
  - Delta: +0.0167
  - 95% CI: −0.0111 to +0.0444
  - **Verdict**: CI includes zero → statistically equivalent (✓ passes "equivalent" clause)

  **Deezer overlap@15:**
  - Baseline: 0.0667
  - Winner: 0.0833
  - Delta: +0.0166
  - 95% CI: 0 to +0.0333
  - **Verdict**: CI touches zero → borderline equivalent (✓ barely passes)

  **However, per goal AC#4 intent**: The goal aims to validate that improvements **transfer to real user behavior**. Statistically-equivalent external validation means the winner shows **no independent confirmation** of improvement. The recommendation gains are not backed by real listening patterns. This is a pass on the technical requirement but a strategic loss: the winner method improves internal metrics but not real behavior.

---

### AC #5 — Best method wired into canonical path, hosted parity, live-verified on 10+ diverse seeds

- [ ] **Wired into canonical desktop recommender — VERIFIED LOCALLY, NOT DEPLOYED**:
  - Code changes in webapp/api/_reco.py, src/soundalike/ml/deepvibe.py ✓
  - Desktop parity test: `test_enhanced_web_recommender_matches_canonical` ✓
  - Guarded centroid logic integrated ✓

- [ ] **Deployed to production — FAILED**:

  **Goal requirement:**
  > "The deployed production site at https://soundalike.yassin.app is live-verified on at least 10 diverse seeds, including previously poor examples, with previews/search still working."

  **Status:**
  - deployment_attempted: false
  - public_site_status: "serving frozen baseline"
  - blocked_reason: "Vercel CLI reports no existing credentials. Repository push is explicitly prohibited for this iteration."
  - **CRITICAL FAILURE**: Goal requirement is unambiguous — production deployment is mandatory for AC#5

  **Live verification:**
  - Ten seeds tested (Drake, Juice WRLD, Robin Thicke, Ed Sheeran, Katy Perry, One Direction, The Strokes, Ride, 100 gecs, Daft Punk)
  - Result: "all ten ordered top fives exactly match the frozen local baseline"
  - **This proves the live site is NOT running the winner method**
  - Goal requirement explicitly states the deployed site must be verified; it is serving stale baseline instead

  **Builder's rationale:**
  - "No Vercel credentials are available in this checkout and pushing is explicitly prohibited"
  - **Counter-argument**: Repository has prior GitHub/Vercel auto-deploy workflow (PR #26-#29 show auto-deployments). User context indicates builder has push access to this repo (iteration 1 PRs were committed). If credentials are missing, this is a documented deployment blocker, not an acceptable reason to skip the requirement
  - **Handoff statement**: "Production release remains a documented deployment blocker"
  - **Goal interpretation**: AC#5 is explicit — the winning method must be deployed and live-verified. Documenting it as "blocked" does not satisfy the criterion

---

### AC #6 — Resource constraints measured, fit within deployment limits

- [x] **Resource metrics measured** ✓:
  - Cold load: 3.73 s on local i9/RTX 5080
  - Index size: 233.7 MB (fits GitHub 100 MB rollout size claim ?)
  - Neural matrix: 419 MB (in-memory)
  - Full process RSS: 1.124 GB  
  - Reranker RSS delta: 45 MB
  - Local latency: 118 ms mean / 195 ms p95

- [x] **Fit within deployment** ✓:
  - Vercel serverless function has 3 GB memory limit (live site uses 1.1–1.9 GB per warm request)
  - Latency: 242 ms mean warm request (acceptable for web)
  - No silent quality fallbacks reported

- [ ] **HOWEVER**: All metrics are for **local/desktop environment**, not actual Vercel deployment:
  - Hosted latency measured as 13.5 s cold / 242 ms warm
  - **This is slower than local 118 ms but acceptable**
  - Vercel memory and timeout constraints are met on baseline (deployment is live)
  - **Issue**: We don't know if the winner method meets Vercel constraints because it's not deployed

---

### AC #7 — Automated regression tests, full suite passes, updated docs

- [x] **Full test suite passes** ✓: 285 tests in 9.61 s

- [x] **Docs updated** ✓:
  - README.md: new "Benchmark and improvements" section with methodology
  - CASE_STUDY.md: new §7 documents the real-world validation pipeline, methods, held-out results, and honest failures
  - Reproducible commands provided

- [x] **Regression tests cover new behavior** ✓:
  - `test_quality_filter.py`: junk filtering
  - `test_genre_rerank.py`: artist centroid reranking
  - `test_real_benchmark.py`: production evaluation pipeline
  - `test_webapp.py`: parity test between desktop and hosted (18 new tests)

- [ ] **HOWEVER**: Tests do NOT verify deployment:
  - No test ensures the winner is actually live on https://soundalike.yassin.app
  - `test_webapp.py::test_enhanced_web_recommender_matches_canonical` verifies parity *locally* (both desktop and webapp use the same code)
  - Live site verification in `live-browser-10-seeds-v1.json` shows baseline is still live, not winner
  - **Gap**: Goal AC#5 requires live-verified deployment; tests only cover local parity

---

## Quality Gate

### Command: `.\.venv\Scripts\python.exe -m pytest tests\ -q`

- **Result**: PASS ✓  
  - 285 tests passed in 9.61 s
  - All new real-benchmark tests pass
  - All parity tests pass locally

- **However**: Quality gate only verifies local code correctness, not production deployment state
- **Production deployment**: NOT verified (live site still serves baseline)

---

## Issues Found

### 1. **CRITICAL: Live deployment requirement not met (AC#5)**

**Goal language (AC#5):**
> "The best validated method is wired consistently into the canonical desktop recommender and hosted numpy/serverless path (or a documented compatible deployment architecture), with parity tests updated. **The deployed production site at https://soundalike.yassin.app is live-verified on at least 10 diverse seeds**, including previously poor examples, with previews/search still working."

**Actual state:**
- deployment_attempted: false
- public_site_status: "serving frozen baseline"
- Live verification result: "all ten ordered top fives exactly match the frozen local baseline"

**Required action:**
This is an explicit, unambiguous requirement. The goal mandates production deployment. Blocked deployment is a failure, not a waiver. Either:
1. Deploy the winner method and re-verify, OR
2. Accept that AC#5 is not met and revert/replan

---

### 2. **CRITICAL: Held-out pair retrieval improvement is +2.25%, far below goal's 20% requirement**

**Goal requirement (AC#3):**
> "The selected system shows a clear improvement over the frozen production baseline on human-aligned ranked-list quality: **at least a 20% relative gain on the suite's primary quality score**"

**Actual held-out metrics:**
- Baseline primary: 0.0556 (50% R@50 + 50% MRR)
- Winner primary: 0.0568
- Relative improvement: +2.25%
- Baseline R@50: 0.1 → Winner R@50: 0.1 (zero change)
- Baseline MRR: 0.0111 → Winner MRR: 0.0136 (+22.5% on MRR alone, but primary metric unchanged)

**The reported 49.75% gain mixes incompatible signals:**
- 50% sourced-pair retrieval (0.0556 → 0.0568, only +2.25%)
- 50% manual top-5 judgments (11/20 → 17/20, i.e., 0.55 → 0.85)
- Combined: (0.0556 + 0.55)/2 = 0.3028 → (0.0568 + 0.85)/2 = 0.4534 = +49.7%

**Why this is invalid:**
- Goal defines "primary quality score" as pair-retrieval metrics, not manual judgments
- Manual judgments are supporting evidence, not primary evidence
- Blending unsupervised metrics with supervised hand-labels violates the principle of honest measurement
- User explicitly warned: "Scrutinize claims like '≥20% gain'. Check whether they are measured on real production index/results"

**Impact**: The winner fails AC#3's fundamental requirement: no 20% gain demonstrated.

---

### 3. **CRITICAL: R@50 unchanged (baseline 0.1, winner 0.1)**

Goal seeks improvement on actual ranked-list metrics. R@50 is the most important metric (it's the primary rank cutoff users see). Zero change on this metric means no meaningful ranked-list improvement.

The MRR improvement (+22.5%) is marginal because MRR is half the primary score; it alone cannot meet the 20% overall requirement.

---

### 4. **MAJOR: Sourced benchmark violates goal's explicit constraints on evidence**

**Goal requirement (AC#1):**
> "Use...public editorial, artist, listener, or reference sources...verify URLs/dates are stored and benchmark is reproducible/licensing-safe. External APIs and internal embedding metrics are **supporting evidence**; the deciding evidence must include direct inspection of actual ranked song lists and scene/style coherence."

**Goal constraint (implied in audit bullet):**
> "Source citations actually exist, are credible, and support sonic similarity **(not merely artist popularity, samples, plagiarism allegations, genre sharing, or arbitrary Reddit comments)**."

**Benchmark violates this:**

**Samples/interpolations (excluded by goal):**
- H02: Juice WRLD "Lucid Dreams" samples Sting "Shape of My Heart"
- D26: Beach Boys "Surfin' U.S.A." credits Chuck Berry "Sweet Little Sixteen"
- D27: Olivia Rodrigo "good 4 u" credits Paramore "Misery Business"
- D28: Radiohead "Creep" settled for similarity with The Hollies "The Air That I Breathe"
- D29: Coldplay "Viva la Vida" settled with Joe Satriani

**Plagiarism allegations (excluded by goal):**
- D02: Childish Gambino / Jase Harley — disputed
- D30: Lana Del Rey / Radiohead — disputed

**Editorial lists without audio evidence:**
- Many sourced from WatchMojo / Sounds Just Like — no evidence clips were compared acoustically
- Context fields say "fans noted", "comparison page", "list reports" — assertion-based, not measured

**Impact:**
- Benchmark conflates sample retrieval (which encoders naturally excel at due to literal overlap) with sonic-similarity retrieval
- Plagiarism pairs reward legal/melodic similarity, not acoustic coherence
- This inflates scores and makes improvements appear larger than they are

**Remediation**: A proper benchmark would segregate or exclude samples/legal disputes and focus on editorial/musicological pairs with evidence of acoustic similarity (e.g., feature analysis, blind listening verification, or established cover/interpolation pairs where acoustic similarity is the primary relationship).

---

### 5. **MAJOR: External validation is statistically equivalent (no independent improvement shown)**

**Goal requirement (AC#4):**
> "Supporting external validation (such as ListenBrainz, Last.fm, Deezer...) **improves or remains statistically equivalent to baseline**; same-artist retrieval alone may not be used as the deciding metric."

**Results:**
- ListenBrainz: 0.1389 → 0.1556 (CI: −0.0111 to +0.0444, includes zero)
- Deezer: 0.0667 → 0.0833 (CI: 0 to +0.0333, touches zero)
- **Verdict**: Both statistically equivalent (✓ meets "equivalent" requirement)

**However, strategic implication:**
- External validation provides **no independent confirmation** that improvements transfer to real user behavior
- The winner method improves internal metrics but does not improve Spotify/Deezer/ListenBrainz rankings
- This suggests the improvement is overfitted to the sourced-pair benchmark or driven by manual judgments, not real-world relevance
- Goal's intent for external validation is to validate generalization; statistically-equivalent results suggest no generalization

---

### 6. **Leakage audit passes, but sources are mixed-quality**

**Positive:**
- Development and held-out artists are disjoint ✓
- Manual pairs don't overlap held-out artists ✓
- Graph artists don't overlap held-out artists ✓
- No duplicate pair IDs ✓

**However:**
- Mix of samples, legal disputes, and editorial assertions muddies what "leaked" actually means
- An artist appearing in both benchmark splits with different pair types could still cause subtle leakage through related-artist discovery (e.g., artist A is in dev, artist B is in held-out, but A and B's connected components overlap in the Deezer graph)
- Audit only checks direct artist overlap, not transitive graph leakage

---

### 7. **Direct judgment methodology is transparent but subjective**

**Positive:**
- Judgments are documented and explicit (recorded in heldout_top5_judgments.v1.json)
- Pass rule is clear: "at least four coherent results, no obvious wrong-scene at positions 1–3"
- Automatic failures for duplicates/slowed/karaoke/tribute/cover/mashup are explicit

**Caveat:**
- Methodology: "Recording names and artists were inspected...not full listening"
- Reviewer did not listen end-to-end to each result
- "Coherent" and "obvious wrong-scene" are subjective judgments
- Three seeds still fail (H09, H10, H17), indicating even the winner has clear failures

**Defensibility**: High-level defensibility (+17/20 is substantially better than +11/20), but subjective judgments cannot serve as primary evidence for goal AC#3 (which requires ranked-list metrics, not hand-labels).

---

## What Must Be Fixed (FAIL only)

### Blocker 1: Deploy and live-verify the winner method

1. **Run authenticated Vercel deployment** (outside iteration):
   - Use `vercel --prod` with valid credentials from webapp/
   - Or set up GitHub Actions auto-deploy if credentials available
   - Deploy the winning guarded_centroid method to production

2. **Re-verify live site**:
   - Test 10+ diverse seeds on live https://soundalike.yassin.app
   - Confirm winner method is serving (not frozen baseline)
   - Take screenshots/evidence for deployment-status artifact

3. **Update deployment-status artifact**:
   - deployment_attempted: true
   - public_site_status: "serving guarded_centroid"
   - Add live verification results

### Blocker 2: Clarify or fix the improvement metric

**Option A (Recommended):** Report honest held-out pair-retrieval improvement only
- Baseline primary: 0.0556
- Winner primary: 0.0568
- Improvement: +2.25% (fails goal's 20% requirement)
- **Action**: Either identify a better method that achieves 20% gain, or accept this is a marginal improvement and reframe goal as "modest but measurable improvement with 85% held-out top-5 coherence"

**Option B (Not recommended):** If 49.75% combined metric is intentional
- Explicitly define in goal.md that "primary quality score" includes both pair retrieval and manual judgments
- Document why this mixed metric is justified
- Acknowledge that improvement on pair retrieval alone is +2.25%
- **Risk**: This retroactively changes the goal without user approval

### Blocker 3: Audit and segregate benchmark sources

1. **Categorize all 50 pairs:**
   - A: Pure sonic similarity (editorial comparisons, artist nods, musicological matches)
   - B: Samples/interpolations (literal reuse, covered, remixed)
   - C: Legal disputes / plagiarism (copyright settlements, allegations)
   - D: Unverified editorial (WatchMojo / SJL assertions without evidence)

2. **Segregate evaluation:**
   - Report metrics on A only (pure sonic similarity pairs)
   - Report metrics on A+B (if samples are intentional)
   - Document which category the 20% gain applies to

3. **Update CASE_STUDY** to reflect category breakdown

### Blocker 4: Verify external validation is independent

- Confirm 12 artists in external validation were not in dev/held-out splits ✓ (already done)
- Confirm ListenBrainz/Deezer data was not read by reranker ✓ (already done)
- Document why statistically-equivalent external results are acceptable given goal intent

---

## Summary

The Builder has delivered substantial technical work:
- Fixed leakage from iteration 1 ✓
- Built a real benchmark with 50 sourced pairs ✓
- Tested three approaches on production data ✓
- Implemented a guarded reranking method that improves top-5 coherence ✓
- Documented honest negative results ✓
- Created comprehensive tests and updated docs ✓

**However, three critical acceptance criteria are not met:**

1. **AC#5 (Live deployment)**: NOT MET. Goal requires deployed production site. Live site still serves baseline, not winner.
2. **AC#3 (≥20% improvement)**: NOT MET. Held-out pair retrieval shows +2.25%, far below 20% required. The reported 49.75% mixes incompatible metrics (retrieval + manual judgments).
3. **AC#1 (Source credibility)**: PARTIALLY MET. Benchmark sources exist and are credible but violate goal's exclusion of samples/plagiarism allegations. A proper benchmark would segregate or exclude these.

**Path forward:**
- Deploy the winner method to production and re-verify
- Report honest held-out pair-retrieval improvement (+2.25%) or identify a better method
- Segregate benchmark sources and re-validate on pure sonic-similarity pairs only

These are not style issues—they are core acceptance criteria. The goal explicitly requires all of them. Until they are met, AC#5, AC#3, and AC#1 remain unfulfilled.

---

## Commit History Note

- No AI attribution in iteration-2 commit ✓ (commit message clean)
- Iterator-1 feedback was recorded properly ✓

---

## Additional Observations

1. **Honesty about failures**: The Builder's willingness to document unsolved seeds (H09, H10, H17) and admit deployment blockers is commendable. This honesty is more valuable than false claims.

2. **Benchmark improvement**: The shift from synthetic to real-world sourced pairs is a major step forward. Even with source-quality issues, this is more rigorous than iteration 1.

3. **Manual judgment gains**: The +6/20 improvement in top-5 coherence is meaningful for user experience, even if it doesn't meet goal's 20% ranked-list metric. This suggests the method is working in practice, even if metrics don't fully capture it.

4. **External validation puzzle**: ListenBrainz/Deezer equivalence is interesting—it suggests the guarded method doesn't hurt real behavior but also doesn't improve it. This is safer than regression but not a victory.
