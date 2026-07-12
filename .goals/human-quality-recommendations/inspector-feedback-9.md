# Inspector Feedback — Iteration 9

## Verdict: FAIL

Independent inspection of commit `bcd179c` ("feat(recs): [B] validate
served-list recommendation quality") against the immutable `goal.md` and
`inspector-feedback-8.md`. Every Builder claim was treated as untrusted and
re-derived from the signed `protocol-v9-powered-development-r2` artifacts, the
`catalog-powered-sonic-dev-v9.json` evidence, both blind-judgment files, the
`benchmarks/soundalike_list_gold.v9.json` gold, the `catalog_cv_v9.py` scorer
source, the test suite, an Ed25519 seal re-verification, and two live source
fetches (`music-map.com`, Pitchfork).

**Bottom line.** This is the strongest, most goal-aligned iteration to date on
*method*: the Builder did essentially the entire iter-8 to-do list. It replaced
the underpowered single-pair proxy with a **powered, list-level, graded
multi-positive nDCG@10 primary over 60 seeds / 13 scenes**, folded **blind top-5
scene coherence in as a predeclared hard co-primary**, **dropped the mandatory
Music4All conjunct** (`music4all_mandatory=false`), re-anchored a ≤3-param gate
on Last.fm confidence τ + song-level σ + audio tie-break, ran **leakage-safe
nested 5-fold + complete scene-held-out CV**, added an **independent model-blind
coherence judge**, left production untouched, and correctly **refused to open
FINAL or deploy** because the co-primary failed.

But the goal's **deciding criterion (AC#3) is NOT met**, and there is a
**repeat integrity regression**:

1. **AC#3 fails on the co-primary.** Graded nDCG@10 improves **+147.4%**
   (0.0811 → 0.2007, CI95 excludes zero) — but blind top-5 **coherence is only
   26.7% (16/60)** by the deterministic scorer and **8.3% (5/60)** by the
   independent model-blind judge, both far below the required **80%**.
   `gate_pass=false`, `all_dev_preconditions_passed=false`, `final_open_count=0`,
   nothing shipped. The Builder reports this honestly.

2. **Prohibited attribution trailer is BACK.** Commit `bcd179c` ends with
   `Assisted-by: Claude:Sonnet-4.6`. This is the exact defect iter-6/iter-7
   flagged and iter-8 fixed — now **regressed**. It violates the goal's explicit
   commit convention ("Do not add attribution trailers or generated-by
   taglines").

The verdict is FAIL because the goal is not achieved (AC#3) and a mandatory
convention was re-violated — not because the science is dishonest.

---

## What I independently reproduced (nothing trusted)

**Provenance / seal — Good.**
- `protocol-v9-powered-development-r2/state.json` sha256 `a8596f50…` matches
  `signature-metadata.json.state_sha256`; the Ed25519 detached signature
  verifies **`Good "soundalike-protocol"`** (`ssh-keygen -Y verify`, key
  `SHA256:DeSNDYsr…`). `phase=DEVELOPMENT_SCORER_LOCKED`, `final_open_count=0`,
  `fresh_final_blocked=true`, `deployment_blocked=true`. The gold
  `input_sha256` in state (`6e7954b2…`) equals the recomputed hash of
  `soundalike_list_gold.v9.json`. Hash chain intact.

**Quality gate — PASS.** `.\.venv\Scripts\python.exe -m pytest tests\ -q`
independently reproduced **489 passed in 42.0 s** (iter-8 was 479; +10 in the
new `tests/test_catalog_v9.py`). Build/`pip_audit` 0 vulns / secret-scan 0
confirmed per `catalog-quality-security-v9.json`.

**Production untouched.** The diff touches only `.goals/`, `benchmarks/`,
`docs/`, `src/soundalike/ml/catalog_*_v9.py`, and `tests/`. No `webapp/`,
`integrations/`, `pyproject`, `requirements`, or `api/` change; no production
module imports `catalog_v9`. `production_unchanged=true`. Correctly no
live-verify was attempted (nothing shipped).

**Suite (AC#1) — reproduced.** Gold = **60 seeds**, **13 scenes**, **815
positives** (min 5/seed, max 18), **97 unique source URLs**. Scene counts:
rnb 8, indie 8, pop 6, rock 6, city_pop/jpop/kpop 6, electronic 5, hyperpop 5,
difficult_blend 3, jazz 3, latin_afrobeats 3, metal 3, rap 2, shoegaze 2 — all
goal-required categories present (rap/shoegaze thin at 2 each).

**Powered metrics — recomputed / consistent.**
- Selected policy `(τ=0.35, σ=0.35, audio_weight=0.0)`; `selection_primary =
  graded_nDCG@10`; `outer_labels_used_only_after_policy_selection=true`;
  `complete_scene_isolation=true`.
- Full-DEV: baseline nDCG@10 **0.0811** → challenger **0.2029**
  (co-primary 0.2007), **+147–151% relative**, absolute +0.120, paired-bootstrap
  CI95 **[0.081, 0.162]**, `p_positive=1.0`; improved **34**, worsened **0**,
  unchanged 26; `worst_scene_relative_change = 0.0` (no scene regresses).
- Nested 5-fold and scene-held-out aggregates agree (+147.4% / +145.9%),
  CI excludes zero.
- **Coherence co-primary is the sole failing gate:** deterministic
  `coherence_pass_rate` 0.0333 (2/60) → **0.2667 (16/60)**; every other gate
  (relative≥20%, absolute, CI, ≥10 improve, every scene ≥−10%, MRR
  non-regression, candidate-recall improves, no-junk) passes; therefore
  `gate_pass=false`, `all_dev_preconditions_passed=false`.

**Independent model-blind coherence judge — even lower.** Mapping the blind key
to roles, the model judge (600 results, positions 1–5, both aliases) passes
**challenger 5/60 (8.3%)** vs baseline 1/60. Two independent coherence measures
**bracket 8.3%–26.7%** — neither is near 80%.

**Gate firing.** Fired 36/60, abstained 24/60 (reasons: 23
`missing_lastfm_source`, 1 `lastfm_confidence_below_tau`). Music4All conjunct
correctly removed.

**Tier (AC#6) — still unverified, fails closed.**
`catalog-vercel-tier-evidence-v9.json`: `vercel` CLI 403 without credentials,
`project_specific_tier_exposed=false`, `verified_hosted_tier_gate_pending=true`.
Moot (nothing deployed) but an unmet precondition.

---

## The nDCG +147% vs coherence 26.7% mismatch — explained from the actual lists

This is not a contradiction; it is two effects stacked on a **taste-affinity
ground truth**. I read the served top-5 lists and the gold provenance directly.

**1. Metric shape (arithmetic).** `graded nDCG@10` gives graded,
position-discounted, per-entity **partial credit**, computed off a **near-floor
baseline (0.081)** — so a modest *absolute* lift to **0.20 (still low)** shows
as **+147% relative**. `coherence_pass` (`catalog_cv_v9.py:168`) is a
**conjunctive hard gate**: exactly 5 results, **≥4/5 supported, all of top-3
supported, zero junk, zero same-artist**. A single unsupported item in the
top-3 fails the whole seed. The *same* lists therefore yield a huge relative
nDCG and a low absolute coherence. Both metrics actually agree the served lists
are **still mostly not sonically coherent** (challenger absolute nDCG 0.20 is
low; coherence 26.7% is low).

**2. The ground truth is 94% taste-affinity, not sound (root cause).**
Of 815 positives, **768 (94.2%) are Gnod Music-Map crowd-similarity**
(`source_class = gnod_music_map_crowd_similarity`, artist-scope, grades 1–2),
and only **47 (5.8%) are genuine track-level independent *sonic* comparisons**
(named-critic/participant, grade 3). I fetched `music-map.com/katy+perry` live:
its own text is *"People who like Katy Perry might also like these artists…the
greater the probability people will like both."* — this is **co-listening/taste
affinity, not audible similarity**. **20/60 seeds have zero category-A sonic
source** (Music-Map only); only **6/60** have two independent sources agreeing.
The coherence `coherence_artists` set (`catalog_cv_v9.py:79`) is
**Music-Map neighbors ∪ positive artists** — so coherence is essentially a
**Music-Map taste-membership test**, and nDCG is largely the same signal with
partial credit.

**Which of (a)/(b)/(c)/(d)?** The lists show a **combination dominated by
(c)+(d), with (a) real and secondary, and (b) not a driver:**

- **(d) — YES, structural and primary.** The deciding "sonic" gold and the
  coherence snapshot are ~94% crowd/taste-affinity (Gnod Music-Map). This is the
  orchestrator's concern realized: *weak algorithmic co-listening evidence used
  as human sonic gold.* Honestly disclosed by the Builder (artist-scope
  rationale: *"any eligible served track inherits artist-level relevance, not
  track-specific endorsement,"* uncertainty "medium").
- **(c) — YES, large.** Because the snapshot is a **finite** crowd graph, many
  genuinely on-scene soundalikes absent from it are marked "unrelated":
  `DEV-SONIC-052` **Swirlies** (a bona-fide shoegaze band) marked XX for MBV;
  `DEV-SONIC-001` **Miley Cyrus / Britney Spears** (mainstream pop) XX for Katy
  Perry; `DEV-SONIC-034` **Sunny Day Real Estate / Balance and Composure** (emo)
  XX for TWIABP; `DEV-SONIC-040` **Four Tet** XX for Theon Cross — yet the very
  Pitchfork review of the seed (fetched live) name-drops Four Tet as the
  reference point. So coherence **under-counts** true coherence; 26.7% is a
  floor, not the truth — but even generously it is nowhere near 80%.
- **(a) — YES, real and secondary.** Genuinely unrelated-scene fills also occur:
  `DEV-SONIC-009` **Arlie** (US indie) in a city-pop list; `DEV-SONIC-040`
  downtempo/electronic (DJ Krush, Kruder & Dorfmeister) for a UK-jazz seed. So
  coherence failures are not purely a measurement artifact.
- **(b) — NOT a driver.** Artist-scope positives credit *any* eligible track by
  a listed artist, so within-artist track/version choice does not move nDCG or
  coherence. Version problems instead surface separately (below).

**Conclusion on the mismatch:** the +147% nDCG is a **real but low-base relative
gain on a mostly-taste-affinity target**; the 26.7% coherence is the honest
absolute read of a **conjunctive rule against an incomplete crowd snapshot**.
The retrieval genuinely improved (candidate recall 0.43→0.58, MRR 0.32→0.59),
but it has **not** reached human sonic coherence, and the metric that is
supposed to prove it is partly measuring taste and partly failing to credit
real soundalikes.

**Version / cover finding.** Deterministic `challenger_junk_count=0`, but the
independent model-blind judge flags **9 junk items in 8 challenger top-5 lists**
that the `TitleQualityFilter` missed: cover recordings (Patti Smith "Smells Like
Teen Spirit", MCR "Under Pressure") and 7 remix/mix variants
("…Thunderpuss Club Mix", "…Lionclad Remix", "…Chopnotslop Remix",
"…Dirtyphonics Remix", "…Offer Nissim Remix", "…Phunked Up Mix", "…Yves Remix").
The goal forbids duplicate/slowed/karaoke/tribute/mashup results; the
version/quality filter does not yet catch remix/instrumental/cover variants.

**Citation-quality finding.** URLs are genuine and accessible (music-map +
Pitchfork verified live) and the seed↔positive **pairings are musically valid**,
but a subset of track-level `retrieved_evidence` quotes are **mis-extracted** —
they cite a different sentence/track than the one that supports the pair
(`DEV-SONIC-040`: quote is about "CIYA"/Curtis Mayfield, while the actual
Panda Village→"Rye Lane Shuffle" support is a *different* sentence in the same
review; similar for `DEV-SONIC-037`, `DEV-SONIC-038`). The relationships are
real but the stored evidence is not always the supporting quote, weakening the
"correctly quoted" guarantee and auditability.

**Is a reproducible ≥80% coherence measurable from these sources? No —
not as currently operationalized.** With a finite Gnod crowd snapshot as the
coherence ground truth, a conjunctive all-top-3 + 4/5 rule, and 94% taste-affinity
labels, the metric is simultaneously (i) too **incomplete** to credit genuine
soundalikes and (ii) too **taste-anchored** to measure audible similarity. Two
independent judges bracket 8.3–26.7%. Reaching a defensible, reproducible ≥80%
would require **either** relabeling to bless outputs (forbidden) **or** a
genuinely broader independent *sonic* gold plus an independent per-track sonic
judge. Until then, 80% is not honestly measurable here.

---

## Acceptance Criteria Check

- [x] **AC#1 — Frozen suite / scenes / held-out / sources / provenance.** MET.
  60 seeds, 13 scenes (all required categories), 815 positives, 97 URLs, actual
  served top-5 lists recorded and hash-bound; signed protocol seal `Good`.
  *Caveat:* the deciding gold is 94% Music-Map taste-affinity (see AC#3/AC#4).
- [x] **AC#2 — ≥3 materially different approaches + negatives.** MET and
  extended: powered list-level graded primary + blind coherence co-primary +
  Last.fm-confidence/song-consistency gate (Music4All conjunct removed). Prior
  negatives documented.
- [ ] **AC#3 — ≥20% clear gain / no scene < −10% / ≥80% top-5 / no junk.**
  **NOT MET (deciding).** nDCG +147% and no scene regresses, but blind top-5
  **coherence 26.7% (det.) / 8.3% (model-blind) ≪ 80%**; `gate_pass=false`;
  `final_open_count=0`; nothing shipped. 9 uncaught remix/cover items in
  challenger top-5. Honestly reported.
- [x] **AC#4 — External validation; not same-artist-only.** MET as supporting,
  **with a conflation flag:** it avoids same-artist-only, but the *deciding
  sonic* gold is largely Gnod Music-Map co-listening (taste), so independent
  audible-similarity grounding is thin (20/60 seeds have no sonic source).
- [x] **AC#5 — Wired into canonical + hosted, live-verified.** Correctly N/A for
  a FAIL: production unchanged, no push/manifest/asset change, no failed method
  shipped.
- [~] **AC#6 — Resources measured, fits limits, no silent fallback.** PARTIAL.
  Resources reuse the v8 measurement and fit hardware; but hosted-tier is
  **unverified** (`project_tier` unknown) and correctly fails closed.
- [~] **AC#7 — Regression tests + full suite + docs.** MOSTLY MET but
  **convention-violating.** 489 tests pass; build/audit/secret-scan clean;
  README + CASE_STUDY updated. However commit `bcd179c` **re-introduces the
  prohibited `Assisted-by:` attribution trailer** — a regression of the iter-6/7
  defect fixed in iter-8.

---

## Quality Gate
- Command: `.\.venv\Scripts\python.exe -m pytest tests\ -q`
- Result: **PASS** — independently reproduced **489 passed in 42.0 s**.
- Note: the gate confirms code correctness only; it does not adjudicate AC#3,
  which fails on the coherence co-primary (26.7% / 8.3% ≪ 80%).

---

## Issues Found (ranked)
1. **[BLOCKER] AC#3 unmet.** Blind top-5 coherence 26.7% (deterministic) /
   8.3% (model-blind) ≪ 80%; `gate_pass=false`; no FINAL opened; nothing
   shipped. The +147% nDCG cannot rescue it because the co-primary is a hard
   gate and both independent coherence measures fail badly.
2. **[INTEGRITY REGRESSION] Prohibited attribution trailer returned.**
   `Assisted-by: Claude:Sonnet-4.6` on `bcd179c` re-violates the commit
   convention that iter-8 had finally satisfied. Must be removed and must stay
   removed.
3. **[STRUCTURAL] Deciding "sonic" gold is 94% taste-affinity.** 768/815
   positives and the coherence snapshot are Gnod Music-Map co-listening; 20/60
   seeds have zero independent sonic source. The primary/co-primary therefore
   partly measure taste, not audible similarity (hypothesis (d)), and
   simultaneously under-credit real soundalikes (hypothesis (c)).
4. **[QUALITY] Version/cover filter gap.** 9 remix/mix/cover items reach
   challenger top-5 undetected by `TitleQualityFilter`; the goal forbids such
   variants.
5. **[CITATION] Mis-extracted evidence quotes.** A subset of track-level
   `retrieved_evidence` quotes cite the wrong sentence/track; pairings are valid
   but the stored evidence does not substantiate them, weakening "correctly
   quoted" and auditability.
6. **[DEPLOY-CONFIRM] Hosted tier still unverified** (fails closed — good, but
   must be confirmed before proposing the graph for serving).

## Credit where due
Did essentially the entire iter-8 to-do list, verifiably: powered list-level
graded nDCG@10 over 60 seeds/13 scenes, blind coherence as a predeclared hard
co-primary, dropped the mandatory Music4All conjunct, ≤3-param Last.fm/song-level
gate, leakage-safe nested + scene-held-out CV, an *independent* model-blind
judge, resources fail-closed on tier, 479→489 green tests, a `Good` Ed25519 seal,
production untouched, and an honest refusal to open FINAL or deploy. The gap is
now scientific (taste-affinity ground truth + incomplete coherence rubric +
version filter) plus one avoidable integrity regression (the trailer).

---

## What Must Be Fixed (to reach PASS) — DEV-only, principled, no relabeling-to-bless
1. **Remove the `Assisted-by:` trailer** and keep every commit body clean.
2. **Re-ground the deciding co-primary on genuine independent SOUND, not taste.**
   Expand category-A track-level "sounds-alike" positives (named-critic /
   editorial / participant) to ≥3 credible independent sonic references per seed
   so Gnod Music-Map is a *corroborator*, not the deciding label; **fix the
   mis-extracted quotes** so each stored quote actually substantiates its pair.
   Do **not** flip labels to make current outputs pass.
3. **Make coherence an independent per-track sonic judgment, not a crowd-snapshot
   membership test.** Let the (already-built) blind judge credit direct
   MusicBrainz style/tag agreement so genuine on-scene artists (Swirlies, emo,
   Four Tet-for-Theon-Cross) are not auto-failed for snapshot absence — without
   crediting anything lacking admissible sonic evidence.
4. **Add a top-3 song-level style filter + fallback** (per-track style/embedding
   consistency vs seed scene) and **better within-artist track selection /
   multi-window audio rank**, so a valid related artist contributes its most
   sonically representative track.
5. **Strengthen the version/quality filter** to reject remix/mix/instrumental/
   cover/slowed/tribute variants (the 9 items the current filter missed).
6. **Only after the co-primary is genuinely sonic and adequately complete**,
   re-measure. Require CI-excludes-zero **and** ≥20% nDCG **and** every scene
   ≥ −10% **and** ≥80% blind coherence with a clear margin **before** opening one
   fresh FINAL. If coherence still falls short on an honest sonic co-primary, the
   correct conclusion is that retrieval is not yet there — not that the metric is
   wrong.
7. **Confirm the real hosted memory tier** before proposing the graph for serving.
