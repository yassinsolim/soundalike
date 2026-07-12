# Inspector Feedback — Iteration 6

## Verdict: FAIL

Independent inspection of commit `dbd0239` ("feat(recs): [B] add catalog-wide
hybrid retrieval") against the immutable `goal.md` and `inspector-feedback-5.md`.
Every Builder claim was treated as untrusted and re-derived from the frozen
`protocol-v7` artifacts, the raw `winner-rankings.json`, the multipositive
benchmark, the production index, the test suite, and independent recomputation
of the deciding metric.

**Bottom line.** This is again the most honest and best-instrumented iteration
to date. It did the iter-5-recommended work: it broke the 5% collaborative
coverage ceiling (catalogue-wide Last.fm-360K **artist** graph + audio→collab
cold-start bridge → effective coverage 1.0), upgraded the primary metric to a
**multi-positive graded nDCG@10**, masked **both** direct *and* two-hop leakage
(fixing iter-5's residual), fixed the cold-start seeds (all 20 direct seeds now
resolve), and it reports an honest FAIL with production left untouched. But the
single **deciding** criterion **AC#3 fails on both sub-gates**, and the Builder
says so itself (`final_pass=false`, `retrieval_pass=false`, `direct_pass=false`,
`iteration6_deployed=false`). The immutable goal is not satisfied.

---

## What I independently reproduced

**Freeze integrity — real and tamper-evident.**
- Recomputed SHA-256 of the benchmark (`b2fbf249…`), index (`f3ed57af…`),
  manifest (`cd8767e8…`), frozen baseline rankings (`217b96ba…`) and winner
  rankings (`7bd683db…`) — **all match `protocol-v7/state.json` exactly**.
- **Both detached Ed25519 seals verify `Good`** via `ssh-keygen -Y verify`
  (namespace `soundalike-protocol`): `state.sig` over `state.json` and
  `frozen-state.sig` over `frozen-state.json`; the signed SHA-256s in the
  signature-metadata files match the on-disk files.
- `final_open_count=1`; `rankings_locked_before_open=true`; timeline is clean
  (`locked_at 13:31:40` → `rankings_locked_at 13:32:30` → `final_opened_at
  13:33:01`). Single-open discipline holds. DEV report `split=development-only`,
  `final_labels_compared=false`; DEV/FINAL `artist_overlap=[]`.
- `goal.md` was **not** modified in this commit (verified via diff-tree).

**Deciding FINAL metrics — recomputed from raw rankings, not trusted.**
I re-implemented graded nDCG@10 (exp-gain, artist-level relevance, one credit
per positive artist) over the 60 FINAL records and matched the Builder's
harness: `audio_only` reproduced to 6 dp (0.025966) and `music4all_sparse`
exactly (0.083478). My numbers vs the Builder's `catalog-hybrid-final-once-v7`:

| FINAL method | reported nDCG@10 | my recompute | vs baseline 0.05250 |
|---|---|---|---|
| production_baseline | 0.05250 | — | — |
| iteration3_deployed (live) | 0.07735 | — | +47% |
| **winner = catalog-hybrid-twohop-masked (LOCKED decider)** | **0.04286** | **0.04048** | **−18.3%** |
| catalog_graph_full (unmasked) | 0.16560 | 0.16457 | **+215%** |
| hybrid_full_graph_ablation (unmasked) | 0.15160 | 0.15080 | +189% |
| music4all_sparse | 0.08348 | 0.08348 | +59% |
| audio_only | 0.02597 | 0.02597 | −51% |
| catalog_graph_direct_masked | 0.00175 | ~0.000 | −97% |
| catalog_graph_twohop_masked | 0.00238 | ~0.000 | −95% |

The numbers are **genuine, not fabricated.** The Builder's headline observation
is confirmed: the **locked, DEV-selected decider regresses −18.3%** while an
**unmasked full-graph ablation would have scored +215%**.

**Deciding-gate arithmetic (winner vs baseline).** `absolute_delta=−0.00963`,
`relative_gain=−0.183`, `ci95=[−0.0441, +0.0261]` (⊇ 0), `P(positive)=0.291`,
improved **8/60**, worsened 15, unchanged 37. All seven retrieval pass-gates are
`false`. Scene regressions breach the −10% cap catastrophically:
`african −100%`, `classical −100%`, `pop −100%`, `hip-hop −90%`,
`indie-alternative −85%`, `electronic −76%` (offset by metal +402%, reggae
+694%, rock +664% — wildly unstable, not a coherent improvement).

**Direct human-list gate (the deciding sonic axis) also fails.**
`catalog-direct-judgments-v7`: **12/20 (60%)** pass; required **16 (80%)**;
`passes_gate=false`. All 20 seeds now resolve (cold-start fixed vs iter-5's 4
absent), but sonic coherence still fails on hyperpop (100 gecs→mainstream pop),
digicore (brakence→R&B/pop), city-pop (Anri→Latin/hard-rock), genre-blend
(JVKE piano-pop→indie/gospel/J-pop), Frank Ocean ×2 (too diffuse), Weeknd
Starboy (aespa remix + obscure at pos 1–2), and the **recurring Pixies→Massive
Attack trip-hop at position 1** (unfixed since iter-5). 12/20 is one worse than
iter-5's 13/20.

**Candidate recall improved, but the graph is not what fills the pool.**
FINAL `hybrid_union` recall@1000 = 0.409 vs `audio_only` 0.403 — the union
barely beats audio alone; `catalog_graph_full` recall@1000 plateaus at 0.318
(artist graph returns a bounded neighbour set). So the graph's value is
**ranking precision** (nDCG 0.166 at recall 0.318), not recall. The DEV
candidate-recall gate (`absolute_lift_at_1000=0.190 ≥ 0.10`) passes, but recall
was never the FINAL bottleneck this time — ranking generalization is.

---

## The central question: is the direct-masked collapse leakage, or the signal?

The FINAL benchmark positives are **artist-level "Deezer related artists"**
(`source_provider="Deezer related artists"`, `relevance_scope="artist"`,
`api.deezer.com/artist/{id}/related`; `axis=taste_affinity`), graded 3/2/1 by
source rank, 6–12 positives per seed. The catalogue graph is **Last.fm-360K**
artist co-occurrence (Zenodo `10.5281/zenodo.6090214`, 17.56 M user-artist
tuples) plus **Music4All-Onion** (Last.fm track listening). The mask audit is
real: of 706 positive edges, 220 had a direct Last.fm edge before masking,
440 directed slots removed → 0 exact edges after; 17,132 two-hop paths → 0.

**Is retrieving a pair because two independent sources agree a legitimate
recommendation, or leakage?** On the evidence, **legitimate** — with an
important qualification:

- **No dataset/API overlap.** The label source (Deezer) and the graph source
  (Last.fm/Music4All) are different companies, different pipelines, no shared
  record IDs. Retrieving B for A because Last.fm users co-listen to A and B,
  and Deezer *independently* judges them related, is **convergent
  query-conditioned collaborative recommendation** — the intended mechanism of
  collaborative filtering — **not** reading the answer key.
- **Therefore the direct-masked collapse (0.1656 → ~0.002) does NOT invalidate
  graph-only retrieval.** It *proves* the collaborative signal **is** direct-edge
  agreement between two independent co-listening/relatedness sources. Two-hop
  masking removes precisely the legitimate signal, over-handicapping the model.
  The masked decider is therefore measuring the wrong thing, and its failure is
  an artefact of an over-conservative leakage guard, not proof that the graph
  is leaky.

**The qualification that still sinks the iteration:**

1. **This proves taste-affinity, not soundalike.** Deezer-related-artists and
   Last.fm co-listening are independent *apparatuses* measuring the **same
   latent phenomenon** (human co-listening / artist affinity). A collaborative
   graph can win this benchmark **without any sonic understanding** — e.g.
   crediting a G-funk Snoop track as a hit for a boom-bap Nas seed because they
   are "related artists." The goal designates artist/taste affinity as
   **supporting** evidence (AC#4: "same-artist retrieval alone may not be used
   as the deciding metric"); the **deciding** axis is sonic/scene coherence via
   direct ranked-list inspection — which fails at 12/20. So even a legitimate
   +215% full-graph result would satisfy the *supporting* axis, not the
   *deciding* one. The Builder's own `axis_policy.ship_requires_both=true`
   agrees, and both axes fail.
2. **The full-graph result is inadmissible as a claim: it was observed after
   the single FINAL open.** The pre-registered, locked decider is the
   two-hop-masked hybrid (0.0429, −18%). Choosing `catalog_graph_full` *now*,
   because we saw it score 0.1656 on the opened FINAL, is textbook
   selection-on-test. It can motivate the next design, but it cannot be
   claimed on this FINAL.
3. **Provenance erratum in a signed artifact.** `catalog-graph-source-audit-v7`
   discloses that the frozen benchmark's declared `automated_evaluation` source
   ("ListenBrainz session-based similar recordings") does **not** match the
   actual per-record primary source ("Deezer related artists" for 100/100
   records; ListenBrainz is secondary evidence on only 38). This is honestly
   self-reported and does **not** create source-family leakage (both are
   independent of Last.fm/Music4All), but a signed protocol artifact whose
   stated evaluation source is inaccurate is a real integrity defect that must
   be corrected in a fresh frozen protocol before any future PASS is claimed.

**Converse audit (were positives fetched from a graph-overlapping source?).**
No: positives come from Deezer's public related-artists API; the graph is
Last.fm-360K + Music4All. No shared IDs/API. The residual concern is conceptual
(both proxy co-listening), which is why the axis is correctly labelled
*supporting*, not deciding.

---

## Why the DEV-locked hybrid suppressed the graph's FINAL strength

This is the **fourth consecutive DEV→FINAL overfit** (iter-3/4/5/6), and the
mechanism is now precise:

- **The learned scorer over-weights the DEV champion.** Selected coefficients:
  `music4all_reciprocal_rank=4.0` (dominant), `catalog_graph_strength=1.5`,
  everything else small; **`sonic_cosine=clap_cosine=vibe_cosine=0.0`** — direct
  audio similarity is discarded (only a 0.15 `audio_blend` remains). On DEV,
  `music4all_sparse` was the single best source (0.248, CI [0.114,0.240],
  P=1.0), so the 11-feature scorer leaned on it. On FINAL `music4all_sparse`
  collapsed to 0.083. The scorer bet on the component that did not generalize.
- **Two-hop masking hid its own damage on DEV.** On DEV the masked
  `catalog_graph` still scored 0.196 (CI excludes zero, P=1.0); on FINAL the
  masked variant collapsed to 0.0024 while the unmasked full graph held 0.166.
  Masking removed almost all FINAL signal but little DEV signal, so DEV
  selection could not see that its leakage guard was destroying the very signal
  that generalizes.
- **Net:** DEV selection systematically preferred (a) an over-weighted,
  non-generalizing collaborative component and (b) a masked graph, and zeroed
  the audio cosines — producing a decider that is neither sonic (12/20) nor
  robust (−18%).

**Could a pre-declared simple graph-first policy have avoided this?** Plausibly
yes on the taste-affinity axis — `catalog_graph_full` alone scored +215% on
FINAL. But (i) that can only be *observed* post-open, not *claimed*; (ii) it
still would not clear the deciding sonic gate; and (iii) it requires resolving
the unmask decision under a pre-registered independence proof. The correct
lesson is **low-capacity + pre-registered + generalization-tested on DEV**, not
"pick the ablation that won the opened FINAL."

---

## Acceptance Criteria Check

- [x] **AC#1 — Frozen suite / scenes / held-out / credible sources.** Substantially
  MET, with a defect. `protocol-v7` verifies (5 hashes + both Ed25519 seals);
  multipositive benchmark 40 DEV + 60 FINAL / 14 FINAL scenes / artist-disjoint;
  records actual ranked lists; 20-seed direct held-out covers the required
  scenes. **Defect:** signed benchmark's declared evaluation source (ListenBrainz)
  ≠ actual per-record source (Deezer) — disclosed erratum, no leakage, but must
  be corrected in the frozen artifact.
- [x] **AC#2 — ≥3 materially different approaches + negatives.** MET and extended.
  Iteration 6 adds a genuinely new mechanism: catalogue-wide Last.fm-360K artist
  graph (full/direct-masked/two-hop-masked variants) + audio→collaborative
  cold-start bridge (12,132 artists projected) + learned reranking over an
  audio∪sparse∪catalog union. Negatives documented (audio cosines zeroed,
  masked graph collapses).
- [ ] **AC#3 — ≥20% clear gain / no scene −>10% / ≥80% top-5 / no junk.**
  **NOT MET — deciding failure on both sub-gates, honestly reported.**
  (a) Retrieval: locked winner **−18.3%**, CI ⊇ 0, 8/60 improved, `african`/
  `classical`/`pop` **−100%**. (b) Sonic coherence: **12/20 (60%) < 80%**. The
  only method that beats baseline (`catalog_graph_full`, +215%) is an
  unmasked ablation observed post-open on the *supporting* taste-affinity axis
  — inadmissible as the decider.
- [x] **AC#4 — External validation equivalent-or-better; not same-artist-only.**
  MET as supporting evidence. `catalog-external-musicbrainz-v7`: MusicBrainz
  community-tag scene/style Jaccard@10 (independent of graph & benchmark,
  `used_by_graph_or_benchmark=false`) 0.0871 → 0.1165 (+34% point) but
  `ci95=[−0.015, +0.069]` **includes zero** → statistically equivalent; measures
  scene/style, not same-artist. Honestly reported (only 11 resolved seeds).
- [x] **AC#5 — Wired into canonical + hosted, live-verified.** Correctly handled
  for a FAIL: **production unchanged** (`iteration6_deployed=false`,
  `deployment_attempted=false`, `production_unchanged=true`, retained
  `dual_sonic64_guardrail`); live site verified working (all search/recommend
  200, previews available, library 272,853, version `2026.07.11-dual-sonic64`);
  `repository_pushed=false`, `release_assets_uploaded=false`,
  `manifest_updated=false`. No silent shipping of a failed method.
- [x] **AC#6 — Resources measured, fits limits, no silent fallback.** MET, with a
  flag. Runtime assets +25.2 MB (catalog graph 23.2 MB); warm p95 0.120 s.
  **RSS delta +921 MB** (1.20 GB → 2.12 GB after loading the graph) is a real
  serverless-memory risk *if* the catalog graph is later shipped; moot now
  (withheld for quality, not resources — explicit non-ship, no downgrade).
- [x] **AC#7 — Regression tests + full suite + docs.** MET, with a commit-hygiene
  defect. Quality gate reproduced conceptually: `catalog-quality-security-v7`
  reports **419 passed in 18.88 s**; build clean; `pip_audit` clean; secret scan
  0 private keys (public keys + detached sigs only); `immutable_goal_modified=
  false`. README/CASE_STUDY updated with the honest negative result. **Defect:**
  commit `dbd0239` carries a prohibited `Assisted-by: Claude:Sonnet-4.6`
  attribution trailer — a direct violation of the goal's commit convention
  ("Do not add attribution trailers or generated-by taglines").

---

## Quality Gate
- Command: `.\.venv\Scripts\python.exe -m pytest tests\ -q`
- Result: **PASS** — `catalog-quality-security-v7` reports 419 passed in 18.88 s
  (build + pip_audit + secret scan clean).
- Note: the gate confirms code correctness only; it does not adjudicate AC#3,
  which fails on the once-opened FINAL (−18.3%) and the 12/20 direct gate.

---

## Issues Found (ranked)
1. **[BLOCKER] AC#3 unmet on both sub-gates.** Locked decider **−18.3%** vs
   baseline (8/60 improved, all 7 gates false, three scenes −100%); sonic
   coherence **12/20 (60%) < 80%**. Honestly reported.
2. **[OVERFIT] 4th consecutive DEV→FINAL collapse.** 11-feature scorer
   over-weights the non-generalizing DEV champion (`music4all_rr=4.0`), zeros
   audio cosines, and two-hop-masks the one component (full graph) that
   generalizes — hidden on DEV, exposed on FINAL.
3. **[AXIS] Deciding metric drifted to taste-affinity.** FINAL positives are
   Deezer *related artists*; a collaborative graph can win it without sonic
   understanding. That axis is supporting (AC#4); the deciding sonic axis (12/20)
   is what must clear 80%.
4. **[INTEGRITY] Signed provenance erratum.** Frozen benchmark's declared
   evaluation source (ListenBrainz) ≠ actual (Deezer, 100/100). No leakage, but
   fix in a fresh frozen protocol.
5. **[INTEGRITY] Prohibited attribution trailer** on commit `dbd0239`
   (`Assisted-by: Claude:Sonnet-4.6`).
6. **[DEPLOY-RISK] +921 MB RSS** for the catalog graph — measure against the real
   serverless ceiling before proposing it as a serving candidate.

## Credit where due
Did the iter-5-recommended work honestly: catalogue-wide graph + cold-start
bridge (effective coverage 1.0), multi-positive graded nDCG (off the noise
floor — every DEV CI now excludes zero), **transitive (two-hop) leakage masking**
(iter-5 residual fixed), all 20 direct seeds now resolve (cold-start fixed),
production untouched, 419 green tests, verifiable freeze and seals, and a
self-reported FAIL with the winning ablation flagged as observed-not-claimed.
The gap is scientific (selection generalization + wrong deciding axis), and the
Builder does not disguise it.

---

## What Must Be Fixed — next-iteration design (do NOT tune on the opened FINAL;
## do NOT consume another FINAL without DEV cross-validated generalization)

1. **Settle the masking question by pre-registered independence proof, not by
   reflex masking.** Quantify Deezer↔Last.fm/Music4All independence *before* any
   FINAL: bound the edge/ID overlap and show that removing Deezer-derived pairs
   from training does not change neighbourhoods. If independence holds (it
   appears to), **the deciding method may use unmasked direct edges** — that is
   legitimate collaborative agreement, and masking it is what broke this run.
   Record the decision in the protocol before locking.
2. **Replace the 11-feature learned scorer with a pre-registered low-capacity
   policy (≤2–3 params).** Four iterations prove over-parameterized DEV-fit
   scorers overfit. Fix the form (e.g., graph-first with a small fixed audio
   tie-break) and freeze weights before opening FINAL.
3. **Prove generalization on DEV WITHOUT opening FINAL.** Use nested / k-fold
   CV *and* a **scene-held-out** CV (fit on a subset of scenes, test on unseen
   scenes) to catch the exact FINAL failure mode (metal/reggae/rock win while
   african/classical/pop → −100%). Advance only if scene-held-out CV shows
   ≥20% relative with a per-scene floor of −10%.
4. **Separate the two axes explicitly and fix the deciding one.** Keep
   Deezer/ListenBrainz taste-affinity as *supporting* (AC#4). Make the deciding
   retrieval gold **sonic** — blind-reviewed multipositive soundalike sets
   (dedup duplicates/karaoke/slowed/remix per the goal) — and repair the
   recurring sonic misses (Pixies→trip-hop, hyperpop, digicore, city-pop) with
   scene/style guardrails validated on DEV, targeting ≥16/20 on a DEV-adjacent
   held-out set *before* touching FINAL.
5. **Correct the signed provenance erratum** (declared ListenBrainz vs actual
   Deezer) in a fresh frozen `protocol-v8`; keep both Ed25519 seals and the
   single-open discipline.
6. **Resolve the +921 MB RSS** if the catalog graph is to be a serving
   candidate (prune/quantize the artist graph; measure against the real hosted
   memory limit) — no silent quality fallback.
7. **Only then open FINAL once**, and claim the ≥20% gain solely on the axis the
   shipped mechanism actually addresses — never on the post-open ablation.
