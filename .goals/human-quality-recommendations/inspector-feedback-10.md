# Inspector Feedback — Iteration 10

## Verdict: FAIL

Independent inspection of commit `e272149` ("feat(eval): [B] add blinded human
soundalike evaluation") against the immutable `goal.md` and
`inspector-feedback-9.md`. Every Builder claim was treated as untrusted and
re-derived: I recomputed all protocol/list/state content hashes with **both** the
browser's canonical semantics and the Python `content_hash`, re-verified the
Ed25519 seal with `ssh-keygen -Y verify`, reproduced the MTAT 19/29 accuracy,
paired-bootstrap CI and Wilson CIs from the committed vectors, re-ran the
real-index version-quality audit (identical `content_sha256`), fed a **real
browser-produced export through the aggregator's validator**, and **drove the
evaluator end-to-end in a live Chrome session** (load → render → blind A/B →
shared-result dedup → rate → autosave → export+HMAC → resume).

**Bottom line — distinguish two things the goal cares about:**

1. **Evaluation infrastructure: VERIFIED and essentially READY.** This iteration
   correctly abandons the iter-9 taste-affinity gold and builds the honest path
   to the goal's deciding evidence: (a) an independent **human sonic calibration**
   on MagnaTagATune odd-one-out votes with a disciplined once-opened,
   artist-disjoint test; and (b) a **frozen, signed, blinded 60-seed listening
   evaluator + anti-proxy aggregation CLI** for the user and friends to rate the
   actual served commercial-track lists. All of this is real, reproducible,
   well-tested, and honest.
2. **Recommender quality: NOT demonstrated (AC#3 unmet).** No human ratings exist
   yet (`ratings_count_at_freeze=0`, `human_rater_exports_ingested=0`,
   `sonic_human_report_exists=false`). The MTAT learned method beats the
   incumbent only **19/29 vs 16/29** with a paired CI that **spans zero**
   (`P(Δ>0)=0.758`), so nothing was re-embedded, promoted, or deployed. The
   Builder **explicitly does not claim AC#3** and fails closed.

The verdict is **FAIL because the goal's deciding human-aligned evidence has not
yet been produced** — not because the science is dishonest or the build is
broken. This is the correct, honest state after iter-9. The infrastructure being
verified is a genuine step forward; it is not the same as the recommender being
good.

---

## What I independently reproduced (nothing trusted)

**Protocol freeze / signatures / list hashes — ALL GOOD.**
- `protocol-v10.json`, `served-lists-v10.json`, `state.json` each recompute their
  stored `content_sha256` under **both** the browser `docHash` canonical
  (sorted-key JS semantics, reimplemented) **and** the Python `content_hash`
  (`json.dumps(sort_keys)`), so browser-validated files also validate in the CLI.
- `protocol.served_lists_sha256 == lists.content_sha256` (hash-bound); state
  cross-refs protocol/lists; `state.evaluator_sha256` equals the actual
  `human_eval_v10.html` bytes (the evaluator is hash-pinned).
- `sha256(state.json) == signature-metadata.state_sha256`, and the detached SSH
  Ed25519 signature verifies **`Good "soundalike-human-eval"`**
  (key `SHA256:c9Vzptc7X6TbwM/tHyOvOt25dGF8BPjeEvC60/iw+Do`, matching the
  fingerprint claimed in the quality/security artifact).
- **Blinding intact / no key leak.** `git ls-files` shows only public keys,
  allowed-signers, signatures, and the blinded pack — **no method-role key and no
  private signer keys are committed** (`OPENSSH PRIVATE KEY` absent from all
  tracked files). The pack's lists carry only opaque `list_id` + `position`; a
  full-text scan finds no `method/baseline/challenger/score/policy` keys.

**Blinding / randomization — GOOD.** Per-session order is
`SHA256(pack_hash ∥ rater_id ∥ session_id ∥ label ∥ item_id)` for seeds and
lists; lists are surfaced as "A"/"B" by shuffled position; the role map lives
only in the gitignored private key. Confirmed live: the page states "no method
identity is available in this page or the public pack."

**Shared-result dedup — GOOD (verified live).** The `seen` set makes a track that
appears in both lists render its rating controls once; the duplicate shows
"Shared result: rated once; this judgment is accounted for in both lists." Seed 1
happened to have all five results shared across A and B, and the UI handled it
exactly. The aggregator mirrors this: ratings are keyed by `result_id`, so one
human grade is credited to every method position containing that track.

**localStorage privacy — GOOD (verified live).** Storage key is scoped to
`protocol_hash : anon-rater : session`; IDs are random; **zero external network
requests** were made (only the page load + inline `data:` SVG control icons). CSP
is `default-src 'none'; connect-src 'none'` (no fetch/XHR/upload), with
`media-src https:` only for previews. XSS surface is closed: all attacker-
controllable strings pass through `escapeHtml`/`escapeAttr`, and result/list IDs
are regex-validated `^[LT]-[a-f0-9]{20}$`.

**Export / import schema / signature — GOOD (verified live end-to-end).** Export
filters incomplete ratings, adds `duration_ms` + an HMAC over the canonical
payload; import re-checks protocol/list hashes and the HMAC before resuming. I
cleared the in-memory ratings, re-imported the export, and got "Valid partial
session resumed" with all ratings restored. The HMAC is honestly labelled as
**integrity, not authenticity** (the key travels in the export) — which is why
the aggregator additionally requires a collector Ed25519 approval.

**Aggregation CLI (`sonic_human`) — GOOD.** `_load_bound` hash-binds protocol +
lists + private key and requires roles `{production_baseline, challenger}`;
`_verify_collector_approval` requires a detached Ed25519 `.sig` per export
(identity `soundalike-human-rater`). **Anti-proxy enforcement**: any export whose
`source_kind`/`provider` or serialized body contains `lastfm/deezer/music4all/
gnod/model/proxy/editorial` is rejected; `source_kind` must be `human_listener`
and `provider` `standalone_local_evaluator`. Anti-fabrication: session-bounded
timestamps, positive durations, `interaction_ms ≤ duration+5s`, constant-time
HMAC compare. The inferential design is **correct**: within-rater A/B deltas are
averaged per seed and then bootstrapped over seeds (avoids pseudo-replication and
rater confounding); nDCG@5 is standard graded `2^g−1 / log2`; agreement is a
chance-corrected pairwise kappa. With no exports it **fails closed and deletes
the output**. I built a real export in the browser and confirmed the aggregator's
`_validate_export` **accepts** it (2 result ratings, 1 list rating, human_listener).

**MTAT provenance / license — HONEST.** No audio is committed or re-hosted; both
CSVs, three archive parts, extracted clips, mels and checkpoints stay under
gitignored `ml_data`; only URLs, schemas, counts, sha256s and aggregate scores
are committed. The absence of a dataset-wide audio license is stated, and
citation is provided (Law et al. 2009; MIREX AMS + Evalutron only as protocol
references, not as a claimed dataset). Download hashes are documented (I could
not re-verify the ~3 GB archives without downloading, but the record is
transparent).

**Vote parsing — CORRECT.** Odd-one-out = unique vote-max; ties excluded;
≥3 total votes required. Counts reconcile exactly: 533 rows = 307 accepted +
30 tied(≥3 votes) + 196 too-few(<3), and 446 unique-winner = 307 accepted +
139 too-few, 87 tied = 30(≥3) + 57(<3). Confidence = (winner−runnerup)/total.

**Artist-disjoint split — CORRECT and leakage-safe.** Weighted-Louvain artist
communities (resolution 1.0, seed 20260712) are assigned whole to train/dev/test
by exact DP, cross-community rows discarded, and a **hard invariant raises** if
any artist crosses a split. Result: 86/28/29 constraints, 97/35/35 artists,
`artist_overlap` empty for all three pairs, and clips also disjoint.

**Single-open TEST / method lock — CORRECT.** The representation set and the
material-win rule are written before any test access; the state guard raises if
`test_open_count != 0`; the method is selected on DEV and hash-locked
(`test_labels_compared=false`) before the single TEST open; `open_count=1`.

**Representation scores + 19/29 accuracy/CI — REPRODUCED EXACTLY.** From the
committed `correct_vector`s: MTAT-triplet+FMA **19/29 (65.5%)**, incumbent
artist-SupCon 16/29, LAION-CLAP 17/29, FMA-SupCon 16/29, vibe/DSP 13/29. Paired
`Δ=+0.1034`, 50k-iteration bootstrap CI95 **[-0.138, +0.345]**, `P(Δ>0)=0.75768`
— matched to ~1e-9 with the same seed. All Wilson CIs matched exactly. The
closest-pair odd-one-out prediction logic is correct.

**No unjustified catalog re-embedding — CONFIRMED.** `material_win = (Δ≥5pts AND
CI_low>0)` is **False** because `CI_low=-0.138`. Hence
`catalog_reembedded=false`, `catalog_reembedding_permitted=false`, no spec-cache
rebuild, no song tie-break change, no FINAL, no deploy. No embedding artifact
appears in the diff. This is an honest negative release decision.

**Version / canonical filter on the REAL index — REPRODUCED EXACTLY.** I re-ran
`quality_audit_v10` against `ml_data/deepvibe_index_v5.npz` and got an **identical
`content_sha256`** (`e0a8602758…`): 272,853 rows, **16,931 filtered (6.2%)**,
**0 false negatives** vs an 11,470-row explicit-derivative recognizer, **0 false
positives** vs 6 curated legitimate controls ("Cover Me"/Springsteen,
"Karaoke"/Drake, "Love X Love"/Benson, …), 4,353 same-artist canonical-preference
groups, `artist_specific_rules=false`. Query-aware handling is principled:
`is_eligible_for_query` gives a canonical seed only canonical results, lets a
derivative seed admit the *same* derivative classes (subset rule; mashups stay
strict), and `prefer_canonical` dedups within one artist/canonical-title without
any cross-artist/popularity rule. *Nuance:* the FN recognizer shares regex logic
with the filter, so "0 FN" is a strong **consistency** check rather than fully
independent human recall; the FP controls are genuinely independent.

**Quality gate — PASS.** `.\.venv\Scripts\python.exe -m pytest tests\ -q` →
**511 passed in ~22 s** (iter-9 was 489; +22, including targeted tests for vote
parsing, split disjointness, once-opened test, material-win rule, freeze/blind/
shared, anti-proxy rejection, deterministic dedup, and "bootstrap never pairs
different raters"). Build/`pip_audit` (0 vulns)/secret-scan (0) per the
quality/security artifact; new deps `networkx 3.6.1`, `laion-clap 1.1.7`.

**Attribution trailer — CLEAN (iter-9 regression fixed).** The commit body is
exactly `feat(eval): [B] add blinded human soundalike evaluation`; no
`Assisted-by`/`Co-authored-by`/`Generated-by`.

---

## Live browser verification (Chrome)

Served the frozen `human_eval_v10.html` + real protocol/lists over localhost and
drove it:
- **Load/freeze validation** accepted the real hash-bound, RANKINGS_LOCKED files
  ("Locked files validated. Autosave is active.").
- **Render**: Seed 1/60 ("Fightboat — TWIABP", scene indie), two blinded lists,
  MIREX 3-class + optional 0–10 + junk/version per result, whole-list coherence +
  top-3-unrelated per list.
- **Shared dedup**: verified (all-5 overlap handled).
- **Audio**: results with previews render `<audio preload="none">` pointing at
  real `https://cdnt-preview.dzcdn.net/...` URLs; playback is wired.
- **Autosave/Export/Resume**: state saved to localStorage; export HMAC round-trips
  and is accepted by the Python aggregator; clear+import restores the session.
- **Console**: no JS/CSP errors (one benign a11y hint about data-attribute inputs).

---

## Production status

- **Deployed recommender: UNCHANGED.** No push (14 commits ahead of origin),
  `recommender_deployed=false`, version retained `2026.07.11-dual-sonic64`; live
  site `https://soundalike.yassin.app/` returns HTTP 200. No model/index/embedding
  change.
- **BUT the serving-time version/quality FILTER did change in both production
  paths** — desktop `quality_filter.py` (imported by the canonical
  `deepvibe.py` recommender) and hosted `webapp/api/_reco.py` — to reject
  remix/mix/edit/cover variants (directly addressing iter-9's 9 uncaught items).
  Parity is covered (`test_webapp.py` mask + `test_quality_filter.py`), and the
  change is honestly documented in `CASE_STUDY.md`. `catalog_policy_v9.py`'s edit
  is DEV-scorer-only. **So `production_changed:false` is accurate for
  deployment/model-version but not literally for serving-filter source**: anyone
  running the desktop app now filters more aggressively. If this filter is meant
  to ship, it needs the AC#5 hosted live-verification; if not, it should be
  framed as "staged, not deployed" rather than "production unchanged."

---

## Is the evaluator ready to produce the deciding evidence?

**Infrastructure: yes** — verified end-to-end, blinded, signed, anti-proxy,
privacy-preserving, resumable, and reproducible. **Two practical readiness
caveats** before the user + friends can produce *trustworthy sonic* evidence:

1. **Preview coverage is low.** Only **119/480 unique results (25%)** and
   **22/60 seeds (37%)** have a supplied preview; **11/60 seeds have zero
   previewable results**, and the mean previewable share of list positions is
   ~25%. Without audio, "how similar does this **sound**" degrades into a
   title/artist recognition judgment — precisely the proxy the goal forbids. The
   HTML's "use a legal local/player copy" fallback puts inconsistent effort on
   each rater. **Raise preview coverage (re-resolve Deezer/preview URLs, or embed
   a licensed player) before or during collection**, or the sonic evidence will
   be sparse and biased.
2. **One-rater bias.** The inter-rater kappa is undefined with a single rater, and
   the paired bootstrap over 60 seed-level deltas from one person measures *that
   person's* taste, not a population. 

**Minimum raters / workload for a useful CI:**
- One full session ≈ **480 result ratings + 120 list judgments over 60 seeds,
  ~90–150 min**.
- Target **≥3 raters (ideally 5)**, each completing the full 60 seeds (or ≥40 with
  heavy overlap) so per-seed within-rater A/B deltas average out idiosyncrasy and
  kappa is estimable — roughly **4.5–12.5 person-hours total**. A single rater
  yields only a descriptive point estimate, cannot bound one-rater bias, and
  cannot satisfy AC#3's population-level ≥80%/≥20% claim.

---

## Acceptance Criteria Check

- [x] **AC#1 — Frozen suite / scenes / actual lists / provenance.** MET and
  strengthened. 60 seeds, 13 scenes, blinded actual served top-5 lists recorded,
  hash-bound and Ed25519-signed. Plus an independent MTAT human-sonic calibration
  with documented provenance/license.
- [x] **AC#2 — ≥3 materially different approaches + negatives.** MET across the
  program; iter-10 adds a learned MTAT-triplet+FMA-regularized projection vs
  incumbent/CLAP/FMA-SupCon/DSP, with the learned win honestly rejected on CI.
- [ ] **AC#3 — ≥20% gain / no scene <−10% / ≥80% coherent top-5 / no junk.**
  **NOT MET (deciding).** Zero human ratings collected; `sonic_human_report_
  exists=false`; MTAT learned Δ CI spans zero. The Builder does not claim it.
  This is the sole blocking reason for FAIL.
- [x] **AC#4 — External validation, not same-artist-only.** MET/improved: MTAT
  odd-one-out is genuine independent **human sonic** judgment (replacing iter-9's
  94% taste-affinity gold), used as calibration only, no win claimed.
- [~] **AC#5 — Wired into desktop + hosted, live-verified.** N/A for the
  recommender (nothing shipped). **Open item:** the version-filter change touches
  both production paths but is not deployed or live-verified.
- [x] **AC#6 — Resources measured, fits hardware, no silent fallback.** MET:
  RTX 5080 timings recorded (CLAP 8.65 s, projection sweep 12.32 s, etc.), pack
  257 KB, evaluator ~15 KB; no silent quality fallback (fails closed).
- [x] **AC#7 — Regression tests + full suite + docs.** MET: 511 tests green;
  README + CASE_STUDY §13 document the calibration, the honest negative decision,
  the evaluator, limitations, and reproduction. Commit convention clean.

---

## Quality Gate
- Command: `.\.venv\Scripts\python.exe -m pytest tests\ -q`
- Result: **PASS** — independently reproduced **511 passed** (~22 s).
- Note: the gate confirms code correctness only; it does not adjudicate AC#3,
  which is unmet because no human evidence exists yet.

---

## Issues Found (ranked)
1. **[BLOCKER — expected] AC#3 has no human evidence.** `ratings_count_at_freeze=
   0`, `sonic_human_report_exists=false`; MTAT learned Δ CI spans zero. The
   deciding, human-aligned improvement is not demonstrated. (Correctly not
   claimed.)
2. **[READINESS] Low preview coverage.** 25% of results / 37% of seeds
   previewable; 11 seeds have none. Risks sparse and recognition-biased "sonic"
   ratings. Improve before collecting deciding evidence.
3. **[PRECISION] "production_changed:false" understates the serving-filter
   change.** The desktop + hosted quality/version filters changed (behavior
   change if run/deployed). Well-documented and parity-tested, but either ship +
   live-verify (AC#5) or relabel as staged-not-deployed.
4. **[MINOR] FN audit is a consistency check, not independent recall.** The
   explicit-derivative recognizer shares regex logic with the filter; "0 FN" is
   reassuring but not an independent human-labelled recall measurement.
5. **[PROCESS] One-rater bias must be avoided.** Need ≥3 (ideally 5) raters with
   overlapping full sessions for a defensible CI and estimable kappa.

## Credit where due
This is the right pivot after iter-9. Rather than relabel outputs to "pass," the
Builder built the honest machinery to earn AC#3: an independent human sonic
calibration with textbook artist-disjoint, once-opened test discipline and a
correctly-rejected non-significant win (no re-embedding), plus a frozen, signed,
blinded, anti-proxy listening evaluator that I verified end-to-end in a browser
and against the aggregator. Tests grew 489→511 with property-level coverage,
the seal verifies `Good`, no private keys leak, production is not silently
changed, and the prohibited attribution trailer is gone. The remaining gap is
exactly what it should be: **actual human ratings have not been collected**, and
verified evaluation infrastructure is not the same as a verified recommender.

---

## What Must Be Fixed (to reach PASS)
1. **Collect real human ratings.** Have the user + **≥3 raters** complete the
   60-seed study; run the collector-signed exports through
   `human_aggregate_v10` to produce a `sonic_human` report.
2. **Raise preview coverage first** (re-resolve preview URLs or embed a licensed
   player) so the judgments are genuinely sonic, not recognition proxies.
3. **Meet AC#3 on that human evidence** with a clear margin: ≥20% nDCG@5 gain,
   ≥80% coherent top-5, CI excluding zero, and no scene regressing >10% — only
   then open a fresh FINAL / propose deployment. If the honest human result falls
   short, the correct conclusion is that retrieval is not yet there.
4. **Decide the version-filter's production status**: ship it and live-verify on
   the hosted site (AC#5), or clearly mark it staged-not-deployed.
5. **Report inter-rater kappa** alongside the deltas to guard against one-rater
   bias before any promotion.
