# Goal: Production-ready V3 similarity

## User Request

Keep working on V3 until it is substantially better than the previous version
and returns songs that are genuinely similar. Research and apply the best
methodologies, use the available compute and storage, commit the research and
documentation, keep the Git history clean, and never add AI attribution to
commits.

## Refined Goal

Build and independently verify a lawful, reproducible V3 recommender that
materially outperforms the currently deployed V2 (`dual_sonic64_guardrail`) and
the frozen full-track CLAP incumbent. V3 may change representations, training
objectives, retrieval, reranking, supervision, or serving architecture, but it
must pass leakage-safe automated evaluation and a statistically credible blind
human listening pilot before being called production-ready. Failed experiments
must be preserved honestly, and V2 must remain available as the rollback path.

## Acceptance Criteria

- [ ] Automated primary gain: a candidate frozen before final evaluation beats
  the appropriate V2 incumbent by at least 20% relative on the predeclared
  primary similarity metric on artist-disjoint held-out evidence, with the 95%
  paired confidence interval above zero. The gain must be positive on at least
  four of five official folds; no fold may regress the primary metric by more
  than 5%.
- [ ] Metric safety: held-out Recall@10 and MRR must each be no worse than 1%
  relative to V2 overall. Any different metric trade-off requires explicit
  human evidence showing that users prefer it; a composite score alone cannot
  hide a material regression.
- [ ] Genuine similarity: a version-blind, randomized human pilot covering at
  least 20 diverse seeds and at least 200 valid pairwise judgments prefers V3
  over V2 at least 60% of the time, with a 95% confidence lower bound above 50%.
  Pilot seeds, judgments, and evaluator-only pairs may not be used for training
  or model selection.
- [ ] Leakage and provenance: training uses only approved train partitions;
  validation chooses candidates; each final holdout is opened once only after
  the candidate and protocol are hash-frozen. Audio, metadata, model weights,
  licenses, package versions, seeds, folds, candidate pools, and artifact
  checksums are bound into reproducible manifests. Spotify audio is never
  downloaded or retained.
- [ ] Production behavior: V3 is integrated behind an explicit version or
  feature gate, V2 remains a tested rollback, failures are surfaced rather than
  silently falling back, and deterministic parity/restart checks pass. Warm
  serving p95 latency, peak memory, and artifact size must remain within the
  repository's documented production budgets or within 20% of V2 unless a
  documented deployment design moves the expensive work offline.
- [ ] User evaluation: the evaluation page can serve a sealed V2-versus-V3
  blind pack, persist judgments safely, reject duplicate/malformed submissions,
  and export an auditable local result without exposing method identity.
- [ ] Quality gates: the smallest relevant targeted tests pass throughout, then
  `C:\Users\solim\Spotify-Statistics\.venv\Scripts\python.exe -m pytest -q`,
  `npm test` from `webapp`, and `git diff --check` pass. Pre-existing
  line-ending-only fixture failures must be either resolved without changing
  evaluator payloads or precisely documented and proven unrelated.
- [ ] Documentation and research: committed Markdown documents record primary
  sources, licensing, hypotheses, experiment protocols, rejected approaches,
  complete benchmark tables, uncertainty, resource usage, reproducible
  commands, serving design, rollback, and honest limitations. Update the
  README, `docs/CASE_STUDY.md`, and directly related full-track/model docs.
- [ ] Clean history: changes are logically scoped and reviewable; generated
  data/model artifacts stay outside Git; the branch ends clean; new commit
  messages and committed files contain no automated attribution, model
  authorship credit, or co-author trailers.
- [ ] Independent verification: a fresh read-only verification pass reproduces the decisive metrics,
  audits leakage/provenance and licenses, exercises the production path and
  rollback, reviews the blind-pilot evidence, and returns PASS. Until every
  criterion passes, documentation must say experimental rather than
  production-ready.

## Scope Boundaries

**In scope:**
- Researching lawful music-native encoders, self-supervised representations,
  contrastive/distillation objectives, hard-negative mining, section-level late
  interaction, calibrated/selective fusion, ANN retrieval, and offline serving.
- Using CPU, RTX 5080 GPU, RAM, and external storage for bounded and resumable
  experiments.
- Reusing the sealed CLAP store and current MusicFM-FMA canary where provenance
  matches, while replacing the approach when evidence rejects it.
- Training only on official train partitions and selecting on validation
  partitions, with independent final evidence and a blind human pilot.
- Updating application code, tests, evaluation tooling, documentation, and the
  user-testing flow needed for a production candidate.

**Out of scope:**
- Ripping, downloading, caching, or redistributing Spotify audio.
- Production use of checkpoints or datasets whose commercial rights are absent,
  noncommercial-only, gated, or unresolved.
- Training on held-out test labels, editorial pairs, sealed evaluator seeds, or
  human judgments reserved for final evaluation.
- Claiming success from validation-only, proxy-only, cherry-picked, single-fold,
  or composite-only improvements.
- Removing or silently changing V1/V2 behavior before V3 passes every gate.
- Unrelated repository refactors, destructive history rewrites, or committing
  large generated model/data artifacts.

## Applicable Project Conventions

**Quality gate commands:**
- `C:\Users\solim\Spotify-Statistics\.venv\Scripts\python.exe -m pytest -q`
- `npm test` from `webapp`
- `git diff --check`
- Run focused full-track, evaluation, recommendation, API, and ratings tests
  before the full gates.

**Commit convention:**
- Conventional commits with a title at most 72 characters.
- Keep implementation, evidence, and documentation changes logically scoped.
- No attribution trailer or AI/model authorship reference is permitted. This
  user requirement overrides generic goal-skill trailer guidance.

**Guidelines:**
- No `AGENTS.md`, `CONSTITUTION.md`, `.agents/guidelines`, or
  `.github/guidelines` exists in this worktree.
- Existing evidence and commands are documented in `README.md`,
  `docs/CASE_STUDY.md`, `docs/FULLTRACK_AUDIO.md`, and prior `.goals/` records.

**Rules:**
- Initial branch: `exp/v3-musicfm-canary` at
  `cdd99df3060cfa9e8cd996556198b0f916c4f7cd`.
- The MusicFM layer-7 replacement, fixed blends, and frozen selective policy
  failed promotion. The one-time test is consumed and cannot be used to tune
  another independence claim.
- A hash-bound 1,702-track MusicFM final-test union is sealed externally at
  `C:\soundalike-data\mtg-jamendo-fulltrack-artifacts\musicfm-fma-final-test-canary-union`.
  Its already-frozen policy was audited once and rejected.
- A new 8,192-track artist-disjoint protocol is frozen for the semantic-head
  branch. Its 1,194-track shadow labels remain unopened until train-only fitting
  and development-only selection are complete and hash-frozen.
- Preserve V2 artifacts and signed evaluator state. Keep experimental artifacts
  external to Git and commit only code, compact reports, and documentation.
