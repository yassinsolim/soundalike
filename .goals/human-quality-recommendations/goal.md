# Goal: Human-quality song recommendations

## User Request

The current recommendations feel very inaccurate and demotivating despite the larger
song pool. Make soundalike genuinely useful and enjoyable by iterating on the data,
models, objectives, retrieval, and ranking as needed. Use the RTX 5080, i9-14900KF,
and 48 GB RAM fully where useful; research and try serious alternatives; test the
product personally; and document the decisions and results in the project docs.

## Refined Goal

Replace the current metric-driven-but-subjectively-weak recommendation behavior with
a demonstrably more human-coherent system. Establish an honest frozen baseline,
experiment with multiple materially different approaches, ship only the strongest
validated method, and verify the live application across mainstream, niche, deep-cut,
and genre-blending songs. External APIs and internal embedding metrics are supporting
evidence; the deciding evidence must include direct inspection of actual ranked song
lists and scene/style coherence.

## Acceptance Criteria

- [ ] A reproducible, versioned evaluation suite freezes the current production
      baseline and covers at least 50 real seed songs across at least 12 distinct
      scenes, including rap, R&B, indie, shoegaze, hyperpop, electronic, metal,
      jazz, city-pop/J-pop/K-pop, Latin/Afrobeats, and difficult genre-blending or
      deep-cut cases. It records actual top-ranked recommendations, not only scalar
      embedding metrics.
- [ ] At least three materially different improvement approaches are implemented
      and compared against the frozen baseline. At least one must go beyond tuning
      the existing alpha/diversity knobs (for example collaborative/co-listening
      priors, neighborhood/graph reranking, a new training objective/model, learned
      reranking, or a robust hybrid). Failed approaches and why they were rejected
      are documented.
- [ ] The selected system shows a clear improvement over the frozen production
      baseline on human-aligned ranked-list quality: at least a 20% relative gain
      on the suite's primary quality score, no scene category regresses by more than
      10% relative, and at least 80% of a separately held-out set of 20 difficult
      seeds have a coherent top 5 with no obvious unrelated-scene recommendation in
      positions 1-3. Duplicate originals, slowed/reverb copies, karaoke, tribute,
      and seed-title mashups must not appear as recommendations for the seed.
- [ ] Supporting external validation (such as ListenBrainz, Last.fm, Deezer,
      playlist co-occurrence, or another researched source) improves or remains
      statistically equivalent to baseline; same-artist retrieval alone may not be
      used as the deciding metric.
- [ ] The best validated method is wired consistently into the canonical desktop
      recommender and hosted numpy/serverless path (or a documented compatible
      deployment architecture), with parity tests updated. The deployed production
      site at https://soundalike.yassin.app is live-verified on at least 10 diverse
      seeds, including previously poor examples, with previews/search still working.
- [ ] Resource and deployment constraints are measured: training/inference time,
      index size, cold-start/load memory, and recommendation latency. The solution
      must fit the existing available hardware and the actual hosted deployment
      limits without silent quality fallbacks.
- [ ] Automated regression tests cover the new ranking/filtering behavior, the full
      existing test suite passes, and relevant documentation (README and case study
      or equivalent existing docs) explains the baseline failure, experiments,
      chosen method, measured improvement, limitations, and how to reproduce the
      evaluation/training/deployment.

## Scope Boundaries

**In scope:**
- Audio encoder/model/objective experiments, collaborative or graph signals,
  candidate generation, reranking, deduplication, hard-negative handling, data
  quality, and index rebuilding.
- Installing research/development dependencies needed for rigorous experiments.
- Updating desktop and hosted recommendation paths, tests, deployment assets,
  release/index manifest, README, and existing technical documentation.
- Honest negative results and reverting approaches that do not improve direct
  recommendation quality.

**Out of scope:**
- Unrelated Spotify authentication, account, playlist-save, branding, or general UX
  work unless required to expose or verify the improved recommender.
- Collecting or committing user passwords, tokens, private playlist data, or secrets.
- Claiming perfection from proxy metrics, same-artist mAP, library size, or coverage.
- Shipping a model merely because it is newer/larger; it must meet the criteria above.

## Applicable Project Conventions

**Quality gate command:**
- `.\.venv\Scripts\python.exe -m pytest tests\ -q`
- Relevant reproducible evaluation and live-browser verification described above.

**Commit convention:**
- Conventional commits (default), with required `[B]`/`[I]` role marker.

**Guidelines:**
- No AGENTS.md, CONSTITUTION.md, or repository guideline directory was found.
- Follow the engineering and validation principles documented in
  `docs/CASE_STUDY.md`, especially the ArcFace regression lesson.

**Rules:**
- No passwords or secrets in git; Spotify authorization remains OAuth 2.0 PKCE.
- Preserve train/validation/test isolation and prevent evaluation leakage.
- Internal metrics are hypotheses, not verdicts; validate actual recommendation
  lists and independent human-behavior evidence before shipping.
- Preserve hosted/desktop ranking parity and index-manifest integrity.
