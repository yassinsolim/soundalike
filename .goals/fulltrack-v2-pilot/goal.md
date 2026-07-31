# Goal: Launch the full-track v2 blind pilot

## User Request

Remove the random test evaluation, start the v2 goal/git flow, and launch a
20-song blind listening pilot. Make v2 the canonical `/evaluate` page, retain
the existing study at `/evaluate-v1`, choose the songs automatically for broad
coverage, and use the user's listening feedback to evaluate similar-sounding
recommendations without prematurely changing production.

## Refined Goal

Launch a production-hosted, 20-seed blind pilot for the completed full-track
Jamendo fusion experiment. Compare the frozen hybrid baseline with all three
trained candidate families using fold-correct, artist-disjoint artifacts and
rights-safe browser-playable audio. Preserve the existing v17 study and its
local state at `/evaluate-v1`, isolate both studies' private submissions, and
leave the production recommender unchanged until later evidence supports a
separate promotion decision.

## Acceptance Criteria

- [ ] The confirmed random test record from 2026-07-25 is absent and the legacy
      v17 private inbox begins this rollout with zero records.
- [ ] A deterministic, tested pack builder selects exactly 20 unique held-out
      Jamendo seeds spanning materially different tag/scene, tempo, and CLAP
      texture regions; selection is reproducible and documents its diversity
      evidence.
- [ ] Every seed and recommendation is bound to the sealed 55,701-track store,
      official artist-disjoint fold, exact candidate artifact, source track
      identity, and a license that permits the chosen public pilot use.
- [ ] Each seed blindly compares the frozen hybrid baseline and the three
      trained families (`nonnegative_linear`, `monotonic_network`, and
      `channel_gated_embedding`) with five ranked outputs per method. Method
      ordering and identifiers reveal no model identity to the listener.
- [ ] All pilot audio is browser-playable from an explicitly verified lawful
      delivery path. No source/full-track audio or generated decoded audio is
      committed to Git, no commercial preview is mislabeled as Jamendo
      full-track evidence, and required attribution/license information is
      retained and shown without compromising pre-rating blinding.
- [ ] `https://soundalike.yassin.app/evaluate` serves the new v2 pilot and
      `https://soundalike.yassin.app/evaluate-v1` serves the prior v17 study.
      Existing v1 browser state remains resumable; v1 data is never migrated
      into v2 state.
- [ ] V1 and v2 use distinct local-storage namespaces, protocol/list hashes,
      submission validation, and private Blob prefixes. Both require explicit
      consent, immutable deduplication, bounded strict parsing, and private
      analyst-only retrieval; neither exposes a public listing/read endpoint.
- [ ] The v2 page supports partial progress, reload/resume, export/import, and
      explicit submission. A fresh valid test submission is observed in only
      the v2 inbox, is validated, and is deleted before handing the pilot to
      the user.
- [ ] Automated tests prove deterministic pack generation, fold/model/store
      binding, exact schema/hash checks, method blinding, v1/v2 state and inbox
      isolation, invalid/tampered submission rejection, and preservation of
      the existing evaluator.
- [ ] Local Python and Node quality gates pass, `git diff --check` passes, and
      independent security/code review finds no unresolved high-confidence
      defect.
- [ ] Live desktop and mobile browser verification confirms both evaluator
      routes load, audio plays, progress survives reload, v2 submission works,
      no model identity leaks before rating, and no console/network errors or
      material performance regressions occur.
- [ ] `webapp/api/_reco.py`, `webapp/api/recommend.py`, hosted index constants,
      and production recommendation behavior remain unchanged. The resulting
      pilot is explicitly labeled research evidence and cannot automatically
      promote a model.
- [ ] The implementation is committed on
      `feat/fulltrack-fusion-training`; after independent PASS, the branch is
      pushed, a pull request is opened, required CI is green, and the verified
      change is merged/deployed without force-push or history rewriting.

## Scope Boundaries

**In scope:**
- A deterministic 20-seed Jamendo-native pilot pack and its evidence manifest.
- Frozen hybrid versus all three completed trained fusion families.
- Canonical v2 and archived v1 evaluator routes.
- Rights-safe audio delivery, attribution, explicit-consent private ingestion,
  analyst tooling, tests, deployment, live verification, and git/PR flow.
- Automated Jamendo metrics and cross-fold/artifact checks as secondary
  evidence; the user supplies genuine listening judgments.

**Out of scope:**
- Promoting or wiring a candidate into the production recommender.
- Treating one rater or this pilot as statistically sufficient for promotion.
- Claiming the AI assistant listened to or judged audio.
- Reddit/commercial-song validation until a compatible commercial-catalog
  bridge can score the same songs honestly.
- Spotify/Deezer scraping, unlawful redistribution, or presenting excerpts as
  full-track perceptual evidence.
- Retraining the already completed 45-job matrix unless a reproducible defect
  proves an artifact invalid.

## Applicable Project Conventions

**Quality gate commands:**
- `C:\Users\solim\Spotify-Statistics\.venv\Scripts\python.exe -m pytest -q`
- `npm test` from `webapp`
- `git diff --check`

**Commit convention:**
- Conventional commits with required `[B]`/`[I]` role markers.
- Builder trailer: `Assisted-by: OpenAI:GPT-5.6-Sol`
- Inspector trailer: `Assisted-by: OpenAI:GPT-5.6-Terra`
- Also include the repository-required
  `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>` trailer.

**Guidelines:**
- No `AGENTS.md`, `CONSTITUTION.md`, `.agents/guidelines`, or
  `.github/guidelines` files were present.
- Follow `docs/FULLTRACK_AUDIO.md`, especially its evidence, licensing,
  fold-isolation, integrity, and human-gated-selection boundaries.
- Follow `.goals/human-quality-recommendations/goal.md` for the established
  rule that internal metrics are hypotheses, not promotion verdicts.

**Rules:**
- Preserve artist-disjoint train/validation/test isolation and never use tags,
  ratings, or test identities during training.
- Fail closed on identity, checksum, schema, license, path, or protocol drift.
- Never commit credentials, private ratings, local inbox data, source audio, or
  generated audio artifacts.
- Keep production hosted/desktop recommendation behavior byte-compatible.
- Use only genuine user-provided ratings as human evidence.
