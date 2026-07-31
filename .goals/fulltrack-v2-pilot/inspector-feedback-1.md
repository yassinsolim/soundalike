# Inspector Feedback — Iteration 1

## Verdict: FAIL

## Acceptance Criteria Check

- [ ] Legacy reset — not fully verified. The repository `private-ratings-inbox/` has zero files and no tracked `2026-07-25` record was found, but there is no authorized, retained listing/count proving that the deployed v17 private Blob inbox has zero records.
- [x] Deterministic 20-seed pack — verified. `tests/test_fulltrack_pilot.py` passed (16 tests), the tracked pack has 20 unique-artist seeds, 20 scenes, all three tempo bins, and five texture regions. The external rebuilt public/private documents are byte-identical to the selected pair.
- [x] Store/fold/artifact/license binding — verified. The strict validator accepted the tracked public pack with the external private unblinding document. It binds fold 0/test, the sealed 55,701-track store, 184 first-party Jamendo track records/licenses, the frozen hybrid, and exact seed-17 artifacts for all three candidate families.
- [x] Blind four-method, five-result comparison — verified. Every one of the 20 seeds has four opaque lists of five results; private evidence maps them to `frozen_hybrid`, `nonnegative_linear`, `monotonic_network`, and `channel_gated_embedding`, while the public validator rejects model identity leakage.
- [ ] Browser audio/attribution — only the pack evidence is complete. The manifest records lawful HTTPS Jamendo MP3 HEAD evidence and no committed audio, but there is no v2 browser page to play the audio or show post-rating attribution.
- [ ] Canonical v2 and retained v1 routes — FAILED. Live `https://soundalike.yassin.app/evaluate` is the 60-seed Deezer-backed **v17** page; `https://soundalike.yassin.app/evaluate-v1` returns HTTP 404.
- [ ] v1/v2 state and inbox isolation — FAILED. No v2 evaluator, protocol/list hash, local-storage namespace, submission validator, or private Blob prefix is implemented. The existing Node tests cover only the v17 collector.
- [ ] v2 resume/import/submit lifecycle — FAILED. No v2 UI or private v2 inbox exists, so partial progress, reload/resume, import/export, explicit valid v2 submission, validation, and deletion evidence cannot be verified.
- [ ] Required automated coverage — FAILED. The new 16 pilot unit tests validate builder/document integrity, and existing evaluator tests pass with an explicit source path, but no tests cover the required v1/v2 state/inbox isolation, v2 tampering/submission workflow, or preservation through the new routes.
- [ ] Required quality gates — FAILED. The mandated Python command fails at collection: `C:\Users\solim\Spotify-Statistics\.venv\Scripts\python.exe -m pytest -q` imports `soundalike` from `C:\Users\solim\soundalike-fulltrack` rather than this worktree, producing six import errors. `npm test` passed (15 tests), `git diff --check` passed, and the focused pilot suite passed only with `PYTHONPATH=src`.
- [ ] Live desktop/mobile verification — FAILED. Desktop inspection found the old v17 page and its meta-CSP console error; v1 is 404 and v2 does not exist. Therefore v2 audio, reload, submission, blinding, mobile behavior, and error/performance checks are impossible.
- [x] Production recommender unchanged/research-only — verified. `webapp/api/_reco.py`, `webapp/api/recommend.py`, and the hosted index hashes are unchanged from the initial commit; the pack marks itself research-only with `promotion_allowed: false`.
- [ ] Final PR/CI/merge/deploy workflow — deferred until the implementation passes independent review; no deployable v2 change is present.

## Quality Gate

- Command: `C:\Users\solim\Spotify-Statistics\.venv\Scripts\python.exe -m pytest -q`
- Result: FAIL
- Details: six collection errors due to the configured interpreter resolving the other `C:\Users\solim\soundalike-fulltrack` worktree; it cannot import this commit's full-track modules.
- Command: `npm test` (from `webapp`)
- Result: PASS — 15/15 tests passed.
- Command: `git diff --check`
- Result: PASS.

## Issues Found

1. This commit delivers a credible offline pack-builder milestone, not the requested launched pilot. There are no v2 web assets, API route/validator, deployment configuration, or v2 tests. The committed pack is under `.goals/`, not a deployed evaluator data path.
2. Production routing is incompatible with the goal: `/evaluate` still serves the old v17 study and `/evaluate-v1` is absent. Moving v17 without preserving its exact local state and private ingestion would lose the stated compatibility guarantee.
3. The required private ingestion properties have not been implemented or demonstrated for v2: separate namespace/hash/protocol, explicit consent, strict bounded validation, immutable deduplication, private analyst retrieval, and no public reads.
4. The required quality command is not reproducible from this repository because the specified virtual environment imports a different checkout. The focused suite passing under `PYTHONPATH=src` does not make the mandated command pass.
5. Zero-record evidence for the legacy deployed private inbox and the required fresh-v2-submit-then-delete evidence are missing.

## What Must Be Fixed

1. Implement and deploy the v2 evaluator using the validated public pack: blind list order per session, Jamendo playback, attribution shown only at the appropriate point, consent, partial autosave/resume, export/import, and explicit submission. Do not commit audio or the private unblinding map.
2. Move the exact current v17 evaluator to `/evaluate-v1`; preserve its local-storage data and its existing private ingestion. Make `/evaluate` serve only v2, then verify both production routes on desktop and mobile.
3. Add a distinct v2 protocol/list hash, local-storage namespace, submission schema/validator, and private Blob prefix. Add analyst-only retrieval with no public listing/read route. Demonstrate a valid v2 submission exists only in the v2 inbox, validate it, delete it, and retain auditable zero-record evidence for the legacy inbox.
4. Add tests for v1/v2 state and inbox isolation, v2 hash/schema/tamper rejection, consent/deduplication/bounds, blinded UI behavior, and route preservation. Make the exact prescribed Python command resolve this checkout and pass without `PYTHONPATH` overrides.
5. Run the full local gates and live route/audio/reload/submission/error checks after deployment. Then complete the branch push, PR, CI, and merge/deployment steps only after a subsequent independent PASS.
