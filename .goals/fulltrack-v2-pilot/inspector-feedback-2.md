# Inspector Feedback — Iteration 2

## Verdict: FAIL

## Acceptance Criteria Check

- [ ] Legacy reset — the local `private-ratings-inbox/` directory is absent and no ratings files are tracked, but no authorized retained Blob listing/count proves that the deployed v17 prefix begins at zero.
- [x] Deterministic 20-seed pack — the committed public pack validates and has 20 unique held-out seeds/artists, 20 scenes, slow/medium/fast tempo coverage, five CLAP texture regions, fold `0/test`, and a 55,701-track binding. The full Python suite passed.
- [x] Store/fold/artifact/license binding — the pack and validator bind the sealed store, fold, source identities, locked artifacts, 184 first-party Jamendo tracks, and retained licenses. All 184 public audio URLs target `prod-1.storage.jamendo.com`; every record has `pilot_use: true`.
- [x] Blind four-method/five-result comparison — the public pack contains 20 × 4 opaque lists with five ranked results each. Browser inspection and public-asset checks found no trained-family, baseline, artifact, or unblinding identifiers before rating.
- [x] Lawful browser audio and attribution — no tracked audio or private ratings files exist. In a muted local browser playback check, the Jamendo seed reached `readyState: 4`, advanced playback time, and made a successful HTTP 206 media request. Attribution/license links appeared only after the corresponding list rating; no listening judgment was made.
- [ ] Canonical v2 and retained v1 production routes — FAILED externally. On 2026-07-31, `https://soundalike.yassin.app/evaluate` returned the old 60-seed v17 page and loaded `protocol.json`/`served-lists.json`; `https://soundalike.yassin.app/evaluate-v1` returned HTTP 404. The local Vercel routing configuration and static routes work, but they are not deployed.
- [x] V1/v2 implementation isolation — locally verified. The v1 payload uses the retained `soundalike-human-v17` state and `human-ratings/v17/` path; v2 uses `soundalike-fulltrack-v2`, distinct protocol/pack hashes, `/api/ratings-v2`, and `human-ratings/fulltrack-v2/`. The v2 handler is POST-only, strict/bounded, consent-gated, deduplicated and private; its only list/read implementation is the analyst CLI using private Blob access.
- [ ] V2 resume/import/submit lifecycle — local browser autosave survived reload and v2 tests cover strict signed export/import and accepted in-memory private records, but the required authorized fresh Blob submission, inbox-only observation/validation, and deletion before handoff were not performed or evidenced.
- [x] Automated coverage — `npm test` passed 28/28 and the Python suite passed 778 with 9 skips. Coverage includes public-pack validation, hashes, blinding, v1/v2 state/inbox separation, HMAC/schema/ID/rating tampering, consent, bounds, deduplication, private analyst download, and recommender preservation.
- [x] Required local quality gates — PASS: `C:\Users\solim\Spotify-Statistics\.venv\Scripts\python.exe -m pytest -q` passed 778/787 tests; `npm test` passed 28/28; both the worktree and Builder-commit `git diff --check` checks passed.
- [ ] Live desktop/mobile verification — FAILED externally because production serves v17 at `/evaluate` and has no `/evaluate-v1`. Local v2 desktop/mobile checks passed: 20-seed rendering, one-column mobile layout without horizontal overflow, controls at least 44px high, technical Jamendo playback, post-rating attribution, and reload/resume. The local static-server check cannot substitute for deployed private submission.
- [x] Production recommender unchanged/research-only — verified against the initial SHA: `webapp/api/_reco.py`, `webapp/api/recommend.py`, `webapp/index.html`, and `src/soundalike/data/index_manifest.json` are unchanged. The v2 pack/protocol explicitly prohibit promotion and production-recommender changes.
- [ ] Push/PR/CI/merge/deploy workflow — FAILED. Local `HEAD` is Builder commit `7e320bb`, but `origin/feat/fulltrack-fusion-training` remains at `f553126`; no remote Builder commit, PR, CI result, merge, or production deployment is available.

## Quality Gate

- Command: `C:\Users\solim\Spotify-Statistics\.venv\Scripts\python.exe -m pytest -q`
- Result: PASS — 778 passed, 9 skipped in 36.65 seconds.
- Command: `npm test` (from `webapp`)
- Result: PASS — 28 passed, 0 failed.
- Command: `git diff --check`
- Result: PASS — worktree and `7e320bb^..7e320bb` checks were clean.

## Issues Found

1. The production launch is incomplete. The live canonical route is still v17 and the required archived v1 route is absent; the v2 page, v2 API, and public pack are only present in the local unpushed commit.
2. The external private-Blob acceptance evidence is absent: no authorized zero-count record for v17, no fresh v2 receipt validated through the analyst workflow, no proof that it is isolated from v17, and no deletion record.
3. The required release process has not happened. The remote feature branch is still the initial SHA; therefore there can be no PR, required-CI result, merge, or deployment verification.
4. Local implementation review found no unresolved high-confidence code/security defect in the v2 ingestion boundaries: strict parsing, exact schemas/hashes, bounded bodies, explicit consent, private immutable paths, no public read/list handler, response redaction, and private analyst retrieval are covered. The browser reports only non-blocking form-field naming accessibility issues; the inherited v1 meta-CSP console warning predates this change.

## What Must Be Fixed

1. Push the verified Builder commit to `feat/fulltrack-fusion-training`, open the required PR, obtain green required CI, and merge/deploy without rewriting history.
2. After deployment, independently verify on `https://soundalike.yassin.app` that `/evaluate` is v2 and `/evaluate-v1` is v17 on desktop and mobile, including the deployed headers, public pack, technical Jamendo playback, resume, blinded pre-rating UI, and error/network behavior.
3. With authorized private Blob access, record the v17 prefix count, submit one fresh valid v2 test snapshot, validate it only with `ratings:v2-inbox`, confirm v1 isolation, delete the exact test object, and retain only non-sensitive count/deletion evidence.
