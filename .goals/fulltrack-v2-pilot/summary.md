# Full-track v2 pilot summary

## Achieved

- Deployed a deterministic, rights-bound 20-seed Jamendo listening pilot.
- Published v2 at `/evaluate` and preserved the byte-locked v17 study at
  `/evaluate-v1`.
- Isolated v1/v2 browser state, protocol hashes, APIs, private Blob prefixes,
  validation, deduplication, and analyst retrieval.
- Verified lawful browser playback, delayed attribution, blinding, resume,
  signed import/export, explicit consent, and submission behavior.
- Kept production recommendation code and hosted index bindings unchanged.
- Passed 778 Python tests, 28 Node tests, `git diff --check`, security review,
  live desktop/mobile checks, CI, deployment, and independent inspection.
- Submitted, validated, isolated, and deleted one controlled v2 test record;
  both private inboxes ended with zero records.

## Iterations

1. **FAIL:** The lawful deterministic pack was complete, but evaluator,
   ingestion, and deployment surfaces were missing.
2. **FAIL:** The complete local evaluator and isolated ingestion passed all
   local gates, but had not been released or verified against private storage.
3. **PASS:** PR #36 merged and deployed; live routes, playback, state/API
   isolation, inbox lifecycle, protected hashes, and all quality gates passed.

## Inspector issues resolved

- Added canonical v2 and retained-v1 routes.
- Added strict isolated v2 state, submission validation, storage, and tests.
- Fixed Python test resolution so the prescribed command uses this checkout.
- Completed normal PR, CI, merge, deployment, live browser, and private-inbox
  verification without force-pushing or changing the recommender.

## Recommendation

Use genuine listening judgments on the production v2 page. Treat the resulting
ratings as research evidence only; do not promote a model automatically from a
single-rater pilot.
