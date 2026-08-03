# V3 similarity research

V3 is experimental. The production and rollback path remains
`dual_sonic64_guardrail`. This log records positive and negative evidence so a
small proxy improvement cannot be presented as a production-quality jump.

## Acceptance gates

- At least +20% relative on the predeclared primary held-out metric.
- Paired 95% confidence interval above zero.
- Positive primary delta on at least four of five folds, with no fold below -5%.
- Pooled Recall@10 and MRR no worse than -1% relative.
- A blind V2-versus-V3 pilot with at least 200 valid judgments, at least 60% V3
  preference, and Wilson 95% lower bound above 50%.
- Exact model, data, split, package, and artifact hashes; V2 remains available.

## Research basis

| Method | Primary source | Applied conclusion |
|---|---|---|
| Debiased contrastive learning | [Chuang et al.](https://arxiv.org/abs/2007.00224) | Do not treat likely same-tag candidates as negatives. |
| ANN hard negatives | [Xiong et al.](https://arxiv.org/abs/2007.00808) | Mine negatives from the same CLAP retrieval distribution used at inference. |
| Margin-MSE distillation | [Hofstätter et al.](https://arxiv.org/abs/2010.02666) | Match score margins, not incompatible absolute teacher/student scores. |
| Supervised contrastive learning | [Khosla et al.](https://arxiv.org/abs/2004.11362) | Multi-positive tag supervision is preferable to same-track augmentation alone. |
| MusicFM | [Won et al.](https://arxiv.org/abs/2311.03318) | Test a music-native frozen representation before training a new encoder. |

License screening is fail-closed. The selected MusicFM-FMA metadata is MIT and
the pinned source is MIT/Apache-2.0. The FMA checkpoint was chosen instead of
the separate MSD release. MERT-v1-330M, music2vec-v1, and MuQ-large-msd-iter
currently declare `cc-by-nc-4.0` in their model metadata, so they are not
production candidates.

## MusicFM-FMA canary

Pinned inputs:

- source commit `b83ebedb401bcef639b26b05c0c8bee1dc2dfe71`;
- model revision `4513b38bc25ad1d227b1980819b9691ba97f4d87`;
- checkpoint SHA-256
  `68392eee13d34c2941b3761934abb6b1e67b2e9df498695bda2ea5c1087d4b96`;
- 24 kHz, 30-second windows, 15-second hop, layer 7, 1,024 dimensions;
- batch size 2, measured peak CUDA allocation about 2.55 GB.

Standalone MusicFM trailed CLAP on Recall and NDCG. Uniform-window MaxSim
contained complementary MRR signal, but fixed blends failed one or more
validation folds. Nested score-only gating improved MRR/NDCG on all five outer
validation folds while slightly reducing mean Recall, so it was frozen and
audited once rather than tuned further.

## One-time official-test audit

Frozen policy:

1. retrieve the CLAP global top 200;
2. compute the CLAP 50% global / 25% uniform / 25% section hybrid;
3. compute MusicFM uniform-window MaxSim;
4. z-normalize within each query pool;
5. apply a 25% MusicFM residual only when MusicFM score standard deviation is
   at most `0.05948563385754824`.

| Pooled 539-query metric | CLAP | Candidate | Relative delta | Paired 95% CI |
|---|---:|---:|---:|---:|
| Recall@10 | 0.11113 | 0.11217 | +0.94% | -0.00362..0.00618 |
| MRR | 0.28774 | 0.29003 | +0.80% | -0.01036..0.01474 |
| graded NDCG@10 | 0.13599 | 0.13587 | -0.09% | -0.00455..0.00408 |

Recall fold deltas are -6.81%, +2.99%, +4.54%, +6.75%, and -2.09%. The
candidate fails four primary checks and is rejected. Report payload SHA-256:
`e201582e1425b61a38e9b7af4ba111f65cd1f3e8778c2470616466416aff8868`.
The official test is consumed and cannot support a retuned independence claim.

## Development-only representation probes

All rows below use already-consumed validation folds and never open official
test labels.

| Candidate | Recall delta | MRR delta | NDCG delta | Recall-positive folds | Decision |
|---|---:|---:|---:|---:|---|
| unrestricted supervised score fusion | -5.63% | +0.81% | -4.53% | 2/5 | reject; abandoned CLAP |
| bounded 5% learned residual | +0.96% | +0.40% | +0.83% | 2/5 | reject; too small/unstable |
| semantic tag head, CLAP+MusicFM, ridge 10, 30% residual | +2.98% | +1.24% | +2.49% | 5/5 | scale, not promote |

The semantic head predicts 183 genre, mood/theme, and instrument labels from
frozen global embeddings. Training excludes every held-fold artist. Similarity
uses predicted tag profiles, not held labels. The consistent result suggests
that representation-level semantic supervision is a better lever than another
raw score blend, but +2.98% is still far below the final gate.

## Frozen scale experiment

Before scaling, `fulltrack_v3_protocol.py` froze:

| Split | Tracks | Artists | Track-ID hash |
|---|---:|---:|---|
| train | 5,864 | 1,119 | `357639db357de2ec464e01f683bbd401503b195b6c927db87dce610ba169a7f6` |
| development | 1,134 | 229 | `94dcd748a1b5d758029f486b5bf2a76c1c15a3e4a26015ab12b2c9b4fcfdd2f5` |
| shadow | 1,194 | 229 | `a029fa7b3d66df281f1d76fdf7dfabf6e2af043bda5591188a7f02b498196424` |

Artist overlap is zero. Protocol payload SHA-256 is
`d697240384003ba1a7d9e00d281462b005a3e02abac49964cd3ba4e128292738`.
Shadow labels remain unopened. The 8,192-track MusicFM extraction is resumable,
stores no copied audio, and binds this protocol hash into the sealed store.

`fulltrack_v3_semantic.py` now implements the scaled run before any shadow
evaluation. Its streaming label filter retained exactly 5,864 train plus 1,134
development labels, produced the expected 183-tag vocabulary, and retained zero
shadow labels. The trainer compares frozen CLAP, MusicFM, and concatenated global
representations with ridge values 1/10/100 and semantic residual weights
5/10/20/30%. Ridge uses the scalable LSQR primal path rather than a
5,864-by-5,864 kernel eigendecomposition.

The development-to-shadow gate was fixed before results: at least +15% Recall@10,
a paired 95% interval above zero, at least four of five artist-hashed reporting
folds positive, no fold worse than -5% Recall, and no MRR/NDCG regression beyond
1%. The next allowed sequence is: complete extraction, run the trainer on train
and development only, hash-freeze the winning model, and open shadow once only if
that gate passes. A weak development result ends this branch without shadow
access.

### Scaled CLAP-only development frontier

The CLAP arm was run while MusicFM extraction continued. These are development
results over 938 evaluable queries; shadow labels remain unopened.

| Method | Recall delta | MRR delta | NDCG delta | Recall-positive folds | Decision |
|---|---:|---:|---:|---:|---|
| scaled ridge tag head | +4.06% | +0.63% | +1.62% | 4/5 | below gate |
| raw 16-neighbor tag propagation | +9.79% | -2.48% | +1.38% | 4/5 | reject; unsafe, -9.74% worst Recall fold |
| calibrated safe neighbor profile | +3.63% | -0.35% | +1.09% | 5/5 | below gate |
| nested confidence-gated neighbors | +10.11% | +0.35% | +3.51% | 4/5 | below gate |
| learned confidence gate | +1.01% | -1.62% | -0.51% | 4/5 | reject; -15.52% worst Recall fold |
| nonlinear metric/tag head | +5.19% | +2.78% | +2.77% | 5/5 | stable, below gate |
| metric head plus frozen CLAP text tags | +5.56% | +2.38% | +2.95% | 5/5 | stable, below gate |
| metric head plus nested neighbor gate | +11.28% | +0.84% | +4.89% | 5/5 | best eligible CLAP-only development result; below gate |
| symmetric cross-pair scorer | +0.04% | -0.20% | -0.27% | 2/5 | reject; train-pair overfit |

The best CLAP-only nested result has a positive Recall paired interval
(`+0.00250..+0.01146` absolute), no negative Recall fold, and positive aggregate
safety metrics, but it still fails the preregistered +15% threshold. The
label-using method-selection oracle reached +28.84% Recall and is reported only
as an ineligible upper bound; it is not a deployable gate.

Development has now been used for method selection. It is not independent test
evidence. The only remaining planned scaled branch is the already-preregistered
dual representation after the immutable MusicFM store seals. No result above
authorizes shadow access, listening evaluation, or promotion.

### Frozen dual nonlinear branch

Before the 8,192-track MusicFM store sealed, `fulltrack_v3_metric.py` fixed the
final dual development candidate:

- standardized concatenated CLAP/MusicFM global inputs;
- CLAP-neighborhood triplets with two nearest cross-artist tag positives and two
  nearest zero-shared-tag negatives per eligible train query;
- a 384-hidden/128-latent metric/tag head trained for 200 epochs at seed
  `20260807`, triplet weight 0.25, and margin 0.10;
- 25% predicted-tag plus 75% latent semantic profiles;
- a fixed 75% learned-metric plus 25% zero-shot CLAP text-tag profile; the
  text-only probe was weak (+0.94% Recall), but improved the stable metric head
  to +5.56% Recall with all five folds positive; model construction is seeded
  at 20260807 because the local transformers loader initializes checkpoint-
  absent Roberta pooler parameters;
- 16-neighbor, temperature-0.05 dual label propagation;
- a 20% metric-head fallback and 40% neighbor residual; and
- neighbor activation above the development 40th percentile of maximum weighted
  tag probability, the gate form selected in four of five CLAP nested folds.

The implementation writes a pickle-free NPZ containing neural parameters,
normalization, vocabulary/IDF, frozen text vectors/prompts, the train-only
neighbor reference matrix and labels, and the frozen gate threshold. The report
and model sidecar bind the protocol, text-vector content, and both sealed
manifests. There is no dual hyperparameter grid after the MusicFM result becomes
available: this exact candidate either clears the existing development gate or
is rejected without shadow access.

The 8,192-track MusicFM store sealed before development evaluation. A pre-result
integrity check found that the trainer contained a stale predicted config hash
and incorrectly used the protocol payload hash as the store track-plan hash.
The exact intended binding was independently recomputed from the extraction
configuration, pinned capability, and frozen protocol and matched the sealed
store (`c2b0c316...` config; `43d93666...` track plan). Both constants and the
preregistration were corrected before any candidate report was produced; model,
method, blends, gates, labels, and partitions were unchanged.

### Scaled MusicFM development result

The sealed scaled runs completed without opening shadow:

| Method | Recall | MRR | NDCG | Positive Recall folds | Outcome |
|---|---:|---:|---:|---:|---|
| selected MusicFM ridge | +8.08% | +3.19% | +3.32% | 4/5 | stable, below +15% gate |
| frozen nonlinear/text dual candidate | +2.71% | +0.47% | -0.01% | 3/5 | unstable, rejected |

The selected MusicFM ridge has a positive absolute Recall interval
(`+0.00207..+0.00815`) and a -1.47% worst Recall fold, but it does not meet the
fixed primary-gain threshold. The nonlinear candidate's interval crosses zero
and its worst Recall fold is -8.78%. Neither result authorizes shadow access.
The exact LSQR targets were executed in parallel after a production-size
benchmark proved byte-identical coefficients and a 12.9x speedup; this changed
only execution scheduling, not model definition or results.

### Complementary profile freeze

A final development-only composition probe tested whether the strongest
train-only CLAP neighbor tags and the selected MusicFM ridge profile carry
complementary signal. It evaluated 150 quarter-step profile/score compositions
and baseline residuals without accessing shadow. The selected candidate is
deliberately simpler than the rejected nonlinear branch:

- 50% IDF-weighted, temperature-0.05, 16-neighbor CLAP tag profile;
- 50% selected MusicFM layer-7 ridge-10 tag profile;
- profile-level concatenation, so channel cosine similarities receive equal
  weight; and
- a 50% CLAP MaxSim baseline / 50% semantic-profile score residual.

The repository implementation reproduced the probe exactly over 938 evaluable
development queries:

| Recall | MRR | NDCG | Recall interval | Positive Recall folds | Worst Recall fold |
|---:|---:|---:|---:|---:|---:|
| +15.06% | +1.18% | +4.98% | `+0.00485..+0.01379` absolute | 4/5 | -4.67% |

This is the first scaled candidate to clear every fixed development-to-shadow
check. The self-contained model, development report, sealed stores, exact
method, and final +20% shadow gate are bound in
`.goals/production-ready-v3/artifacts/complementary-shadow-freeze.json`
(`765b4041...`).

### Complementary shadow audit

Commit `237b71e` froze the code and evidence before the one-time shadow audit.
The audit then opened 1,194 shadow tracks exactly once and sealed payload
`17d0a1ec...`:

| Recall | MRR | NDCG | Recall interval | Positive Recall folds | Worst Recall fold |
|---:|---:|---:|---:|---:|---:|
| +4.32% | +4.92% | +7.54% | `-0.00160..+0.00600` absolute | 3/5 | -13.23% |

All pooled metrics improved, and MRR/NDCG intervals were positive, but the
primary Recall result failed four final checks: +20% gain, positive interval,
4/5 positive folds, and no fold below -5%. The fixed gate therefore blocks
listening-pack generation and promotion. The complete immutable result is in
`.goals/production-ready-v3/artifacts/complementary-shadow-result.json`.
This shadow is consumed and must not be used for tuning or a second claim.

### Fresh post-audit protocol

Continued model work uses a new protocol rather than retuning on the failed
shadow. The protocol takes all 32,859 tracks in official fold-0 train and uses
artist identity to enforce the following lineage:

- all 1,577 artists seen anywhere in the consumed 8,192-track protocol become
  training-only, expanding supervision to 31,473 tracks;
- the 562 artists never present in that protocol are deterministically divided
  into 742-track development and 644-track shadow partitions; and
- train/development/shadow artist overlap remains zero.

The 5.92 MB protocol is sealed at payload `8d43ca80...`, selection
`788354f4...`, and file hash `c5c1fc5b...`. Its compact freeze is
`.goals/production-ready-v3/artifacts/fresh-protocol-freeze.json`. Fresh shadow
labels remain unopened. The immediate development work is a larger
CLAP-supervised tag/metric model; no new MusicFM extraction is authorized until
that inexpensive branch demonstrates generalizable value.

### Fresh scaled CLAP candidate

The larger training partition changed the result materially. A 162-candidate
screen found that eight train-only CLAP neighbors at temperature 0.05 reached
+16.21% Recall, +10.35% MRR, and +10.01% NDCG. A focused refinement then
combined two complementary train-supervised tag estimates:

- 50% eight-neighbor tag propagation, with prediction power 0.75 and IDF;
- 50% tag probabilities from a 768-hidden/256-latent MLP trained for 150
  full-batch epochs at seed `20260812`; and
- a 30% frozen CLAP MaxSim baseline / 70% semantic-profile residual.

The production implementation reproduced the selected result exactly over 582
evaluable development queries:

| Recall | MRR | NDCG | Recall interval | Positive folds (all metrics) | Worst Recall fold |
|---:|---:|---:|---:|---:|---:|
| +25.27% | +12.77% | +14.18% | `+0.00859..+0.02986` absolute | 5/5 | +0.70% |

This is the first candidate above the final +20% target before independent
evaluation. The full 416-candidate refinement evidence is committed at
`.goals/production-ready-v3/artifacts/fresh-clap-refinement.json`. Model,
development report, protocol, refinement, method, and final gates are sealed by
`.goals/production-ready-v3/artifacts/fresh-clap-shadow-freeze.json`
(`9236fd75...`). Fresh shadow remained unopened at this freeze point.

### Fresh scaled CLAP shadow audit

Commit `3688af6` froze the implementation and evidence before the one-time audit.
The 644-track shadow was then opened exactly once. Over 465 evaluable queries,
the independent result is:

| Recall | MRR | NDCG | Recall interval | Positive Recall folds | Worst Recall fold |
|---:|---:|---:|---:|---:|---:|
| +5.27% | +3.28% | +2.37% | `-0.00367..+0.01187` absolute | 4/5 | -1.21% |

The candidate is directionally positive and passes the fold-stability and safety
checks, but it fails the fixed +20% Recall requirement and positive paired
interval. The automated gate therefore blocks listening and promotion. The
immutable result is committed at
`.goals/production-ready-v3/artifacts/fresh-clap-shadow-result.json`
(`97ea4f67...`). This shadow is consumed and will not be used for tuning.

An untouched reserve still exists outside fold-0 train. Excluding every artist
used by fold-0 training and every artist present in the historical 512-track
official validation/test selections leaves 945 artists and 6,135 tracks:
3,074 fold-0 validation tracks from 471 artists and 3,023 fold-0 test tracks
from 466 artists, plus 38 unassigned tracks that will be excluded. The next
protocol must freeze the natural validation partition for development and the
natural test partition for one final shadow before any new method sees labels.

### Final untouched reserve protocol

`fulltrack_v3_reserve_protocol.py` now reconstructs both historical official
512-track selections, excludes every artist they touched, and freezes the
remaining natural fold-0 partitions:

| Split | Tracks | Artists | Track-ID hash |
|---|---:|---:|---|
| train | 32,859 | 2,139 | `8da179c6...` |
| development | 3,074 | 471 | `aed55cbb...` |
| shadow | 3,023 | 466 | `39986abc...` |

Artist overlap is zero. The protocol is sealed at payload `ac8c649a...`,
selection `aed5188c...`, and file hash `e04e65a4...`; its compact repository
freeze is
`.goals/production-ready-v3/artifacts/final-reserve-protocol-freeze.json`
(`05abc813...`). Neither reserve development nor reserve shadow labels were
accessed at freeze time. Development may now be used for method selection;
shadow remains a one-time final audit only.
