# Full-track MTG-Jamendo research audio

This pipeline is for lawful, non-commercial research on the official
[MTG-Jamendo Dataset](https://github.com/MTG/mtg-jamendo-dataset). It does not
download from Spotify, rip Spotify audio, grant redistribution rights, or establish
that any resulting model may be commercially deployed. Keep the official
track-specific license/attribution records and assess downstream use separately.
Audio is input-only: extraction never writes or retains WAV files or other audio
copies.

## Evidence boundaries

- Full-track results are labelled **`full_track_jamendo_research`**.
- An exported commercial benchmark v6 replay is separately labelled
  **`preview_30s_commercial`**.
- Jamendo results must not be presented as commercial-catalog results.
- `fulltrack_eval.py` refuses to open or write any path inside signed
  `protocol-v6` state. A commercial replay must be an external read-only JSON
  export; signed state is never an input.

## Exact environment

From the repository root in PowerShell:

```powershell
$python = 'C:\Users\solim\Spotify-Statistics\.venv\Scripts\python.exe'
& $python -m pip install -e '.[ml,fulltrack]'
& $python -c "import av, importlib.metadata as m; print(av.__version__); print(m.version('laion-clap'))"
& $python -m soundalike.ml.fulltrack_extract capability
```

The `fulltrack` extra pins PyAV to `av==18.0.0`; no external `ffmpeg` executable is
needed. The frozen adapter requires `laion-clap==1.1.7`, CUDA, and the already-local
`630k-audioset-best.pt` checkpoint with SHA-256
`8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037`.
The capability command validates these conditions before model use. Model
initialization forces Hugging Face/Transformers offline mode and uses only the pinned
local CLAP checkpoint plus already-local package assets; a missing cache fails closed
instead of downloading. A generic music-model protocol exists,
but no second concrete model is enabled because no second package/checkpoint/
license/CUDA combination has been verified.

## Required local dataset

Default paths:

```text
C:\soundalike-data\mtg-jamendo-raw-full\audio
C:\soundalike-data\mtg-jamendo-raw-full\state
C:\soundalike-data\mtg-jamendo-dataset
```

Use only the official metadata repository and verified official archives. Before
production extraction, `state\collection.complete.json` must declare:

- collection `raw_30s/audio`;
- exactly 100 archives and 55,701 tracks;
- the SHA-256 of both official archive and per-track manifests; and
- the total bytes validated by all 100 per-archive `*.verified.json` markers.

The upstream
[MTG-Jamendo README](https://github.com/MTG/mtg-jamendo-dataset) defines
`raw_30s.tsv` as **tracks with duration more than 30 seconds**. It separately states
that all audio is distributed as 320-kbps MP3 and that `raw_30s/audio` contains all
available audio for that table in full quality. Therefore, `raw_30s` is a duration
eligibility name, not evidence that the files are 30-second excerpts; this pipeline
decodes the supplied full tracks.

Counts alone do not pass. The loader re-hashes the concrete official manifests,
audits the union of per-archive track claims, checks all metadata/manifest/license
joins, rejects missing files and size drift, and verifies all five official folds
are artist-disjoint. URLs, absolute paths, traversal, symlinks, junctions, duplicate
case-colliding paths, and unknown license authorities fail closed.

On Windows, Git may materialize `data\autotagging.tsv` as a 31-byte text placeholder
containing `raw_30s_cleantags_50artists.tsv`. The parser recognizes only that exact
placeholder and prefers the concrete official TSV. It never treats arbitrary text
as a link.

Check the external completion gate without loading CLAP:

```powershell
Test-Path 'C:\soundalike-data\mtg-jamendo-raw-full\state\collection.complete.json'
```

If this prints `False`, **do not run extraction**. The extraction command also
checks the marker before allocating the model.

## Production extraction

After the external completion marker validates all 55,701 tracks:

```powershell
$python = 'C:\Users\solim\Spotify-Statistics\.venv\Scripts\python.exe'
$env:PYTHONPATH = 'C:\Users\solim\soundalike-fulltrack\src;C:\Users\solim\soundalike-fulltrack\webapp\api'

& $python -m soundalike.ml.fulltrack_extract extract `
  --metadata-root 'C:\soundalike-data\mtg-jamendo-dataset' `
  --audio-root 'C:\soundalike-data\mtg-jamendo-raw-full\audio' `
  --state-root 'C:\soundalike-data\mtg-jamendo-raw-full\state' `
  --output 'C:\soundalike-data\mtg-jamendo-fulltrack-artifacts\clap-v1' `
  --window-seconds 10 `
  --hop-seconds 5 `
  --batch-size 32 `
  --repetition-sections 32 `
  --salient-sections 32 `
  --shard-tracks 256
```

Each pending source MP3 is SHA-256 checked against the official track manifest,
then decoded directly to bounded in-memory mono float32 chunks by PyAV. Windows are
10 seconds with a 5-second hop plus one deterministic end-aligned tail window.
Tracks shorter than a window use deterministic repeat-padding. Each window is
normalized; the global vector uses unique temporal-coverage weights. Repeated and
salient sections are deterministic and bounded. Production and CLI defaults retain
up to 32 repeated and 32 salient source-window vectors per track.

The artifact directory contains:

- `ledger.sqlite3`: immutable row plan and completion/checksum ledger;
- `global.f16`: raw float16 global embedding memmap;
- `shards\*.npz`: float16 variable window/repeated/salient arrays, offsets, and
  repeated/salient source-window indices; and
- `store.sealed.json`: final source/config/model/checksum binding.

The sealed manifest and extraction config bind the repeated and salient budgets
independently. Evaluation fails before scoring if a requested section or hybrid
budget exceeds either declaration. Shard validation also requires every section's
persisted source-window index to be unique, in range, and vector-identical to that
source window.

Rows become complete only when a shard generation and its global-row checksums are
committed. Restart validates every completed global row and shard. A crash before a
seal reprocesses bounded work; it never silently skips a pending row. Config,
source, model, checksum, size, and track-order drift all fail closed. A sealed reader
also reconciles ledger shard routing and per-track row/local/decoded metadata against
the manifest-bound shard contents before returning vectors.

## Evaluation

After the store is fully sealed:

```powershell
& $python -m soundalike.ml.fulltrack_eval evaluate `
  --metadata-root 'C:\soundalike-data\mtg-jamendo-dataset' `
  --audio-root 'C:\soundalike-data\mtg-jamendo-raw-full\audio' `
  --state-root 'C:\soundalike-data\mtg-jamendo-raw-full\state' `
  --store 'C:\soundalike-data\mtg-jamendo-fulltrack-artifacts\clap-v1' `
  --output 'C:\soundalike-data\mtg-jamendo-fulltrack-artifacts\clap-v1-eval.json' `
  --fold 0 `
  --part test `
  --maxsim-budget 8 `
  --candidate-pool 200 `
  --min-shared-tags 2 `
  --min-tag-jaccard 0.25 `
  --bootstrap-iterations 2000
```

The single-fold report evaluates:

1. global cosine;
2. uniform-window symmetric mean-MaxSim with the requested fixed budget per side;
3. `0.5 * repeated-section MaxSim + 0.5 * salient-section MaxSim`; and
4. the frozen hybrid `0.50 * global + 0.25 * uniform + 0.25 * section`.

Late interaction only reorders the frozen global top-200 pool; the remaining global
order is unchanged. Relevance requires at least two shared tags and tag Jaccard >=
0.25; NDCG uses Jaccard as a graded label. Labels come from each official split TSV,
not the less-processed `raw_30s.tsv` tag column. This measures artist-disjoint
shared-tag retrieval, not direct perceptual similarity or annotated chorus retrieval.

Equal budgets prevent longer tracks from receiving more MaxSim draws, but do not
equalize effective temporal diversity. Each stored section vector comes from a
distinct selected source-window index. Deterministic index repetition is permitted
only when a track inherently has fewer source windows than the requested,
store-declared budget. Reports disclose, for repeated and salient streams, the
minimum/median/maximum selected source-window counts and number of tracks that
repeat at the requested budget. The repeated stream measures embedding
self-recurrence, which can favor steady textures; it is not a chorus/verse detector.
The 32-vector streams retain score/rank order, so budget 8 and 16 use the top-ranked
prefixes that direct 8- and 16-vector selection would produce; they are not uniform
subsamples of the stored 32.

The machine-verifiable benchmark command fixes all five official folds and all three
budgets (8, 16, and 32):

```powershell
& $python -m soundalike.ml.fulltrack_eval benchmark-all `
  --metadata-root 'C:\soundalike-data\mtg-jamendo-dataset' `
  --audio-root 'C:\soundalike-data\mtg-jamendo-raw-full\audio' `
  --state-root 'C:\soundalike-data\mtg-jamendo-raw-full\state' `
  --store 'C:\soundalike-data\mtg-jamendo-fulltrack-artifacts\clap-v1' `
  --output-dir 'C:\soundalike-data\mtg-jamendo-fulltrack-artifacts\clap-v1-all-folds' `
  --candidate-pool 200 `
  --bootstrap-iterations 2000 `
  --bootstrap-seed 20260714
```

It retains 15 `fold-*-budget-*.json` artifacts plus
`benchmark-summary.json`. Resume skips an artifact only after its payload checksum,
source fingerprint, complete sealed-store content binding, exact current-input test
protocol, metric labels/cutoffs, relevance and method definitions, effective section-diversity
metadata, current fold query/artist/tag/relevance descriptors, and result checksum
all verify. Arbitrary, stale, corrupt, or
protocol-mismatched files are recomputed. All 15 ordered matrix slots must share the
same invariant protocol/store identity before aggregation. The summary preserves
every per-fold metric and defines:

- **fold-macro**: the unweighted mean of the five fold metrics; and
- **query-weighted**: the mean over pooled per-query observations.

For every method and each metric—Recall@K, standard MRR over the complete ranked
list, and graded NDCG@K—the query-weighted summary reports method-minus-global
uncertainty from a deterministic pooled paired bootstrap over aligned
`(fold, track_id)` observations. These method/budget comparisons are descriptive and
have no multiple-comparison correction.

Reports include marginal metric intervals plus paired bootstrap intervals for each
method-minus-global metric, per-metric improved/regressed/unchanged query counts,
latency, observed process RAM, CUDA peak allocation, feature-cache bytes, and store
bytes. Recall is explicitly Recall@K, MRR searches the complete ranked list rather
than the Recall@K prefix, and NDCG@K uses graded Jaccard relevance. Per-category and
per-tag results are descriptive and uncorrected for multiple comparisons.

## Self-supervised fusion training and selection

The trainer compares three bounded fusion families: non-negative linear pair
features, a monotonic network, and a channel-gated embedding. Training supervision
comes only from disjoint temporal views of the same stored track plus mined
negatives. It never reads fold tags, Jamendo tags, tag Jaccard, ratings, external
graphs, audio files, or same-artist positives.

Inspect the fixed five-fold x three-candidate x three-seed plan, then train or
strictly resume all 45 jobs:

```powershell
$trainingRoot = 'C:\soundalike-data\mtg-jamendo-fulltrack-artifacts\fusion-v1'

& $python -m soundalike.ml.fulltrack_train train-all --plan

& $python -m soundalike.ml.fulltrack_train train-all `
  --metadata 'C:\soundalike-data\mtg-jamendo-dataset' `
  --audio 'C:\soundalike-data\mtg-jamendo-raw-full\audio' `
  --state 'C:\soundalike-data\mtg-jamendo-raw-full\state' `
  --store 'C:\soundalike-data\mtg-jamendo-fulltrack-artifacts\clap-v1' `
  --output $trainingRoot `
  --candidate-workers 3 `
  --pairwise-cosine-mode linear-v2 `
  --device cuda
```

Each job writes a checksummed report, checkpoint, and NumPy inference artifact.
Resume recomputes the training-config, dataset, ranking, store, source, and job
bindings, verifies the checkpoint, re-hashes the model files, and compares loaded
fusion metadata with the report before reuse.

### Linear-time pairwise-cosine diagnostic

Training reports include the mean pairwise cosine between the $n$ normalized
global view vectors. The direct `legacy-v1` implementation forms the complete
$n \times n$ similarity matrix and averages its upper triangle. For vectors
$x_1, \ldots, x_n$, `linear-v2` uses the identity

$$
\frac{2}{n(n-1)}\sum_{i<j}x_i^\mathsf{T}x_j
=
\frac{\left\|\sum_i x_i\right\|^2-\sum_i\left\|x_i\right\|^2}
{n(n-1)}.
$$

This changes the diagnostic from $O(n^2d)$ time and $O(n^2)$ additional
similarity-matrix memory to $O(nd)$ time and $O(d)$ reduction workspace for
embedding width $d$. It does not approximate or sample pairs. Floating-point
operation order can cause rounding-level differences, so the two modes should
not be described as bit-identical.

Measured verification:

- A 6,000 x 512 synthetic benchmark took 11.82 s with `legacy-v1` and
  0.0048 s with `linear-v2`, about 2,460x faster for this diagnostic; the
  absolute result difference was $1.19 \times 10^{-19}$.
- A production-shaped fold canary with 2,048 tracks and 4,096 views differed by
  $6.11 \times 10^{-16}$ absolute and $1.56 \times 10^{-15}$ relative.
- Adversarial identical, orthogonal, antipodal, cancellation-heavy, and random
  vector tests passed. The production canary preserved pair hashes, view counts,
  overlap checks, and every non-diagnostic statistic.

The 2,460x figure applies only to pairwise-cosine summarization. View formation,
negative mining, feature extraction, and model fitting still determine much of
end-to-end training time. `linear-v2` also labels the report algorithm as
`sum-vector-v2` and binds the dataset hash to structural objective data, keeping
diagnostic rounding out of negative-mining seeds. Use it for new training
matrices. Keep `legacy-v1` only when strictly reproducing or resuming artifacts
created with the legacy hash semantics; strict resume treats the modes as
different configurations.

Evaluate all trained candidates on the same frozen global top-200 pools:

```powershell
$trainedEval = 'C:\soundalike-data\mtg-jamendo-fulltrack-artifacts\fusion-v1-eval'

& $python -m soundalike.ml.fulltrack_eval benchmark-all `
  --metadata-root 'C:\soundalike-data\mtg-jamendo-dataset' `
  --audio-root 'C:\soundalike-data\mtg-jamendo-raw-full\audio' `
  --state-root 'C:\soundalike-data\mtg-jamendo-raw-full\state' `
  --store 'C:\soundalike-data\mtg-jamendo-fulltrack-artifacts\clap-v1' `
  --trained-root $trainingRoot `
  --output-dir $trainedEval `
  --selection-budget 8 `
  --selection-primary-metric recall_at_k `
  --selection-list-id fulltrack-trained-candidates-v1 `
  --bootstrap-iterations 2000 `
  --bootstrap-seed 20260714
```

The trained run keeps separate cache files, binds every cache entry to the exact
ordered model/report/config hashes and active ablations, and verifies trained
paired deltas from per-query records before reuse. The selected budget and primary
metric are preregistered in `selection/candidate-list.json`. The run also writes
one non-ablation selector report per candidate/fold/seed plus
`selection/selection-inputs.json`; with the default matrix that is 45 reports.

Build the model-selection report directly from that manifest:

```powershell
& $python -m soundalike.ml.fulltrack_selection report-from-manifest `
  --training-root $trainingRoot `
  --manifest "$trainedEval\selection\selection-inputs.json" `
  --output "$trainedEval\selection-report.json"
```

The manifest path verifies every listed raw file hash, semantic content hash,
candidate/fold/seed tuple, candidate-list binding, deciding budget, primary metric,
and derived training-report path. **Automated Jamendo metrics never authorize
promotion.** Without a separately supplied, correctly bound `--trusted-ratings`
artifact from at least three independent human raters, `promotion_allowed` remains
`false`. This research workflow does not deploy a model or establish commercial
use rights.

To inspect an already-exported commercial v6 replay without opening signed state:

```powershell
& $python -m soundalike.ml.fulltrack_eval commercial-v6-replay `
  --replay 'C:\path\outside-signed-state\commercial-v6-replay.json'
```

## Resource accounting

The trusted SHA-verified `batch64` pilot measured:

- 128 full tracks and 5,780 windows;
- decoded duration mean 228.11 s (min 49.44, median 218.15, max 749.24);
- CLI extraction wall time 84.840 s;
- 68.13 windows/s and 1.509 tracks/s;
- worker peak approximately 2.90 GiB RSS / 9.26 GiB private memory;
- GPU peak 7,495 MiB, with maxima of 55 C and 93.25 W; and
- no decoded audio persisted.

A naive `55,701 / 1.509` extrapolation is approximately 10.3 hours. This is a
**planning projection only**, not a measured full-corpus completion; corpus mix,
I/O, thermals, checkpointing, and contention can change sustained throughput.

Deterministic lower-level sizes/bounds are:

- global float16 memmap: `55,701 * 512 * 2` bytes (54.4 MiB);
- each stored window or section vector: `512 * 2` bytes plus offsets/container
  overhead;
- at most 32 repeated plus 32 salient selected vectors per track with defaults;
- decoder output chunk: at most 2 seconds at 48 kHz mono float32;
- model waveform batch: at most 32 x 10 seconds at 48 kHz float32 by default;
- windows per track: hard-capped at 2,048; and
- evaluation fixed-budget cache: hard-capped at 2 GiB by default.

Actual window storage depends on measured track durations and must be reported from
the sealed pilot/store, not guessed. The store never contains encoded or decoded
audio.

## Offline tests

```powershell
$env:PYTHONPATH = 'C:\Users\solim\soundalike-fulltrack-trainer\src;C:\Users\solim\soundalike-fulltrack-trainer\webapp\api'
& $python -m pytest -q tests\test_jamendo_fulltrack.py tests\test_fulltrack_store.py tests\test_fulltrack_extract.py tests\test_fulltrack_eval.py tests\test_fulltrack_fusion.py tests\test_fulltrack_train.py tests\test_fulltrack_selection.py
& $python -m pytest -q
git diff --check
```

Tests use synthetic metadata/audio and fake encoders. They do not use the network,
do not run corpus extraction, and do not access signed protocol state.
