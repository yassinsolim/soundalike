# soundalike

**Find songs similar to the ones you like — an open-source music recommender.**

`soundalike` started life as a first-year university project (`spotify_program.py`): a
terminal script that read a static CSV of top songs and printed min/max/mean stats. This
repo evolves it into a real, working recommendation engine that finds songs matching your
taste — built to work *around* Spotify's 2024 API lockdown rather than depending on it.

It combines several engines — offline audio-feature similarity, an acoustic DSP engine that
measures features straight from the waveform, a **vibe engine** that matches a song's bass
profile and dynamics (the drops), a **deep-vibe engine** that fuses the neural embedding with
that vibe signal, live Spotify (OAuth PKCE), and a **self-supervised neural network trained on
106,000 songs** whose genre-probe accuracy climbs from 0.25 → 0.641 as the training set scales
from 475 to 106k tracks.

> **📖 Want the engineering story?** The [**Case Study**](docs/CASE_STUDY.md) walks through the
> design decisions, the machine-learning scaling experiment, and the GPU/systems challenges I
> solved (data-loading bottlenecks, VRAM-aware training, 11x download speedups, cuDNN kernel
> inspection). It's written as a portfolio-style deep dive.

---

## Why this exists (and why it's built the way it is)

On **2024-11-27, Spotify removed** several Web API endpoints for all *new* apps
([announcement](https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api)),
including the two you'd normally reach for here:

- **Recommendations** — the endpoint behind "Song Radio" / similar-song discovery.
- **Audio Features** — danceability, energy, valence, etc. (the entire basis of the old project).

So a modern tool **cannot** just ask Spotify for similar songs or audio features. `soundalike`
solves discovery with its own engines instead:

| Engine | Signal | Needs credentials? | Coverage |
|--------|--------|--------------------|----------|
| **Deep-vibe** ⭐⭐⭐ | **Fusion** of the learned neural embedding + bass/dynamics, vs a ~1,600-song library | No | Real, listenable songs |
| **Vibe** ⭐⭐ | Frequency-band balance (sub→air) + **dynamics** (the drops), vs a ~1,500-song library | No | Real, listenable songs |
| **Acoustic DSP** ⭐ | Features measured from the **actual audio waveform** (tempo, energy, timbre…) | No | Any track with a preview |
| **Content-based** | Audio-feature similarity on a bundled dataset | No | Songs in the dataset (~855) |
| **Learned model** | A CNN trained on your GPU to embed audio (research track) | No | What you train it on |
| **Live Spotify taste** | Your liked / top / recent tracks as seeds | Free Spotify app (OAuth) | Your library |
| Last.fm *(optional)* | Crowd-sourced "similar tracks" | Free API key | Any track |

The **acoustic engines are the heart of the project**: instead of trusting anyone's precomputed
numbers, they download a 30-second preview and *measure* the sound itself with digital signal
processing, then rank by those measurements. Similarity by the physics of the audio — not by
"people who listened to X also listened to Y" (which is all Spotify radio and Last.fm do). The
**vibe engine** goes furthest, explicitly modelling a track's bass profile and its dynamics (the
drops) so recommendations match the *feel*, not just the timbre.

What Spotify *still* allows (and we use): your library/top/recent tracks, artist genres,
search, and **playlist creation** — so results can be saved straight back to your account.

We never ask for your password. Live access uses OAuth 2.0 with PKCE.

---

## Install

```bash
git clone https://github.com/yassinsolim/Spotify-Statistics.git
cd Spotify-Statistics
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -e ".[dev]"
```

This installs the `soundalike` command.

---

## Quickstart — offline, no credentials

Find songs similar to a single track:

```bash
soundalike similar --title "Blinding Lights" -n 10
```

Care more about some features than others (weights are repeatable):

```bash
soundalike similar --title "Believer" --weight energy=2 --weight danceability=1.5
```

Build a **taste profile** from several songs and get recommendations for the blend:

```bash
soundalike profile --seeds "Blinding Lights; One Dance; STAY" -n 15
```

Explore the dataset (a nod to the original project, done properly):

```bash
soundalike stats
```

---

## Live mode — your real Spotify taste

One-time setup (~2 minutes) is documented in **[SETUP.md](SETUP.md)**: create a free
Spotify app, copy `.env.example` to `.env`, and add your Client ID (plus a free Last.fm key
for full-catalog coverage).

```bash
soundalike login                      # authorize in your browser (OAuth PKCE)
soundalike whoami                     # confirm you're connected

# Recommend from your top tracks using the bundled audio-feature engine:
soundalike recommend --source top -n 25

# Full-catalog recommendations via Last.fm, saved as a new playlist:
soundalike recommend --source liked --engine lastfm --playlist "soundalike picks"

# Export your library to a CSV you can reuse offline:
soundalike pull --source liked --limit 200 --out liked.csv
soundalike profile --file liked.csv -n 30
```

> The `content` engine only matches songs present in the bundled ~855-song dataset, so
> coverage of your personal library is partial. Use `--engine lastfm` for any track.

---

## Acoustic engine — similarity by science ⭐

The flagship engine measures the sound itself and compares those measurements. It needs no
credentials (uses free Deezer previews) and works for any track with a preview.

```bash
# Measure one song's acoustic features straight from its audio (DSP):
soundalike audio-features --title "Babydoll" --artist "Dominic Fike"

# Recommend by measured acoustic similarity:
soundalike audio-similar --title "Babydoll" --artist "Dominic Fike" -n 12

# Seed from your live Spotify taste instead:
soundalike audio-similar --source top --seed-limit 5 -n 20
```

**How it works.** For each track it decodes a 30s preview and computes, with `librosa`:
tempo (BPM), RMS energy, spectral centroid/rolloff/bandwidth, zero-crossing rate, spectral
contrast, and 13 MFCCs (a timbre fingerprint). Features are standardized (so BPM and the
spectral features are comparable), optionally weighted, and ranked by distance to your seed.
A catalog (Deezer) is used *only* to enumerate candidate songs and fetch audio — the ranking
is 100% your measured acoustics, never a crowd "also-liked" signal.

Example: seed *Babydoll — Dominic Fike* → Omar Apollo, Malcolm Todd, more Dominic Fike. A
tight bedroom-pop/indie cluster chosen purely from waveform features.

---

## Vibe engine — match the *feel* of a track ⭐⭐

The acoustic engine above averages each feature over the whole clip, which works for
consistent songs but washes out **dynamics**: a track with quiet verses and a heavy drop ends
up looking "medium" everywhere. The **vibe engine** fixes that by measuring the two things that
actually define a song's feel:

- **Frequency-band balance** — how energy splits across **sub / bass / low-mid / mid / high-mid
  / presence / air**. This is the literal "how much sub-bass, how much highs" of a track.
- **Dynamics** — how much the loudness *moves* (standard deviation, dynamic range, and crest =
  peak / average). This is what separates a steady mellow song from one with a big drop.

It ranks against a **bundled library of ~1,500 real songs** across hip-hop, EDM, electro, pop
and hyperpop (built from Deezer previews), with the low-end and dynamics **weighted highest** so
bass-heavy drop tracks match other bass-heavy drop tracks.

```bash
# Find songs with a similar vibe (works out of the box — library ships with the package):
soundalike vibe-similar --title "Wasting Time" --artist "eric404"

# Emphasize a specific quality, e.g. sub-bass, even more:
soundalike vibe-similar --title "HUMBLE." --artist "Kendrick Lamar" --weight band_sub=4

# Build/refresh your own library (saved to ~/.soundalike):
soundalike vibe-build --per-genre 150
```

It also prints a plain-English read of the seed's vibe, e.g.:

```
Seed: Wasting Time — eric404
  vibe: 123 BPM, very dynamic (big drops), bass-heavy, warm
```

**Why this matters (a worked example).** *Wasting Time* by eric404 is 73% sub-bass with a big
dubstep drop (crest 2.2). The plain acoustic engine, averaging that away, returned soft
bedroom-pop. The vibe engine reads the drops and the sub-bass correctly and returns
hyperpop/electronic tracks that actually match — **aldn**, **Flume**, **Slow Magic** — the right
scene, chosen by the shape of the sound.

---

## Deep-vibe engine — the best matcher ⭐⭐⭐

The vibe engine knows about bass and dynamics but has no *learned* sense of texture; the neural
model has deep texture understanding but is partly blind to energy. The **deep-vibe engine fuses
both**: it embeds a song with the trained neural encoder (texture) *and* measures its vibe vector
(bass profile + dynamics), then ranks a **bundled library of ~1,600 real songs** by a tunable
blend of the two similarities.

```bash
# Fused recommendation (works out of the box):
soundalike deep-vibe-similar --title "Wasting Time" --artist "eric404"

# Dial the blend: 1.0 = pure learned texture, 0.0 = pure bass/dynamics:
soundalike deep-vibe-similar --title "Wasting Time" --artist "eric404" --alpha 0.35
```

Each result shows its breakdown so you can see *why* it matched:

```
Seed: Wasting Time — eric404
  vibe: 123 BPM, very dynamic (big drops), bass-heavy, warm
  blend: 60% learned-texture + 40% bass/dynamics

   1. Never Be Like You — Flume        [blend +2.29 | texture 0.68 | vibe 0.26]
   2. Blink Twice — Cecile Believe     [blend +2.10 | texture 0.72 | vibe 0.24]
   3. idontcareanymore — Alice Gas     [blend +1.96 | texture 0.84 | vibe 0.19]
```

The library pairs the 106k-song neural encoder with a curated set of real tracks (genre charts
plus hyperpop / underground / electronic artists that charts miss), so a niche seed has close
neighbours. Matching the *feel* of a track is genuinely hard — this is the frontier the project
is still pushing on — but the fusion is a clear step past matching timbre or dynamics alone.

---

## Learned-model research track (GPU)

An experimental track trains our own audio-embedding CNN (PyTorch) to place similar-sounding
songs near each other, with a self-supervised contrastive objective (SimCLR/NT-Xent) that
needs no similarity labels. It runs on an NVIDIA GPU (built and tested on an RTX 5080 /
Blackwell, CUDA 13, using channels-last + mixed precision for Tensor-Core speed).

### The result: deep learning wins once it has data

Trained with a self-supervised contrastive objective (no genre labels), a ResNet encoder
learns an embedding space where acoustically similar songs cluster. Evaluated with a kNN
genre probe (chance ≈ 0.28 for 16 genres):

| Model | Training data | kNN genre acc | vs baseline |
|-------|---------------|---------------|-------------|
| Chance (majority class) | — | 0.284 | — |
| Our neural embedding | Deezer 475 | 0.25 | **loses** to baseline |
| Pooled-mel baseline (no ML) | FMA 25k | 0.521 | — |
| Our neural embedding | **FMA-medium 25k** | **0.601** | **+8 pts** |
| Pooled-mel baseline (no ML) | FMA 106k | 0.507 | — |
| Our neural embedding | **FMA-large 106k** | **0.641** | **+13 pts** |

The story is the scaling curve. On **475 tracks the neural net *lost*** to a trivial pooled-mel
baseline (0.25). At **25k tracks it beats the baseline by +8** (0.601). At **106k tracks it wins
by +13** (0.641) — the more data, the wider the margin, exactly as contrastive deep learning
predicts. On FMA-large, **57%** of tracks have a same-genre nearest neighbor in the learned
space (from a model that never saw a label), and it trains on the ~57k *unlabeled* FMA tracks
too (self-supervised needs no labels) while being evaluated only on the ~49.6k labeled ones.

![FMA-large results](docs/fma_large_results.png)

*FMA-large (106k tracks): loss falls while the genre-probe accuracy rises; the UMAP shows
Electronic (top), Rock/Pop (bottom) and a tight Old-Time/Historic cluster; per-genre retrieval
reaches Old-Time 93%, Rock 74%, Classical 67%, Hip-Hop 64%.*

Engineering notes for the 106k run: the 14 GB packed dataset exceeds the 5080's 16 GB VRAM, so
training auto-switches to a **CPU-resident** mode (dataset pinned in RAM, batches streamed to
the GPU) that still keeps it at 99% utilization. The download used aria2 with 16 connections
(~138 MB/s vs ~13 MB/s single-stream). Spectrogram precompute runs across all CPU cores.

### Reproduce it

```bash
# 1. Get FMA (audio + metadata) — see https://github.com/mdeff/fma
#    fma_medium.zip (~22GB) or fma_large.zip (~93GB), plus fma_metadata.zip.
#    Unzip with 7-Zip (the archives use the deflate64 format).
# 2. Build a manifest, pack spectrograms, train, evaluate, visualize:
python -m soundalike.ml.fma --audio-dir FMA/fma_large --metadata FMA/fma_metadata/tracks.csv --subset large --out manifest.csv --include-unlabeled
python -m soundalike.ml.precompute --manifest manifest.csv --spec-dir specs --workers 22
python -m soundalike.ml.pack --manifest manifest.csv --spec-dir specs --out packed.npz
python -m soundalike.ml.train_fast --packed packed.npz --out-dir ml_data/model_fma --epochs 45
python -m soundalike.ml.evaluate --embeddings ml_data/model_fma/embeddings.npz
python -m soundalike.ml.visualize --model-dir ml_data/model_fma

# Recommend with the trained model:
soundalike learned-similar --title "Blinding Lights" --artist "The Weeknd" --model-dir ml_data/model_fma
```

`train_fast` auto-detects whether the dataset fits in VRAM: it stays GPU-resident when it fits
(FMA-medium) and streams from pinned CPU RAM when it doesn't (FMA-large). Smaller quick-start
commands (`soundalike.ml.collect` / `train` / `map`) run the same pipeline on a few hundred
Deezer previews with no external download — handy for a fast sanity check.

The low-level GPU tooling is its own learning artifact:

```bash
python -m soundalike.ml.gpu          # inspect which cuDNN conv algorithm gets selected
```

---

## Python API

```python
from soundalike import ContentBasedRecommender, FeatureConfig, load_bundled_dataset

rec = ContentBasedRecommender(
    FeatureConfig(weights={"energy": 2.0}, metric="euclidean")
).fit(load_bundled_dataset())

for r in rec.similar_to("Blinding Lights", n=5):
    print(r.title, "—", r.artist, round(r.score, 3))
```

---

## How the content engine works

1. Each song becomes a vector of audio features: `bpm, danceability, valence, energy,
   acousticness, instrumentalness, liveness, speechiness`.
2. Features are **standardized** (z-score) so `bpm` (~40–220) and the 0–100 percentages are
   comparable — the step the original project skipped.
3. Optional per-feature **weights** emphasize what you care about.
4. Similarity is computed with Euclidean distance (default) or cosine. A taste profile is the
   centroid of your seed songs; recommendations are the nearest songs, excluding what you fed in.

---

## Dataset

The bundled `spotify_data.csv` (also at `src/soundalike/data/`) has one row per song:

`title, artist(s), release, num_of_streams, bpm, key, mode, danceability, valence, energy,
acousticness, instrumentalness, liveness, speechiness`

Any CSV with the audio-feature columns works via `--dataset path.csv` or `Dataset.from_csv`.

---

## Project structure

```
src/soundalike/
  dataset.py        # load/normalize songs, match by title/artist
  features.py       # feature list, aliases, weighting config
  recommender.py    # ContentBasedRecommender
  profile.py        # parse seed lists (text/CSV/inline)
  cli.py            # the `soundalike` command
  config.py         # .env-based config (never commits secrets)
  spotify/          # OAuth PKCE + Web API client (no deprecated endpoints)
  lastfm/           # similar-tracks client + cross-catalog recommender (optional)
  audio/            # ⭐ acoustic DSP engine: previews (Deezer), librosa features,
                    #    feature cache, acoustic-similarity recommender;
                    #    vibe engine (bands + dynamics) + a bundled ~1,500-song library
  ml/               # GPU research track: gpu (cuDNN inspector), collect, spectrogram,
                    #    model (CNN/ResNet + NT-Xent), data, train, supervised, evaluate,
                    #    fma (dataset loader), precompute, pack, train_fast (GPU-resident),
                    #    map, visualize, recommend
tests/              # pytest suite (offline + network-free live/audio/ml logic)
spotify_program.py  # the original first-year project, kept for posterity
```

Run the tests:

```bash
pytest -q
```

---

## Roadmap

- [x] Content-based recommender (offline, tested)
- [x] Spotify OAuth (PKCE) + library/top/recent fetch + playlist export
- [x] Last.fm cross-catalog similarity (optional)
- [x] **Acoustic DSP engine** — measure features from real audio, rank by science
- [x] **GPU training pipeline** — dataset harvest, CNN/ResNet encoder, contrastive + supervised, cuDNN inspector
- [x] **Scaled to FMA-medium (25k)** — kNN 0.601; beats the no-ML baseline by +8 pts
- [x] **Scaled to FMA-large (106k)** — kNN 0.641; beats the baseline by +13 pts (CPU-resident training)
- [ ] Persist an acoustic-feature store so results play/save back to Spotify at scale
- [ ] Human-in-the-loop rating loop to score recommendation quality
- [ ] Hybrid ranking (acoustic + learned embedding) and DJ-style harmonic sequencing
- [ ] Optional web UI

Contributions welcome — this is meant to be community-built.

## License

MIT — see [LICENSE](LICENSE).
