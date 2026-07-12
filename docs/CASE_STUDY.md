# soundalike — Engineering Case Study

> How a first-year university script became a multi-engine music recommender with a
> self-supervised deep-learning model trained on 106,000 songs.

This document is the story behind the code: the problem, the constraints, the design
decisions, the walls I hit, and how I got through them. The polished "what it is and how to
run it" lives in the [README](../README.md); this is the "how it was built and why."

---

## TL;DR

- **Started with:** a ~180-line first-year terminal script that read a static CSV of songs and
  printed min/max/mean statistics.
- **Ended with:** an installable Python package with **six recommendation engines**, a live
  Spotify integration (OAuth PKCE, no passwords), and **GPU-trained audio-embedding neural
  networks** — a contrastive FMA encoder, a **vibe-aware** encoder that learns a song's bass profile
  and dynamics, and an **artist-aware** encoder fine-tuned on ~87,000 real songs — feeding a
  bundled, out-of-the-box recommender.
- **Headline result:** the learned model's genre-probe accuracy scales with data —
  **0.25 → 0.601 → 0.641** as the training set grows **475 → 25,000 → 106,000** tracks — going
  from *losing* to a no-ML baseline to beating it by **+13 points**.
- **Vibe result:** a multi-task "vibe-aware" encoder raises how much vibe its embedding space
  encodes from **linear-probe R² 0.82 → 0.94** on 1,738 held-out real songs, with the biggest
  gains on bass and dynamics — the qualities that define whether two songs *feel* the same.
- **Scale result:** growing the library to ~87k songs across every genre exposed the *encoder* as
  the bottleneck; a domain-matched **artist-aware** fine-tune, a **higher-dimensional embedding**
  (256→384) and embedding **whitening** turned incoherent cross-genre matches into scene-coherent
  ones (Miles Davis → Brad Mehldau/Lee Morgan; Explosions in the Sky → This Will Destroy You/Mono;
  NewJeans → CHUU/LOONA, not random pop).
- **Objective + validation result:** a controlled 5-seed sweep found an **ArcFace + GeM** encoder
  that beat supervised-contrastive by **+23% on same-artist mAP** — but validating it against
  *independent human behavior* (ListenBrainz co-listening + Deezer related-artists) revealed it
  **regressed real cross-artist recommendation** (and botched niche genres like city pop/hyperpop).
  An internal metric had rewarded the wrong thing, so I **reverted** and built a `cross_artist_agreement`
  metric that measures inter-artist geometry — the honest "measure, ship, re-measure, revert" loop.
- **Human-quality retrieval result:** a 107-pair, component-disjoint protocol exposed another
  failed generalization. The locked audio-only method improved DEV primary 0.01506→0.02558, but
  the once-opened 40-pair FINAL test moved one counterpart to rank 14, with primary 0→0.000595
  and a paired 95% interval touching zero. It was rejected and not deployed.
- **Built and validated on:** an NVIDIA RTX 5080, the full automated suite, and a clean packaged
  wheel.

---

## 1. The problem

The original project (`spotify_program.py`) was a good learning exercise but fundamentally
limited: it read one static 855-row CSV and computed aggregate statistics. The goal was to turn
it into something real — **a tool that finds songs that genuinely sound like the ones you
like**, better than the mediocre "song radio" features that already exist.

### The constraint that shaped everything

The obvious approach — ask Spotify's API for similar songs and audio features — is **no longer
possible**. On **2024-11-27, Spotify removed** the Recommendations and Audio Features endpoints
for all new apps ([official announcement](https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api)).
Those are exactly the endpoints this idea would normally depend on.

That constraint became the project's defining design driver: **if you can't get similarity or
audio features from Spotify, you have to compute them yourself.** That's not a limitation — it's
the whole point of what makes this interesting.

---

## 2. Architecture: five engines, one idea

Every engine answers the same question — "what sounds like this?" — but from a different signal
and with different tradeoffs.

| Engine | Signal | Credentials | Coverage |
|--------|--------|-------------|----------|
| **Deep-vibe** ⭐ | Vibe-aware neural embedding **fused** with measured bass/dynamics | None | Bundled ~1,700-song library |
| **Vibe** | Frequency-band balance + dynamics, vs a ~1,500-song library | None | Real, listenable songs |
| **Acoustic DSP** | Features measured from the raw waveform (librosa) | None | Any track with a preview |
| **Content-based** | Audio-feature vectors, standardized + weighted | None | Bundled dataset |
| **Learned model** | A CNN trained to embed audio (contrastive) | None | Whatever it's trained on |
| **Live Spotify / Last.fm** | Your real listening + optional crowd data | Free API keys | Your library / any track |

A deliberate design principle runs through the acoustic engines: **ranking is done purely by
the sound.** A music catalog (Deezer) is used *only* to enumerate candidate songs and fetch their
audio — never to decide what's "similar." That keeps the recommendations grounded in acoustics
rather than "people who listened to X also listened to Y," which is what every existing tool
already does.

---

## 3. The machine-learning story (the interesting part)

The most ambitious engine trains a neural network to place similar-sounding songs near each
other in an embedding space. It's **self-supervised** using a contrastive objective (NT-Xent,
the SimCLR loss): two augmented snippets of the *same* track are pulled together while snippets
of *different* tracks are pushed apart. This needs **no similarity labels** — which is essential,
because nobody hands you ground-truth "these two songs are similar" data.

### Act 1 — an honest failure

The first attempt trained on ~475 songs harvested from free previews. The result was humbling
and instructive: **the neural network lost to a trivial baseline.** A no-learning approach
(mean+std pooling of the spectrogram) recovered genre at **0.375** accuracy; the neural net
managed only **0.25** — barely above the **0.24** chance rate.

This is textbook behaviour: **contrastive deep learning is data-hungry.** With only a few
hundred examples the model can tell individual clips apart without ever learning what makes a
*genre* cohere. Rather than hide this, I measured it explicitly and treated it as the signal to
scale up. (I also confirmed genres *were* separable from audio — the baseline's 0.375 ≫ 0.24
chance proved the signal existed; the model just needed more data to capture it.)

### Act 2 — scaling to real data

The [Free Music Archive](https://github.com/mdeff/fma) provides labeled audio at scale. I trained
on FMA-medium (25k tracks) and then FMA-large (106k tracks), evaluating with a **kNN genre
probe** — freeze the embeddings, then see how well a simple classifier recovers genre from them.
A model that has learned real musical structure will score well above chance.

| Training data | Neural kNN | No-ML baseline | Chance | Verdict |
|---------------|-----------|----------------|--------|---------|
| 475 tracks | 0.25 | 0.375 | 0.24 | **loses** to baseline |
| FMA-medium 25,000 | 0.601 | 0.521 | 0.28 | **+8 points** |
| FMA-large 106,000 | **0.641** | 0.507 | 0.29 | **+13 points** |

**The scaling curve is the whole story.** The more data, the wider the neural network's margin
over the baseline — precisely what the theory predicts. At 106k tracks, **57% of songs have a
same-genre nearest neighbor** in the learned space, from a model that never saw a single label
during training.

![FMA-large results](fma_large_results.png)

*Training on 106k tracks: loss falls as the genre-probe accuracy rises (left); the embedding
space forms visible clusters — Electronic at top, Rock/Pop at bottom, a tight Old-Time/Historic
island (middle); per-genre nearest-neighbor retrieval reaches Old-Time 93%, Rock 74%, Classical
67%, Hip-Hop 64% (right).*

### Does it actually recommend well?

The numbers are backed up by qualitative results. Querying the 106k model with mainstream songs
it has never seen (it maps them into the learned space and finds neighbors in the FMA catalog):

- **"Lose Yourself" — Eminem** → 5 of 6 neighbors are labeled **Hip-Hop**.
- **"Clair de Lune" — Debussy** → a Beethoven piano sonata (**Classical**) and other solo-piano
  instrumental tracks.
- **"Bohemian Rhapsody" — Queen** → folk/acoustic ballad tracks, matching its ballad sections.

The model learned to discriminate by acoustic character — rap finds rap, classical finds
classical — purely from the physics of the audio.

---

## 4. Engineering challenges (and how I solved them)

The interesting problems weren't the ML — they were the systems engineering around it.

### Challenge: the GPU was starving

The first FMA-medium training run pinned the GPU at **9% utilization**. Sampling `nvidia-smi`
over time showed a sawtooth: brief bursts to 98% then long stalls — a classic data-loading
bottleneck. The root cause was random reads of 25,000 tiny spectrogram files from a slow
network-mounted drive (measured at ~20 files/second).

**Fix:** I built a consolidation step (`pack.py`) that packs every spectrogram into a single
compact `float16` array, and a training path (`train_fast.py`) that loads the **entire dataset
into VRAM once** and does augmentation *on the GPU*. Result: **99% utilization, 37s/epoch** — the
bottleneck vanished.

### Challenge: the dataset didn't fit in VRAM

Scaling to 106k tracks made the packed dataset **14 GB** — too big to sit in the 5080's 16 GB
VRAM alongside the model. Rather than fail or shrink the data, I made the trainer **auto-detect
its data residency**: it keeps the dataset GPU-resident when it fits (FMA-medium) and switches to
**pinned CPU RAM with per-batch PCIe streaming** when it doesn't (FMA-large). A batch is only
~17 MB, so the transfer overlaps compute and the GPU still runs at **99% utilization**.

### Challenge: downloading 93 GB

FMA-large is a 93 GB archive. A single-stream download ran at ~13 MB/s (~2 hours). I switched to
**aria2 with 16 parallel connections**, hitting **~138 MB/s — an ~11x speedup** (~11 minutes).
Along the way I also had to diagnose and recover from a corrupted download (two writers hitting
the same file) and abandon a problematic drive that blocked executable launches.

### Challenge: understanding the hardware

Out of genuine curiosity about how NVIDIA's libraries pick low-level algorithms, I built a
**cuDNN solver-selection inspector** (`ml/gpu.py`). It surfaces which CUDA kernel cuDNN chooses
for a given convolution — revealing it selected a **TF32 Tensor-Core `cutlass` kernel in NHWC
layout**, plus the layout-transpose kernels that overhead implies. I then demonstrated the
optimization ladder empirically: **NCHW → channels-last (1.34x) → fp16 + channels-last (4.2x)**,
and folded channels-last + mixed precision into training so the 5080 runs near its Tensor-Core
peak.

---

## 5. Iterating from real feedback: the "vibe" engine

The best case study material comes from a feature that *didn't* work at first.

A test query — "find songs like *Wasting Time* by eric404," a hyperpop track with quiet vocals
and a heavy dubstep drop — returned soft acoustic bedroom-pop. Wrong vibe entirely. Rather than
hand-tune, I **measured why**. I analysed the seed and the bad recommendations directly:

| Song | sub-bass % | dynamic range | crest (peak/avg) |
|------|-----------|---------------|------------------|
| *Wasting Time* (the seed) | **73%** | **0.39** | **2.21** |
| a "soft" recommendation | 45% | 0.25 | 1.61 |

The data made the bug obvious. The seed is overwhelmingly **sub-bass** and has **~2× the dynamic
range and crest** (the peak-vs-average spikiness that *is* the drop). But the original engine
**averaged every feature over the whole 30-second clip** — so the quiet intro and the loud drop
blurred into a bland "medium," and the sub-bass dominance wasn't modelled at all. It was blind to
exactly the qualities that define the vibe.

**The fix** was a new feature set that measures what the averages hide:

- **Frequency-band balance** — energy split across seven bands (sub → air), i.e. the literal
  "how much bass, how much highs."
- **Dynamics** — standard deviation, dynamic range, and crest factor of the loudness envelope,
  which capture "does this track have drops?"

These are weighted so the low-end and the dynamics dominate the match, and ranked against a
bundled library of ~1,500 real, diverse songs. The result: the same query now correctly reads
*"123 BPM, very dynamic (big drops), bass-heavy"* and returns hyperpop/electronic tracks in the
right scene (aldn, Flume, Slow Magic). This is the engineering habit that matters most —
**diagnose with data before you change code**, and let the measurement design the fix.

### From hand-crafted vibe to *learned* vibe

The hand-crafted vibe vector works, but it raised a sharper question: could the **neural encoder
itself** learn to represent vibe, instead of relying on hand-weighted features bolted on
afterwards? The plain contrastive encoder is good at timbre but, as the R² numbers below show,
only partly captures bass and dynamics.

So I trained a **vibe-aware encoder** with a multi-task objective: the self-supervised contrastive
loss **plus** an auxiliary head that must predict a 10-dim *vibe target* — seven frequency-band
fractions, loudness dynamics (std + range), and spectral centroid — computed directly from each
song's mel-spectrogram. Predicting that target from a short crop forces the embedding to encode
*how the whole song sounds and moves*. Crucially, the target is derived from the **already-packed
FMA spectrograms**, so the vibe-aware model trains on all 106k songs with **zero re-downloading**
(~131 min on the 5080).

To measure whether it worked, I used a **linear probe**: fit a ridge regression from each
encoder's frozen embeddings to the vibe target on 1,738 held-out real songs, and report
cross-validated R². A linear probe is the standard, honest test of "is this information linearly
present in the representation?"

| Vibe dimension | Baseline encoder | Vibe-aware encoder |
|----------------|------------------|--------------------|
| **Overall (10-dim)** | **0.82** | **0.94** |
| Bass | 0.73 | **0.96** |
| Loudness dynamics | 0.70 | **0.89** |
| Drop size (dynamic range) | 0.70 | **0.85** |

The vibe-aware encoder wins on *every* dimension, and the largest gains are exactly on **bass and
dynamics** — the qualities the original engine was blind to and the ones that decide whether two
songs feel the same. That encoder is what ships in the package and powers the deep-vibe engine.

![Vibe-aware encoder results](vibe_aware_results.png)

A second engineering payoff came out of building this: I split "download a preview" from "embed
it" with a **spec cache**. The library's mel-spectrograms are harvested from Deezer *once* (rate
-limited, resumable, checkpointed) and stored; re-embedding the whole 1,738-song library with a
newly trained encoder is then a local, offline, seconds-long operation. Swapping in a better model
no longer costs an hour of rate-limited downloading — which is what made the baseline-vs-vibe-aware
comparison above a fair, apples-to-apples test on an identical song set.

---

## 6. Scaling the library exposed the real bottleneck (and how I fixed it)

Testing on a niche seed (*Lovers Rock* by TV Girl) returned generic pop — because the bundled
library, curated for the earlier hyperpop test, simply had no dream-pop neighbours. So I **grew the
library in waves — ~1,700 → ~25,000 → ~55,000 → ~87,000 songs** across every scene, crawling the
Deezer **related-artist graph** two hops out from a ~400-artist multi-genre seed list (deliberately
over-sampling niches the charts miss: K-pop and city-pop, Afrobeats, French and Latin rap, techno/
house/DnB, phonk and synthwave, post-rock, shoegaze, black/death metal, jazz, classical, blues,
gospel, reggae). (Deezer's genre endpoints turned out to be useless — they ignore the id and return
the same global list — so the related-artist graph, which *is* genre-coherent, did the work.) Four
engineering details made the harvest practical: a **candidate sidecar** so a restart never re-does
the slow gather; **thread-pool downloads** (the box was 93% idle at 0.8/s single-threaded → ~6/s
across 10 workers); the discovery that Deezer **preview URLs are signed and expire**, so the worker
fetches a fresh URL by track id right before downloading (this alone took the success rate from 0%
back to 100%); and a **dedup pass** that collapses remaster/sped-up/remix/karaoke variants of the
same song to one row, so a seed can't match five copies of one track.

But growing the library made recommendations **worse**, which was the most instructive result of
the whole project. A bigger, more diverse pool contained more songs that were *texture-similar but
vibe-wrong*, and the FMA-trained encoder — trained on mostly instrumental Creative-Commons music —
happily surfaced them (a dream-pop seed matched Creed and Metallica). **The library was never the
ceiling; the encoder was.** Three fixes, two at train time and one at inference:

1. **An artist-aware encoder.** I fine-tuned the encoder on the harvested songs with a
   **supervised-contrastive** objective using the *artist* as the label (PK-sampled batches; songs
   by the same artist are positives), plus the vibe-target auxiliary. "Same artist ⇒ similar" is a
   free, strong style signal, and because the library was crawled through related artists it
   generalizes to *neighbouring* artists. It trains on the cached spectrograms in ~40 min on the
   5080.

2. **A higher-dimensional embedding.** When the library passed ~50k songs, precision on
   already-strong seeds *softened* — a bigger pool means more competing look-alikes crowding a
   fixed-size space. Widening the embedding from 256 to 384 dimensions (which barely changes compute
   — it's just the final projection — and keeps the bundled index under GitHub's 100 MB limit) gave
   the space room to separate ~87k songs, and precision recovered while coverage kept improving. The
   384-d base also scored higher on the held-out genre probe (kNN 0.617 vs 0.606). I also tried
   **512-d** to see if bigger was better still — it wasn't: on the recommendation benchmark it matched
   384-d on precision and was *slightly worse* on coverage (0.445 vs 0.463), at +33% size and memory,
   and its genre-probe kNN actually dropped to 0.609. So 384-d is the measured sweet spot, and the
   encoder's *capacity* is no longer the bottleneck — a useful negative result that says "don't just
   make it bigger."

3. **Whitening.** The embeddings piled into a tight cone (every pair ~0.9 cosine), so raw cosine
   couldn't rank finely. ZCA-whitening the space at load time removes the dominant shared direction
   so similarity keys on what's *distinctive*.

The combined effect, on identical seeds:

| Seed | FMA encoder, raw cosine | Artist-aware 384-d + whitening |
|------|--------------------------|--------------------------------|
| *So What* — Miles Davis | mixed | Brad Mehldau, Lee Morgan, Ahmad Jamal |
| *Your Hand in Mine* — Explosions in the Sky | mixed | If These Trees Could Talk, This Will Destroy You, Mono |
| *Ditto* — NewJeans | mixed | CHUU, LOONA (K-pop) |
| *HUMBLE.* — Kendrick | mixed | Kodak Black, JID, $uicideboy$ |

It's now genuinely scene-coherent across jazz, post-rock, metal, hip-hop, R&B, electronic, indie,
bedroom-pop, K-pop and ambient — including whole scenes (jazz, post-rock, phonk, city-pop) that
simply weren't in the library before. The instructive arc is the coverage-vs-precision tension:
scaling the library helped coverage but *hurt* precision until the encoder was given more capacity
to match — a reminder that "more data" and "better model" are different levers. The throughline is
the same engineering habit as the vibe engine:
**let the failure tell you where the real bottleneck is**, and don't mistake "more data" for
"better model."

### Putting a number on "how big should the library be?"

Rather than keep guessing, I built a label-free benchmark (`soundalike.ml.benchmark`) that measures
the trade-off directly. Holding a song and one same-artist sibling fixed and adding only
*distractors*, **fixed-pair recall@10 falls from 0.17 at 5k to 0.04 at 86k** — a bigger pool does
bury a specific sibling. Meanwhile **held-out nearest-neighbour cosine (coverage) rises from 0.36 to
0.46** — a bigger pool means something close almost always exists. The curves cross near 20k and both
flatten past ~40k.

![Library size vs quality](library_size_sweep.png)

So the "perfect balance" isn't a single number — it depends on which failure you care about. I chose
**coverage-first (~87k)** deliberately: the failures users actually notice are coverage failures (a
niche seed returning nothing in-scene), and the precision cost is better recovered with **smarter
ranking than with a smaller library**. That's what the new `--diversity` (MMR re-ranking),
`--max-per-artist`, and multi-seed *taste-blend* features do — keep the top-K varied and on-point
without throwing away whole scenes. The bundle is also GitHub-capped near ~100 MB, so ~87k is close
to the practical ceiling regardless. The point isn't the exact size; it's that the decision is now
*measured and defensible* instead of a hunch.

### The objective is the lever — a controlled encoder sweep

If the encoder is the ceiling, the obvious question is *how do you raise it?* Rather than guess, I
built a trustworthy head-to-head metric — same-artist **mean average precision** (`score_embeddings`
whitens exactly as production does, then reports mAP + recall@10 + coverage in one call) — fixed a
5-seed baseline, and ran each idea as a controlled experiment where the objective is the only
variable. The result overturned my intuition: **capacity is not the bottleneck; the objective is.**

| Variation | mean mAP (5 seeds) | vs baseline | verdict |
|-----------|:---:|:---:|---|
| Supervised-contrastive, 384-d *(previous ship)* | 0.0396 | — | baseline |
| 512-d encoder | — | worse | ❌ capacity isn't the lever (see §8 note) |
| 3-encoder ensemble (concat) | 0.038–0.040 | −2 to −7% | ❌ combining encoders hurt precision |
| **ArcFace** (additive angular margin) | 0.0477 | **+20%** | ✅ objective *is* the lever |
| **ArcFace + GeM pooling** | **0.0486** | **+23%** | ⚠️ shipped, then **reverted** (see below) |
| ArcFace + GeM, margin 0.3 | 0.0488 | +23% | ➖ tie on mAP, *worse* on the NN probe → rejected |

Two findings drove the (initial) ship. **ArcFace** replaces the plain contrastive push/pull with an
additive angular margin, forcing each song tighter around its artist prototype and further from every
other — a +20% mAP jump on its own. **GeM pooling** swaps the encoder's flat spatial average for a
learnable generalized mean, so the network chooses how peaky its per-clip summary is; interestingly it
learned an exponent *below* 1 (softer than average), and added another ~2%. Pushing the margin higher
(0.3) was a statistical tie on mAP but *regressed* the independent same-artist NN probe — a clean
signal that 0.2 is the sweet spot for noisy related-artist labels, not a number to keep cranking.

On same-artist mAP and on a first qualitative glance it looked great — so I shipped it. Then I did
what you should always do with an *internal* metric: I checked it against the outside world.

### Ship, re-measure, revert: external validation caught a regression

Same-artist mAP asks "are a song's own siblings near it?" That rewards packing each artist into a tight
ball — but a recommender never returns the seed's own artist; it returns *other* artists. So I validated
the shipped encoder against two **independent human-behavior** ground truths, over 24 mainstream *and*
niche seeds: **ListenBrainz** co-listening (people who listen to X also listen to Y) and **Deezer**
related-artists. For each seed I measured the fraction of our recommended artists that real listeners
corroborate, against a random-library baseline.

| Ground truth (independent of our audio) | ArcFace+GeM (shipped) | Supervised-contrastive (old) | Random |
|---|:---:|:---:|:---:|
| ListenBrainz co-listening (24 seeds) | 0.117 | **0.161** | 0.004 |
| Deezer related-artists (24 seeds) | 0.058 | **0.100** | 0.001 |
| Deezer centroid geometry (116 artists) | 0.233 | **0.252** | — |

Both encoders are 26–135× better than random, so both are genuinely sensible — but the **old
supervised-contrastive encoder agreed with real listeners more, on every measure.** Qualitatively the
gap was worst exactly where it hurts: **city pop** (*Plastic Love* — Mariya Takeuchi: old → Hiroshi
Sato, T-Square, Anri, Momoko Kikuchi; ArcFace → Dream Theater, Eric Clapton) and **hyperpop** (*100
gecs*: old → SOPHIE, Dorian Electra; ArcFace → Rezz, Diplo). ArcFace's aggressive artist-separation had
sharpened same-artist retrieval while *distorting the inter-artist geometry that recommendation depends
on* — a metric optimizing the wrong thing.

So I **reverted to the supervised-contrastive encoder** and added `cross_artist_agreement` to the
benchmark: it builds each artist's centroid, ranks the nearest *other*-artist centroids, and scores
overlap against a human related-artist map — the North Star same-artist mAP had missed. The ArcFace/GeM
trainer and pooling stay in the tree as a documented negative result. The lesson is the most valuable
artifact here: **an internal metric is a hypothesis, not a verdict — validate against the real world
before you trust it, and be willing to unship.**

It shows up qualitatively for the *kept* (supervised-contrastive) encoder, exactly where the FMA
encoder was weakest. *Plastic Love* — Mariya Takeuchi returns genuine city pop (Hiroshi Sato,
T-Square, Anri); *OMG* — NewJeans surfaces K-pop neighbours; jazz and black-metal seeds return
scene-royalty. The remaining weak spot (both encoders): ultra-niche breakcore seeds (*Sewerslvt*) leak
into trance — a candidate for the next objective iteration, now measurable via `cross_artist_agreement`.

---

## 7. Ranking quality: the once-opened audio test

Iteration 3 fixed category leakage and deployed a manually coherent guardrail, but its advertised
+88.3% retrieval result was one additional counterpart at rank 37. Recall@10 fell 0.05→0, MRR
fell 0.00625→0.00590, and the paired interval [-0.00256, 0.07703] crossed zero. The held-out set
had also been reused for challenger selection, and removing Wikipedia/notability priors reduced the
reported method to zero hits. It was not evidence that the +20% goal had been met.

### A new benchmark before new training

Version 5 was built before fitting iteration-4 models. It contains 107 deciding Category-A pairs
across 85 named scenes and stores URLs, short excerpts, source classes, access dates, and category
rationales. A post-freeze audit checked 190 unique URLs: 186 returned 200, two were
publisher-access-controlled, and two stale slugs have verified canonical/archive replacements in the
audit artifact. Samples, interpolations, covers, remixes, legal allegations, and weak listicles do not
enter the metric. Named criticism/participant accounts provide the editorial rows; the independent
ListenBrainz similar-recordings dataset supplies human-listening musical-similarity rows, with its
session-based limitation stated explicitly. MusicBrainz recording IDs verify identity.

The 67 development and 40 FINAL pairs share no artist or transitive artist component. Tracks and
artists are unique, and popular, deep-cut, and niche strata are present. Every prior v4 held-out row
is development-only. Before training, the protocol froze and hashed:

- `benchmarks/soundalike_pairs.v5.json`;
- the 40-pair FINAL manifest;
- ranked outputs for pre-goal production, iteration-3 deployed, raw encoder, and all-priors-zero
  audio baselines; and
- a signed state document with `final_open_count = 0`.

The primary was predeclared as `mean(NDCG@10, MRR, Recall@10)`. Success required ≥20% relative
and positive absolute gain, non-regressing Recall@10 and MRR, paired-bootstrap CI above zero, at
least five improved FINAL pairs, and no scene regression beyond 10%. The selected configuration,
checkpoint hashes, and DEV report were locked before the state machine permitted one opening. Review
then hardened the transition further: future runs must enter `RANKINGS_LOCKED`, binding the exact
target-agnostic ranking file hash before `open-final`. The historical failed run generated that file
14 seconds before opening but predated the explicit transition; this limitation is recorded in state.

### Materially different audio experiments

All models excluded every benchmark artist where catalogue rows were used for training. Titles,
artists, Wikipedia/pageviews, popularity, manual pairs, benchmark identities, and FINAL outcomes
were never model inputs.

| Approach | Substantial training/index run | DEV result |
|---|---|---|
| Cross-artist FMA SupCon ResNet | 153,600 examples, 1,200 updates; 272,853-row 128-d index | R@10 0; R@50 0.0149 |
| FMA BYOL ResNet | 102,400 examples, 800 updates; full 128-d index | no top-50 hit |
| EfficientNet/CLAP audio distillation | 256,000 examples, 1,000 updates; 15,570 benchmark rows excluded; full 128-d index | candidate R@1000 0.145, no top-50 hit |
| Independent-pair audio metric/reranker | 3,967 cross-artist positives with zero benchmark-artist overlap | useful only as a two-stage head |
| Three-window MaxSim | 3×128 vectors for all 272,853 tracks over a five-signal candidate union | union R@1000 0.217, but reranker had no top-50 hit |

Candidate recall was the main failure: the learned FMA models usually never admitted the known
counterpart to the first 1,000 candidates. Distillation admitted more counterparts but could not
order them usefully. The locked method therefore used a learned audio-only candidate head and a
rank-stratified tail across EfficientNet, distilled audio, SupCon, and CLAP. The tail is a candidate-
recall mechanism, not a metadata prior.

### DEV selection

| Development Category-A metric | Pre-goal production | Locked audio method |
|---|---:|---:|
| Recall@10 | 0.02985 | **0.04478** |
| Recall@50 | 0.02985 | **0.07463** |
| MRR | 0.00485 | **0.01254** |
| NDCG@10 | 0.01048 | **0.01944** |
| Primary | 0.01506 | **0.02558** |

Relative DEV gain was **+69.9%**. Four pairs improved, none worsened, and the paired primary-delta
95% interval was **[0.000116, 0.028508]**. This was sufficient to lock the configuration; no FINAL
rank or metric had been read.

### FINAL opened once — failure

The state machine opened FINAL at `2026-07-12T08:46:37Z` and immediately finalized it. A second
open is rejected programmatically. The finalized state and bound ranking hash are sealed by a
detached Ed25519 signature (`protocol-v5/state.sig`); the private key was not retained, and the
committed public key verifies the artifact independently.

| Once-opened FINAL (40 pairs) | R@1 | R@5 | R@10 | R@20 | R@50 | MRR | NDCG@10 | Primary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pre-goal production | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Iteration-3 deployed | 0 | 0 | 0 | 0 | 0.025 | 0.000833 | 0 | 0.000278 |
| Raw encoder | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| All-priors-zero audio ablation | 0 | 0 | **0.025** | **0.025** | **0.025** | **0.003125** | **0.007887** | **0.012004** |
| Locked method | 0 | 0 | 0 | 0.025 | 0.025 | 0.001786 | 0 | 0.000595 |

The locked method moved one pair to rank 14. Its absolute primary delta was 0.000595; the paired
95% interval **[0, 0.001786]** includes zero and P(delta>0)=0.6364. The per-scene no-regression
check passed (all production scene contributions were zero), but one improved pair is below the
predeclared five-pair minimum. The production baseline's zero primary makes a relative percentage
undefined in practical terms. The all-priors-zero audio baseline found one pair at rank 8 and was
stronger than the locked method at useful ranks, but it also failed confidence and count criteria.

**Verdict: FINAL failed.** It was not reopened, no hyperparameter was changed afterward, and no
iteration-4 model was deployed. A future attempt requires a new protocol and new component-disjoint
FINAL set.

### Product and resource decision

Production remains the prior `dual_sonic64_guardrail`, whose guarded top five retained 17/20 direct
manual UX passes versus 11/20 for the original baseline. That result is secondary and subjective;
it is not blended with retrieval and is not described as clearing the goal. The current site and
release are unchanged, so search and preview behavior were not put at risk by a failed experiment.

Training used an RTX 5080: SupCon 617.84 s, BYOL 390.55 s, distillation 319.65 s. Full-catalog
embedding passes took 38.66–43.05 s; the 3-window index took 109.73 s. The three selected float16
matrices total 209,551,488 bytes plus a 7,813-byte scorer. With the 299,288,526-byte base index,
the research package would be 508,847,827 bytes. Measured cold load was 10.86 s; process RSS rose
2.21 GB to 2.85 GB, uncomfortably close to the hosted 3 GB limit. Twenty local rankings averaged
79.2 ms (p95 92.1 ms). These are research assets only because FINAL failed; the deployed index is
unchanged.

Reproduction and immutable evidence:

```powershell
$env:PYTHONPATH = "src;."
.\.venv\Scripts\python.exe -m soundalike.ml.final_protocol freeze `
  --benchmark benchmarks\soundalike_pairs.v5.json `
  --index ml_data\deepvibe_index_v5.npz `
  --protocol-dir .goals\human-quality-recommendations\protocol-v5

# Training/evaluation APIs are in soundalike.ml.audio_experiments.
# `lock` and `open-final` now refuse because this protocol is FINALIZED.
.\.venv\Scripts\python.exe -m pytest tests\ -q
```

Exact records are in `artifacts/audio-dev-results-v5.json`, `artifacts/final-once-v5.json`, and
`protocol-v5/state.json`.

---

## 8. Security & correctness

- **No passwords, ever.** Live Spotify access uses OAuth 2.0 **Authorization Code + PKCE** with a
  local loopback callback, CSRF `state` validation, and cached auto-refreshing tokens.
- **No secrets in git.** Credentials live only in a git-ignored `.env`; the repo ships a
  `.env.example` template.
- **No data leakage in training.** The 107-row Category-A benchmark has a 40-pair FINAL set
  separated from development by connected artist components. The distillation and independent-pair
  trainers exclude every benchmark artist; tests reject artist/track duplicates and split overlap.
  FINAL was opened once after checkpoint/configuration hashes were locked.
- **Real-index derivative audit.** On 272,853 rows the filter removes 1,361 candidates. An
  independent pattern set found 348 obvious slowed/reverb/nightcore/karaoke/cover/mashup variants
  with zero false negatives; six curated legitimate risky titles had zero false positives.
- **Release integrity.** Desktop and hosted downloads pin SHA-256; hosted download is atomic and
  fails before loading on a mismatch, and numpy object pickles are disabled.
- **399 automated tests** cover the recommenders, OAuth/PKCE, DSP, vibe and vibe-aware engines,
  the spec cache, recommendation benchmarks, diversity/MMR, GeM pooling, ML split logic, the
  categorized production benchmark, one-open protocol, audio experiments, real-index derivative
  false-positive/negative checks, checksum handling, and exact desktop/hosted parity.


---

## 9. What I'd build next

- **Persist a personal acoustic-feature store** so the engines cover a user's entire Spotify
  library, not just what's in a preview catalog.
- **Blind multi-reviewer listening panel** — add preview-level judgments beyond the sourced-pair
  benchmark and publish agreement, rather than letting one reviewer tune and test the same list.
- **A 512-d or a downloadable (non-bundled) index** — the downloadable index now exists (fetched from
  a GitHub Release past the 100 MB bundle cap), so library coverage can grow further; a wider encoder,
  though, was measured *not* to help (512-d matched 384-d). The next encoder gain should come from a
  better **objective** *selected on the right metric* — `cross_artist_agreement`, not same-artist mAP
  (§6 explains why the ArcFace mAP win didn't survive external validation).
- **Fix the niche weak spot** — external validation showed ultra-niche breakcore seeds (*Sewerslvt*)
  leak into trance. Now that `cross_artist_agreement` can score it against ListenBrainz/Deezer, it's a
  measurable target for the next fine-tune (e.g. harder negatives from a development-only graph).
- **Rotate a new unopened held-out split** after any training on the current development pairs, and
  improve catalogue coverage before claiming known-pair Recall@20 is solved.
- **Contrastive-on-vibe** — mine positive pairs by vibe similarity, not just augmented crops or
  same-artist labels, so the objective pulls same-*vibe* songs together directly (the natural next
  step after ArcFace, since the artist signal is a proxy for vibe, not vibe itself).

---

## 10. Skills demonstrated

For anyone evaluating this as a portfolio piece, the work spans:

- **Machine learning:** self-supervised contrastive learning (SimCLR/NT-Xent), **multi-task
  learning** (contrastive + auxiliary regression), CNN and ResNet encoders, mixed-precision
  training, embedding evaluation (kNN probe, silhouette, retrieval, **linear probing**), UMAP
  visualization — with an honest, measured account of when deep learning does and doesn't help.
- **Digital signal processing:** mel-spectrograms, MFCC/timbre features, frequency-band energy
  analysis, loudness dynamics, tempo and spectral analysis from raw audio.
- **GPU / systems performance:** diagnosing data-loading bottlenecks, VRAM-aware data residency,
  CUDA memory-layout and precision tuning, reading cuDNN kernel selection.
- **API integration & security:** OAuth 2.0 PKCE, token lifecycle management, rate-limit handling,
  secret hygiene.
- **Software engineering:** clean package design, a 285-test suite, packaging, a documented CLI,
  decoupling I/O from compute (the harvest-once spec cache), and reviewed, merged pull requests.
  Includes a reproducible human-aligned evaluation suite, three ranking improvements (quality
  filter, genre reranker, collaborative graph), and desktop/hosted parity tests.
- **Data engineering:** multi-connection downloading, parallel preprocessing across CPU cores,
  compact on-disk formats (float16 caches + models), robust handling of corrupt inputs.

Every result in this document was measured on real hardware and is reproducible from the commands
in the [README](../README.md).
