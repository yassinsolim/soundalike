# Any-song evaluation

How well `/anysong` finds related songs, how it was measured, and what changed between index versions. The
commands are in [ANYSONG.md](ANYSONG.md#building-an-index); the reports are JSON from `eval_index`,
`serve_check`, `robustness` and the known-pairs check.

## Benchmarks

| Benchmark | What a hit means | Source | Size | Split used here |
| --- | --- | --- | --- | --- |
| `genalpha` | another song built on the same melody | [The Gen Alpha Melody](https://www.youtube.com/watch?v=DW0XUsyBBuY) companion playlist (Carl E. Martin) | 65 songs, 1 group | test |
| `mb_covers` | another recording of the same work | MusicBrainz works (CC0) | 849 groups | development / held_out |
| `mb_samples` | the recording it samples | MusicBrainz "samples material" (CC0) | 1,610 pairs | development / held_out |
| `mb_samples_rev` | a recording that samples it (reverse direction) | same pairs, queried from the source | | development / held_out |
| `sampleid` | the recording it samples | [sony/sampleid](https://github.com/sony/sampleid) sample pairs (MIT) | 103 pairs | held_out |
| `v4` | a song people say it sounds like | editorial, critic and listener comparisons (Sounds Just Like, Pitchfork, Wikipedia, WatchMojo, …), each with its sources and an evidence category | 93 pairs | development / validation / held_out |

`v4` is the "what people think songs sound like" set. Its categories differ in nature: `pure_sonic` (54) is
production, timbre or groove resemblance with no shared material; `legal_plagiarism`, `sample_interpolation`
and `cover_remix` share material; `weak_unsubstantiated` (16) are claims without solid support, kept to see
whether the model agrees with them, not as ground truth.

Settings (hub penalty, candidate depth, projection size) are chosen on `development` only. `held_out`,
`validation` and the Gen Alpha `test` group are for reporting. One exception is disclosed under
[Hub penalty](#hub-penalty).

## Metrics

Evaluation uses the API's **song view**: releases with the same normalised artist and title (bracketed suffixes,
`feat.` and dash suffixes removed) count as one song, and the query's own song is excluded. That is what the page
shows, so a remaster of the query does not count as a hit.

- **AP**: mean average precision over all positive songs.
- **HR@k**: share of queries whose first correct song is in the top k.
- **Any tab**: the best rank of the first correct song over the three channels. Each channel is a separate tab on
  the page, so this is the chance that at least one tab shows a correct song in its top k.
- **Best list**: the fused *Best matches* list exactly as the API builds it from each channel's served rows. A
  correct song outside every fused list counts as ranked after all indexed songs.
- **Best, same language**: when evaluated with `--languages`, confidently different-language candidates are
  removed from each channel before fusion, as in the API. Unknown candidates remain; unknown seed languages fail
  open, with the same conservative artist-majority fallback as serving. The report includes the resulting Best
  metrics and the mean number of other-language songs in the unfiltered top 20 for labeled seeds.
- **Exact check**: the two-stage index (PCA candidates, then full-dimensional rerank) against brute force over
  every window, on random and benchmark queries. It confirms the fast path loses nothing.
- **Hubness**: for random catalog queries, how many distinct songs appear in anyone's top 10 (coverage), the
  skew of that count, and the share of all top-10 slots taken by the 100 most frequent songs. A "hub" is a
  track that shows up for everything.
- **Robustness**: catalog previews are edited (sped up 1.25×, slowed 0.85×, ±2 semitones, 10% faster, noise at
  10 dB SNR, phone band, a 10 s excerpt) and embedded live. A hit means the original song still ranks first.
- **Known pairs**: 24 famous relationships checked against the live API, including songs that are not in the
  catalog and are embedded on the fly. These are demonstrations, not an unbiased benchmark.

## Hub penalty

Some tracks sit close to everything and crowd real matches out of every list. `hub_build` scores each track's
"hubness" `h`, its mean similarity to its 10 nearest other songs. Serving subtracts `0.5 h`, which is CSLS
(Conneau et al., 2018).

On the 25k research index, applying it to every channel improved most development splits:

| development split (song AP) | melody | samples | sonic |
| --- | --- | --- | --- |
| mb_covers | 0.604 → 0.623 | 0.239 → 0.253 | 0.095 → 0.102 |
| mb_samples | 0.060 → 0.075 | 0.127 → 0.145 | |
| mb_samples_rev | 0.078 → 0.075 | 0.143 → 0.147 | |
| v4 pure_sonic (n=6) | | | 0.093 → 0.210 |

It also improved hubness: melody top-10 coverage rose from 0.299 to 0.333, and the top-100 share fell from
0.051 to 0.036. **But on Gen Alpha (test), the melody channel got worse:** HR@1 fell from 0.373 to 0.254 and
HR@10 from 0.576 to 0.542.

That test result is what made us look closer, which we disclose here. On development data, we split melody
queries by the hub percentile of their correct songs:

| positives' mean hub percentile | n | AP α=0 | AP α=0.25 | AP α=0.5 |
| --- | --- | --- | --- | --- |
| below p50 | 1,429 | 0.346 | 0.360 | 0.368 |
| p50–p90 | 1,001 | 0.242 | 0.244 | 0.243 |
| p90–p99 | 196 | 0.104 | 0.098 | 0.094 |
| p99 and above | 17 | 0.010 | 0.004 | 0.002 |

The penalty helps correct songs in sparse regions and hurts correct songs that are themselves in dense
regions. A melody family like Gen Alpha is a dense **genuine** cluster: its members' nearest neighbours are each
other, so their `h` is high (median member at p58, 12% at p90 or above), and the penalty pushes them down. The
development splits cannot show this directly, because no development query has more than three correct songs.
`mb_samples_rev` (development) also fell under melody's penalty.

The shipped defaults are therefore `alpha = 0` for **melody** and `0.5` for **samples** and **sonic**. Melody's
`hub.npy` is still built, so the choice can be revisited with `--hub-alpha`. The 271k results below check both
settings.

### Radio-show segments

The largest hub left after the penalty was not a single track. The catalog holds 2,031 segments of the radio
show *A State Of Trance*: track recaps, "Coming Up", intros, shout-outs, interviews. They are all titled "A State
Of Trance (…)", so the API's song key merges them into one song. Each segment has an ordinary `h`, so the
penalty cannot single them out. But a song scores as its best track, and with that many tracks one of them is
usually close to any query. On the 25k index (159 segments), this one "song" was in the top 10 of 1.2%, 2.1% and
1.1% of random queries (melody, samples, sonic). The penalty raised that to 1.7%, 2.3% and 1.7%.

These are spoken-word segments, not songs, so the full index leaves them out. `index_build --max-song-tracks
100` drops every song key with more than 100 catalog tracks. In this catalog that removes exactly this family:
the next-largest song has 25 tracks, and no benchmark track is affected. Each channel's `meta.json` records
what was dropped.

## Full catalog (269,278 tracks)

The numbers above came from a 24,958-track research index: every benchmark track plus about 20,000 random
catalog tracks as decoys. The served index is the whole frozen catalog, 271,309 tracks minus the 2,031 radio
segments. Tracks present in both indexes have identical stored vectors (cosine ≥ 0.9999 per window). Every
change below therefore comes from roughly 11 times more decoys competing for each slot, plus the hub policy.

The 25k column has no hub penalty; the 271k column uses the shipped policy. Benchmarks are reported on
`held_out`, `validation` and `test` only.

**Any tab** (does at least one tab show a correct song?):

| benchmark | n | HR@1 25k → 271k | HR@10 25k → 271k | median rank of first hit |
| --- | --- | --- | --- | --- |
| Gen Alpha (test) | 59 | 0.441 → 0.237 | 0.746 → 0.508 | 2 → 10 |
| covers (held_out) | 1,170 | 0.673 → 0.585 | 0.772 → 0.709 | 1 → 1 |
| samples (held_out) | 747 | 0.138 → 0.096 | 0.214 → 0.162 | 478 → 2,859 |
| samples, reverse (held_out) | 695 | 0.158 → 0.094 | 0.233 → 0.170 | 287 → 1,990 |
| SampleID pairs (held_out) | 43 | 0.140 → 0.140 | 0.279 → 0.186 | 333 → 1,528 |
| v4 "sounds like" (held_out) | 19 | 0.158 → 0.053 | 0.263 → 0.158 | 158 → 1,080 |
| v4 "sounds like" (validation) | 35 | 0.143 → 0.029 | 0.229 → 0.171 | 253 → 1,631 |

For scale, a random ranking finds a single correct song in the top 10 about 0.004% of the time, and one of Gen
Alpha's 64 other members about 0.24% of the time.

**The channel built for each relation**, 271k:

| benchmark | channel | AP | HR@1 | HR@10 | P@10 |
| --- | --- | --- | --- | --- | --- |
| Gen Alpha (test) | melody | 0.034 | 0.220 | 0.458 | 0.132 |
| covers (held_out) | melody | 0.517 | 0.542 | 0.677 | 0.093 |
| samples (held_out) | samples | 0.099 | 0.082 | 0.127 | 0.013 |
| samples, reverse (held_out) | samples | 0.097 | 0.081 | 0.132 | 0.014 |
| SampleID pairs (held_out) | samples | 0.130 | 0.116 | 0.140 | 0.014 |
| v4 pure sonic (held_out) | sound | 0.046 | 0.000 | 0.105 | 0.011 |
| v4 pure sonic (held_out) | samples | 0.079 | 0.053 | 0.105 | 0.011 |

What this means in practice:

- **Covers and shared melodies work.** For more than half of cover queries the top song is another version of
  the same work. For a Gen Alpha song, the melody tab's first other Gen Alpha song is in the top 10 for 46% of
  members, and about 1.3 of the top 10 are Gen Alpha songs on average. On the research index that was 2.8: many
  other catalog songs are close to the same melodic shape.
- **Samples are found when the sampled part is in both previews.** Otherwise the match is ranked in the
  thousands. A 30-second preview often misses the sampled segment or the sampling section of the new song.
- **General "sounds like" claims are the weakest.** The sound channel ranks a person-picked comparison in its top
  10 for about 1 in 10 queries. On these small sets the samples channel does as well or better. Treat that tab
  as an exploration aid, not an answer.

### Hub penalty at full scale

Turning the penalty off (`--hub-alpha 0`) on the same index confirms the policy on the large held-out sets:

| held_out (song AP) | samples off → on | sound off → on |
| --- | --- | --- |
| covers (1,170) | 0.203 → 0.215 | 0.057 → 0.063 |
| samples (747) | 0.094 → 0.099 | 0.006 → 0.010 |
| samples, reverse (695) | 0.090 → 0.097 | 0.010 → 0.010 |
| SampleID pairs (43) | 0.122 → 0.130 | 0.000 → 0.001 |

The small v4 sets move both ways (v4/validation samples AP 0.088 → 0.066, n = 35). For melody, `alpha = 0.5`
repeats the 25k trade-off: covers improve (held_out AP 0.517 → 0.539, HR@1 0.542 → 0.559) and Gen Alpha gets
worse (HR@1 0.220 → 0.169, HR@10 0.458 → 0.424). Melody stays at 0. The covers gain is real, and the
alternative is one flag away.

### Exactness and hubness

The two-stage search returned the same top-10 songs as brute force for all 256 queries per channel (128
benchmark queries and 128 random catalog tracks). Hubness at 271k, top-10 songs over 2,000 random catalog
queries:

| channel | songs ever in a top 10 | skew | top-100 share | most frequent song |
| --- | --- | --- | --- | --- |
| melody | 7.4% | 4.02 | 1.5% | 5 of 2,000 queries |
| samples | 7.7% | 3.64 | 1.2% | 5 of 2,000 |
| sound | 7.7% | 3.62 | 1.2% | 4 of 2,000 |

With the penalty off, samples and sound skew is 3.97 and 3.79. On the 25k index the most frequent song (the radio
show) was in 1–2% of lists; here no song reaches 0.3%.

## Best matches

The page used to open on the melody tab, so a sampled song only appeared if the visitor thought to switch tabs.
It now opens on **Best matches**, one list fused from the melody and samples lists with reciprocal-rank fusion
([Cormack et al., 2009](https://dl.acm.org/doi/10.1145/1571941.1572114)). Each channel serves its top 200 rows, one
release per song. A song scores `sum(w / (k + rank))` over those lists, and the page shows which place it took in
each.

**Choosing the settings.** All choices were made on `development` splits: highest macro HR@10, with configurations
within 0.01 of the best (about one v4 query) counted as tied and ranked by macro AP. The grid covered k ∈ {1, 2, 3, 5,
10, 20, 60}, eleven melody/samples/sound weightings, and melody hub α ∈ {0, 0.25, 0.5}. The same setting won at every
α: **k = 5, melody 1, samples 1, sound 0**. Adding the sound list makes things worse: at k = 20, weight 0.5 lowered
development covers AP by 0.022 and HR@1 by 0.031 (both significant) and reverse-samples AP by 0.006, and its only
gain was one v4 query.

**Held-out results**, 269k index as served. A `*` marks the better value where a paired bootstrap (2,000 resamples)
puts the 95% interval of the difference away from zero.

| benchmark | n | Best matches HR@1 / HR@10 / AP | channel built for it, alone | melody tab (old default) HR@10 |
| --- | --- | --- | --- | --- |
| Gen Alpha (test) | 59 | 0.186 / 0.441 / 0.019 | melody 0.220 / 0.458 / 0.034\* | 0.458 |
| covers (held_out) | 1,170 | 0.549 / 0.673 / 0.513 | melody 0.542 / 0.677 / 0.517 | 0.677 |
| samples (held_out) | 747 | 0.064 / 0.145\* / 0.095 | samples 0.082 / 0.127 / 0.099 | 0.080 |
| samples, reverse (held_out) | 695 | 0.076 / 0.157\* / 0.101 | samples 0.081 / 0.132 / 0.097 | 0.091 |
| SampleID pairs (held_out) | 43 | 0.070 / 0.186 / 0.119 | samples 0.116 / 0.140 / 0.130 | 0.093 |
| v4 "sounds like" (held_out) | 19 | 0.000 / 0.105 / 0.039 | | 0.000 |
| v4 "sounds like" (validation) | 35 | 0.029 / 0.114 / 0.049 | | 0.057 |

One list now does each specialist's job. Covers match the melody tab on every measure, and sample pairs reach the
top 10 more often than on the samples tab itself, because a sample match that the melody channel also ranks
moves up. Against the old default view, the top 10 holds a correct song for almost twice as many sample queries
(0.145 vs 0.080, 0.157 vs 0.091) at no measurable cost on covers. Gen Alpha's first hit is about as early as on the
melody tab (HR@10 0.441 vs 0.458, not significant). Its AP falls because AP counts all 64 other members, and sample
matches take some of the slots that deeper members held. Checking all three tabs (the "any tab" table above) still
finds more: 0.709 for covers, 0.508 for Gen Alpha and 0.162 for samples.

Two settings were checked on held-out data after the choice:

- **k.** k = 20, the common default, has the same Gen Alpha and samples numbers, but its covers HR@1 is lower (0.528
  vs 0.549\*), for slightly higher covers HR@10 (0.680 vs 0.673\*). A small k lets one channel's confident first
  place stand.
- **Melody hub penalty.** Inside the fused list, α = 0.25 raises covers HR@10 by 0.013\* and lowers Gen Alpha HR@10
  from 0.441 to 0.390 (3 of 59 songs); α = 0.5 lowers it to 0.339\*. Melody stays at 0.

### Language gate impact

The full language table labels all 269,278 tracks in each served channel. On the full index, across 2,837
`held_out`, `test` and `validation` queries, 2,550 seeds had a known language (including the artist-majority
fallback). Of those, 1,971 had at least one confidently different-language song in the unfiltered Best top 20;
there were 8,213 such results total, or 3.22 per known-language seed. The other 287 seeds failed open.

The table compares the shipped Best ranking with the same ranking after filtering each channel before fusion.
Unknown candidates remain visible. These are point estimates, not paired-bootstrap significance tests; the small
Gen Alpha and v4 groups are especially directional.

| benchmark (split) | n | Best before: HR@1 / HR@10 / AP | Same-language: HR@1 / HR@10 / AP |
| --- | ---: | --- | --- |
| Gen Alpha (test) | 59 | 0.186 / 0.441 / 0.019 | 0.153 / 0.407 / 0.015 |
| covers (held_out) | 1,170 | 0.549 / 0.673 / 0.513 | 0.565 / 0.685 / 0.527 |
| samples (held_out) | 747 | 0.064 / 0.145 / 0.095 | 0.064 / 0.142 / 0.094 |
| samples, reverse (held_out) | 695 | 0.076 / 0.157 / 0.101 | 0.075 / 0.153 / 0.099 |
| SampleID pairs (held_out) | 43 | 0.070 / 0.186 / 0.119 | 0.093 / 0.186 / 0.130 |
| v4 "sounds like" (held_out) | 19 | 0.000 / 0.105 / 0.039 | 0.000 / 0.053 / 0.029 |
| v4 "sounds like" (validation) | 35 | 0.029 / 0.114 / 0.049 | 0.029 / 0.086 / 0.044 |

The gate removes many unrelated-language results without materially changing sample retrieval in these aggregate
metrics. Covers improve in this sample, while Gen Alpha and v4 sound-alike retrieval lose some cross-language
matches. Those losses are expected trade-offs, not proof that every hidden result is irrelevant; `lang=all` restores
the original ranking. Since 2026-11-05 the homelab serves the filtered list by default, together with the
[scene layer](#scene-layer-2026-11-04).

### What did not help: query expansion

Query expansion re-scores each candidate by mixing its similarity to the query with its average similarity to the
query's top n results (n ∈ {1, 2, 3, 5, 8}, mixing weight λ ∈ {0.5, 1, 2}, 20 variants per channel). On development
splits it added at most 0.005 to melody covers AP over the hub-penalised baseline, and every variant lowered HR@10 (by
0.003 to 0.040). Every samples variant lowered both AP and HR@10. For sound, AP rose by up to 0.026 with HR@10
unchanged, measured on only 31 queries. It is not used.

The tooling for these studies (re-ranking, fusion sweep and paired bootstrap) lives outside the package; `eval_index`
reports the shipped `best` list directly.

### Research follow-up: retrieval and reranking (2026-11-02)

**Decision: keep the existing Best ranking.** A bounded study tested 22 alternative configurations against the
full 269,278-track catalog. None cleared the development gate. These are negative research results, not a new
model release; the website, Spicetify release and production inference were not changed by this study. The
language gate above is a separate preference/filtering change, not evidence of better audio similarity.

#### Protocol and sources

- Capture top-800 candidates per channel for 2,634 distinct development query tracks. Candidate scores are
  full-dimensional reranks inside the existing two-stage search, not a new exhaustive scan of every catalog
  window. Preserve song deduplication, self-exclusion, hub penalties and deterministic tie ordering.
- Evaluate **all-language Best**. The four equally weighted families are covers (1,066 query records), samples
  (806), reverse samples (721), and v4 (25). Queries may occur in multiple families. Do not count the v4
  pure-sonic subset a second time in the macro.
- Fix the selection rule before examining results: no family may lose more than 1 percentage point of HR@10
  or AP; require at least 1 percentage point of macro HR@10 or AP gain. Rank qualifying candidates by macro
  HR@10, then AP. The same gate applies to all follow-up experiments.
- [Bruch et al., fusion functions](https://arxiv.org/abs/2210.11934) motivates score-aware fusion. Test min-max
  and distribution-based normalization, melody weights 0.25/0.5/0.75, complementary sample weights, and sonic
  weights 0/0.25: 12 configurations. Text-retrieval gains are not assumed to transfer to music.
- [Riou et al., SampleID](https://arxiv.org/abs/2510.11507) motivates mean top-k chunk matching and denser
  windows. Test top-2/top-5 pooling within captured candidates, plus 16 rather than 8 query windows
  (hop 24,500 rather than 49,000 samples). Keep the checkpoint and catalog windows unchanged. The paper's
  full-track experiments are not directly comparable with this preview-only, much larger catalog.
- [Artetxe and Schwenk, margin-based mining](https://arxiv.org/abs/1811.01136) motivates dividing sample cosine
  by mean query/candidate neighborhood density. This only reorders the existing sample top-800 and uses
  indexed query density; an unseen-query equivalent was not validated.
- Also test a histogram-gradient-boosted reranker using 24 audio/score/rank features, with no song ID, artist
  ID or benchmark identity as model features. Five-fold cross-validation keeps connected query/positive song
  groups together. Train on positives, hard negatives and sampled candidates, and evaluate out-of-fold
  predictions at four blend strengths. Queries lacking a captured positive remain in evaluation.

#### Development results

All values are point estimates, not significance-tested gains. The normalization row gives separate ranges
across its 12 configurations, not one model's paired result.

| configuration | macro HR@10 | macro AP |
| --- | --- | --- |
| existing Best baseline | 0.278790 | 0.190335 |
| RRF depth 800 instead of 200 | 0.278558 | 0.191148 |
| normalized-score fusion (12 variants) | 0.185575-0.276645 | 0.123998-0.186575 |
| sample top-2 pooling | 0.280375 | 0.191300 |
| sample top-5 pooling | 0.278978 | 0.190653 |
| learned ranking, 25% blend | 0.278868 | 0.195150 |
| learned ranking, 50% blend | 0.275755 | 0.198748 |
| learned ranking, 75% blend | 0.265858 | 0.199295 |
| learned ranking, 100% | 0.250650 | 0.197400 |
| sample density-ratio margin | 0.278790 | 0.190365 |
| 16 query windows, max pooling | 0.279560 | 0.191700 |
| 16 query windows, top-2 pooling | 0.280450 | 0.192120 |

Doubling query windows with top-2 pooling improves macro HR@10 by only **0.166 percentage points** and AP by
**0.179 percentage points**. Stronger learned reranking trades away hit rate for AP, without meeting the
minimum macro gain. No candidate was promoted, no final learned model was saved, and no new candidate
comparisons on reporting splits were run after these failures.

The candidate oracle measures whether a known positive is anywhere in the melody/sample candidate union,
regardless of its final rank:

| development family | depth 200 | depth 800 |
| --- | --- | --- |
| covers | 80.0% | 85.1% |
| samples | 25.4% | 32.6% |
| reverse samples | 28.0% | 38.3% |
| v4 | 40.0% | 56.0% |

There is ranking headroom, but even at depth 800 most sample queries have no known positive available for
reranking. That is a candidate-retrieval limitation, not proof that the positive is absent from the catalog.
Which misses are caused by preview coverage versus representation quality still needs a case-by-case audit.

#### Cost, integrity and next evidence

The dense-query run on the **RTX 5080** processed all 2,634 previews in 271.3 seconds. With batches of four,
median batch time divided by query count was 33.8 ms for embedding and 39.4 ms for sample search. Peak
PyTorch-allocated memory was 3,417 MiB with the sample model/index only. These are **batched throughput
measurements, not API latency or proof that the full three-model service fits unchanged on the 2070 SUPER**.
Before capture, four sparse replay queries matched stored windows at cosine >= 0.9999996 and had 100%
top-200 set overlap. All dense queries completed; decoder output contained only empty-ID3-tag warnings.

The research harness's two unit tests pass. Final checks confirm matching baseline metrics across all four
reports, aligned query IDs, 800 distinct non-self candidates per dense query, finite scores, and matching
recorded source/input/checkpoint hashes. Provenance includes index metadata and ID arrays, not fresh hashes
of every large full-dimensional index array.

Existing reporting sets are regression suites already inspected in earlier work, not untouched tests. A
song-key overlap audit found 1 shared key for covers, 53 for samples and reverse samples, and 0 for v4 between
development and reporting partitions. Any future trained candidate should also report a clean sensitivity
subset excluding queries whose query or positives intersect development keys. Confidence intervals should
resample connected song groups, not assume related query rows are independent. Gen Alpha is effectively one
such group. Unjudged candidates used as training negatives may contain legitimate relationships.

The stronger next experiments need better evidence rather than more fusion-weight tuning:

1. Audit missed known pairs to separate missing shared material in the previews from embedding failures.
   Test longer or multiple excerpts only where authorized audio is available.
2. Collect disjoint human judgments of melody, rhythm, voice and timbre resemblance, including hard negatives.
   Relationship labels such as covers and samples do not fully measure perceptual similarity.
3. [MATCHA](https://arxiv.org/abs/2609.00987) offers 300 triplets and 1,105 perceptual annotations from 83 experts.
   Its [datasheet](https://github.com/roserbatlleroca/matcha/blob/main/metadata/MATCHA_datasheet.md) restricts access
   to the exact commercial audio excerpts. No MATCHA score was computed: arbitrary Deezer previews are not
   interchangeable with its curated excerpts.

Reproducibility artifacts remain outside the package in the research workspace:
`fusion_research_20261102.py`, `learned_fusion_20261102.py`, `margin_research_20261102.py`, and
`dense_query_research_20261102.py`. The `fusion-research-20261102` directory retains frozen candidate captures,
`development.json`, `learned_development.json`, `margin_development.json`, `dense_development.json`, and
`research_summary.json`. The summary records all 22 gate decisions and hashes of the four reports; the
original `selected.json` still selects the baseline. No production rollout is implied by these local artifacts.

### Miss audit: catalog coverage versus preview/model scores

The next diagnostic compared each labeled positive to the entire indexed song identity set and then scored its
stored full-dimensional windows directly against the query's stored windows. This used 3,334 unique
query/positive-song pairs from the same four development families. It downloaded no audio and made no model or
index changes.

**Catalog coverage was not the main failure:** every one of the 3,334 labeled positive song identities had at
least one track in the frozen 269,278-track index. This does not establish that the specific MusicBrainz
recording or the relevant passage occurs in the 30-second preview.

For each channel, the table classifies each labeled query/positive pair by its best same-song full-window
score and compares that score with the captured top-800 candidate tail. Scores include the same hub penalty
as serving. `201–800` and `Top 200` are ranks after track-to-song de-duplication.

| channel | Top 200 | 201–800 | Below exact top-800 boundary | Above boundary but absent (possible shortlist miss) |
| --- | ---: | ---: | ---: | ---: |
| melody | 1,359 | 175 | 1,800 | 0 |
| samples | 948 | 270 | 2,114 | 2 |
| sonic | 451 | 249 | 2,634 | 0 |

Across the melody/sample channels used by Best, **1,478 of 3,334** positive pairs were outside both top-800
candidate lists: 362 covers, 573 sample relations, 532 reverse-sample relations and 11 v4 sound-alike pairs.
Another 61 pairs appeared only in the sonic list, which Best currently does not use. Most misses therefore
already have low similarity under the available preview embeddings; only two sample pairs look like possible
two-stage shortlist misses. Direct stored-vector scoring is not an exhaustive independent re-embedding, but
its candidate-score parity check differed by less than 0.0002 for every positive pair present in the captures.

This narrows the failure to **preview/model representation**, rather than missing catalog rows or a broad
approximate-search bug. It cannot tell whether a sample or resemblance falls outside the available excerpt,
whether the preview is a poor recording, or whether the embedding fails to recognize audible material. Those
cases require authorized longer audio or human listening.

The denser 16-window query experiment barely changed candidate coverage. Query-level share with any known
positive in the melody/sample candidates at track depth 200 was:

| family | 8-window baseline | 16-window + top-2 pooling |
| --- | ---: | ---: |
| covers | 80.0% | 80.4% |
| samples | 25.4% | 26.3% |
| reverse samples | 28.0% | 28.6% |
| v4 | 40.0% | 40.0% |

At depth 800, the corresponding changes were 85.1→85.1%, 32.6→33.4%, 38.3→38.4%, and 56.0→56.0%. These are
development point estimates with no significance claim and do not clear the previously fixed gain threshold.

The user declined further blind preview rating, so the generated pilot worksheet and its answer key were
discarded before any ratings were collected. Follow-up work should focus on a valid full-audio data source and
model evaluation, not more manual preview comparisons.

### Full-length audio source feasibility (2026-11-02)

**Spotify/Spicetify is not a supported path for model audio.** Spotify's
[Web Playback SDK](https://developer.spotify.com/documentation/web-playback-sdk) streams and controls a local
Spotify Connect player and exposes playback metadata; its documented
[player reference](https://developer.spotify.com/documentation/web-playback-sdk/reference) has no waveform or
PCM access. More importantly, the current [Spotify Developer Policy](https://developer.spotify.com/policy)
prohibits analyzing Spotify Content, using it to train machine-learning models, and building products
integrated with streams/content from another service (sections III.5, III.13 and III.14). Do not capture the
Spotify stream, hook private Spicetify internals to extract it, or send it to Soundalike. Spicetify can’t turn
Premium playback into an authorized training corpus.

**Jamendo has full-track streams, but its API is not a corpus-download license.** Its
[track API](https://developer.jamendo.com/v3.0/tracks) returns stream/download URLs and per-track
`audiodownload_allowed`; tracks have individual Creative Commons licenses. The published
[API terms](https://devportal.jamendo.com/api_terms_of_use) limit API use to non-commercial purposes unless
licensed otherwise and prohibit applications designed to cache content or provide offline access. A persistent
local index or training archive is therefore not a safe use of the API as-is. The terms page is dated 2013;
confirm current terms and obtain explicit permission before building any corpus.

**There are research datasets with full-length, openly licensed audio, but they are not the missing mainstream
recordings or sound-alike labels:**

- [MTG-Jamendo](https://github.com/MTG/mtg-jamendo-dataset) documents over 55,000 full tracks under Creative
  Commons licenses. Its raw audio archive is listed at 508 GB; the mood/theme subset is 152 GB. Labels are
  primarily uploader-provided genre, instrument and mood/theme tags—not verified cover/sample/sound-alike
  pairs. The exact license still needs to be respected per track and for any derivative model.
- [FMA](https://github.com/mdeff/fma) documents 106,574 Creative-Commons-licensed tracks, including a
  full-length archive listed at 879 GB. Its metadata and tags support music-analysis research, not a
  relationship-labeled sound-alike benchmark.

Those archives could support an independently scoped research model, but likely will not improve retrieval
for the commercial songs already in this catalog without a relevant training objective and evaluation set.
No archive has been downloaded. Do not start a 150–880 GB download just because the disk has capacity.

The practical route for testing longer excerpts of the target songs is a separate, local-file workflow using
audio the user is authorized to analyze (for example, purchased files or their own recordings), with explicit
rights checks and no Spotify stream extraction. That can distinguish excerpt coverage from representation
quality for those particular songs. It does not create blanket rights to train on or redistribute them. A
Spotify UI integration should not be used to pass Spotify audio into this workflow.

## Scene layer (2026-11-04)

**Why.** For *8PM* by Nemzzz (UK rap), the Best top 20 held three songs by UK-scene artists: Skiifall at 9, DigDat
at 16 and another Nemzzz song at 20. The rest was US rap. Three causes were measured:

- **Unbalanced catalog.** YoungBoy Never Broke Again has 608 served tracks, Chief Keef 615, Future 448 and
  Lil Durk 343. Nemzzz has 28, Central Cee 79 and Headie One 37. Audio similarity that ignores region fills
  slots by base rate.
- **Best looks for shared material, not shared scene.** On 800 catalog seeds, only 9.8 % of its top-10 songs
  are in the seed's sounds top 200 at all.
- **The relationship benchmarks pull the other way.**
  - Of the cover positives that Best already places in the top 10, 54 % are by artists with no scene
    connection to the query artist (33–34 % for samples).
  - Sample positives have lower scene affinity than the songs competing with them.

So likely covers and samples have to be protected using audio evidence; metadata cannot tell them apart. An
earlier attempt trained the encoder to separate artists, and it hurt cross-artist recommendations
([CASE_STUDY §6](CASE_STUDY.md#external-validation-and-rollback)). This layer only reorders the audio candidates.

**Sources.**

| source | use | coverage | terms |
| --- | --- | --- | --- |
| Deezer `/track/{id}` | primary artist of each catalog track | all 269,278 served tracks | personal, non-commercial ([terms](https://developers.deezer.com/termsofuse)) |
| Deezer `/artist/{id}/related` | related-artist graph, up to 20 per artist | 20,054 catalog artists; other artists looked up live | as above |
| Wikidata, via P2722 (Deezer artist id) | citizenship, origin, formation and birth countries; genres | region (stated or inferred) for 46,301 of 65,844 artists, genres for 41,915 | CC0 |
| ListenBrainz similar artists | validation only, never a feature | 491 of 799 test seeds | Creative Commons dumps |
| Spotify | not used | | its [Developer Policy](https://developer.spotify.com/policy) forbids analysing content or training models on it |

**Two findings shaped the features.**

- **Citizenship is not scene.** 4,175 artists have one stated country and a clear neighbourhood majority. For
  13.7 % of them, the stated country differs from their neighbours' (7.6 % of rap artists). 21 Savage is
  British by birth, but 94 % of his graph neighbours are American. Region is therefore 70 % the neighbours'
  stated countries and 30 % the artist's own.
- **Audio does not reveal region.** MuQ-MuLan zero-shot text prompts separate UK from US rap barely better than
  chance: ROC AUC 0.62 per track, 0.68 per artist and 0.49 for a "UK drill" prompt. No prompt-based tags are used.

**What failed.** The table below uses development data: frozen top-800 captures of 2,634 relationship queries.
Changes are in percentage points. The first four rows are research simulations.

| design | covers HR@1 / HR@10 / AP | verdict |
| --- | --- | --- |
| affinity multiplies the audio score (plus a sounds term), no protection | −7.79 / −0.47 / −4.78 | fails |
| scene as a third RRF list (weight 0.5) | −5.44 / +0.28 / −2.63 | fails |
| additive scene bonus | −1.97 / −0.19 / −0.84 | fails on HR@1 and AP |
| fixed three-song head, then scene slots | 0 / +0.38 / −0.14 | fails: samples HR@10 −0.87, v4 −1 query |
| multiplicative boost with standout protection, winner of the preregistered sweep | passed | rejected at confirmation: same-language held-out covers HR@10 −0.77 and AP −0.54, Gen Alpha −2 queries |
| the same, re-screened in both language modes | passed | rejected at confirmation: same-language held-out covers AP −0.56 [−1.06, −0.10] |

**Shipped design.** It follows the research report's recommendation to treat the page as two intents,
relationship search and scene browsing. Each intent gets reserved slots, instead of everything being blended into
one score, much as intent-aware diversification does
([Agrawal et al., 2009](https://doi.org/10.1145/1498759.1498766);
[Santos et al., 2010](https://doi.org/10.1145/1772690.1772780)). Two findings from the literature support the
inputs: relational artist data beats audio for finding artist neighbourhoods
([Korzeniowski et al., 2021](https://arxiv.org/abs/2107.14541)), and recommenders can amplify the dominance of US
music ([Lesota et al., 2022](https://archives.ismir.net/ismir2022/paper/000034.pdf)).

1. **Affinity** of a candidate to the seed's artist:
   - 0.6 for the same artist;
   - otherwise min(1, 4 × graph + 0.3 × region).

   Here, graph is the overlap of the two artists' closed neighbourhoods in the symmetrised related-artist graph,
   |N(a) ∩ N(b)| / √(|N(a)| |N(b)|). Region is the probability that both artists belong to the same country's
   scene. Genre is stored but weighted 0.

   The research prototype used personalised PageRank instead. In a comparison on development seeds, PageRank and
   the overlap passed the same gates; PageRank had slightly higher ListenBrainz agreement (0.239 vs 0.224) and
   slightly lower agreement beyond Deezer's list (0.141 vs 0.146). The overlap needs no graph walk, so a related list fetched
   live for an unseen artist can be used directly.
2. **Head.** The first three songs of the audio ranking stay. The head extends to the last top-10 song that is
   among the first three songs of the melody or samples list, or whose melody score is at least −1.476 or samples
   score at least 0.302.
   - On development labels, these score thresholds have 30 % precision for a labelled cover or sample (recall
     50 % and 21 %).
   - The list-rank rule was added after release (see Selection). Since the fused top six usually interleaves the
     two lists' top three, the head is six songs or more for 88 % of development relationship queries and 97 % of
     scene seeds; for the UK-rap seeds its median is nine.
3. **Continuation.** The remaining songs are ordered by audio score × (1 + 2 × affinity). A song whose melody or
   samples score stands 3 or more standard deviations above its list's mean is lifted as if its affinity were 1,
   so no scene neighbour can pass it.
4. **Scene lane.** Songs with affinity above 0, ordered by affinity × (audio + sounds RRF score).
5. **Slots.** After the head, slots repeat continuation, continuation, lane (A-A-S). Each artist is capped at two
   songs in the top 20.

**Selection.** A preregistration fixed the following before any sweep:

- **Development gates.** Covers, samples and reverse samples may lose at most 0.5 pp of HR@10 and of AP, and
  1 pp of HR@1. v4 may lose at most one query.
- **Objective.** ListenBrainz agreement on 800 artist-disjoint development seeds (one per artist), with guards on
  same-artist crowding and artist diversity.
- **Confirmation.** Run once on 799 test seeds and the reporting splits; Gen Alpha may lose at most one query.

These deviations are disclosed:

- Standout protection and region inferred from neighbours were added after diagnostics on partly collected data.
- The held-out and test splits were consulted for three configurations: the two rejected above and this one.
- The slotted design comes from the research report, which used development data only.
- The final values were chosen from a 12-variant development grid by a rule fixed before the run: pass both
  language-mode gates and the guards, then maximise development ListenBrainz agreement. The grid was graph 1, 2
  or 4 × region 0.3 or 0.6 × same-artist 0.6 or 1.0. All 12 passed, and neighbouring settings differ by less
  than the seed-bootstrap noise (about ±0.013).
- A third sweep over standout windows (65 variants) was superseded by the slotted design before any confirmation.
- Confirmation ran twice with the same parameters. The first run used incomplete metadata and lost one Gen Alpha
  query with all languages and four in same-language mode. The second used the complete table, and that version
  was released.
- **Adjusted after release.** A live check of 24 hand-picked famous pairs (see [Known pairs](#known-pairs)) found
  two that the release pushed out of Best's top 10 while their own tabs still ranked them high: Cupid → its Gen
  Alpha match fell from #5 to #11 (melody #3), and Got To Give It Up → Blurred Lines from #10 to #20 (samples
  #5). K-pop and Motown neighbours had passed them, because their scores sat just below the evidence thresholds.
  On development data, also counting a song's place in its list as evidence traded scene gain for fewer
  displaced relationship hits:

  | head evidence | covers top-10 hits lost / gained | samples | reverse samples | ListenBrainz agreement | 8PM UK-scene songs in top 20 |
  | --- | --- | --- | --- | --- | --- |
  | scores only (first release) | 6 / 14 | 2 / 1 | 3 / 3 | 0.205 | 12 |
  | scores or a list's top 3 (shipped) | 5 / 11 | 2 / 0 | 2 / 3 | 0.188 | 12 |
  | scores or a list's top 5 | 2 / 0 | 1 / 0 | 1 / 0 | 0.131 | 11 |

  Baseline ListenBrainz agreement is 0.127. The project owner chose the top 3, which brings Cupid's match back but
  not Blurred Lines. The held-out and test splits were then evaluated once more, a fourth look. The first release
  measured, with all languages: covers HR@10 +0.94 pp, samples −0.13, reverse samples −0.43; test ListenBrainz
  agreement +0.085; and two lost Gen Alpha queries in same-language mode.

**Relationship benchmarks.** Shipped setting (top-3 rule), full index, official `eval_index --scene`; the research
harness agrees exactly. HR@1 is unchanged in every family. At release the served mode was all languages; since the
language filter went on (2026-11-05), the same-language columns describe what is served.

| benchmark | n | Best: HR@10 / AP | with scene layer | same-language Best | same-language with scene |
| --- | ---: | --- | --- | --- | --- |
| Gen Alpha (test) | 59 | 0.441 / 0.019 | 0.458 / 0.018 | 0.407 / 0.015 | 0.390 / 0.014 |
| covers (held_out) | 1,170 | 0.673 / 0.513 | 0.681 / 0.513 | 0.685 / 0.527 | 0.688 / 0.527 |
| samples (held_out) | 747 | 0.145 / 0.095 | 0.142 / 0.094 | 0.142 / 0.094 | 0.139 / 0.094 |
| samples, reverse (held_out) | 695 | 0.157 / 0.101 | 0.154 / 0.100 | 0.153 / 0.099 | 0.151 / 0.099 |
| SampleID pairs (held_out) | 43 | 0.186 / 0.119 | 0.186 / 0.118 | 0.186 / 0.130 | 0.186 / 0.130 |
| v4 "sounds like" (held_out) | 19 | 0.105 / 0.039 | 0.158 / 0.048 | 0.053 / 0.029 | 0.105 / 0.038 |
| v4 "sounds like" (validation) | 35 | 0.114 / 0.049 | 0.114 / 0.054 | 0.086 / 0.044 | 0.114 / 0.048 |

A paired cluster bootstrap over connected query/positive groups (2,000 resamples) gives these changes with all
languages:

| family | HR@10 change, pp | AP change, pp |
| --- | --- | --- |
| covers | +0.85 [−0.08, +1.88] | +0.06 [−0.04, +0.19] |
| samples | −0.27 [−0.94, +0.40] | −0.05 [−0.12, +0.01] |
| reverse samples | −0.29 [−0.72, 0.00] | −0.03 [−0.08, +0.01] |
| SampleID pairs | 0 | −0.02 [−0.07, +0.02] |

Every interval includes zero, reverse-samples HR@10 only just. On development splits, HR@10 changed by +0.56 pp
for covers, −0.25 for samples, +0.14 for reverse samples and 0 for v4. In same-language mode, the development
changes were +0.47, −0.37 and −0.28, and Gen Alpha lost one query, within the limit.

**Scene quality.** Top 10 of Best on artist-disjoint seeds (one per artist), with a paired seed bootstrap of 2,000
resamples. ListenBrainz rows cover only seeds whose artist has ListenBrainz neighbours: 470 development, 491 test
and 53 UK-rap seeds.

| measure | development (800) | test (799) | UK rap (75) |
| --- | --- | --- | --- |
| other-artist results that ListenBrainz lists as similar | 0.127 → 0.188 | 0.123 → 0.192 (+0.069 [0.062, 0.076]) | 0.122 → 0.150 (+0.028 [0.009, 0.048]) |
| the same, excluding artists on Deezer's related list | 0.098 → 0.135 | 0.093 → 0.134 (+0.041 [0.034, 0.048]) | 0.087 → 0.114 (+0.027 [0.010, 0.047]) |
| other-artist results from the seed's scene country | 0.355 → 0.416 | 0.359 → 0.428 (+0.070 [0.063, 0.076]) | 0.231 → 0.292 (+0.061 [0.040, 0.084]) |
| seeds with another song by the same artist | 0.261 → 0.349 | 0.254 → 0.317 (+0.063 [0.044, 0.080]) | 0.173 → 0.227 |
| seeds with four or more songs by the same artist | 0.013 → 0 | 0.015 → 0 | 0.027 → 0 |
| distinct artists | 9.56 → 9.56 | 9.57 → 9.54 | 9.44 → 9.49 |
| songs also in the sounds top 200 | 0.098 → 0.161 | 0.105 → 0.168 (+0.064 [0.058, 0.069]) | 0.076 → 0.111 |

Read these rows with three caveats:

- **Circularity.** ListenBrainz and Deezer's related lists both come from listening data, and they overlap
  heavily: a third of Deezer's 20 related artists are in ListenBrainz's top 20, against 0.1 % for random
  artists. The first row therefore partly re-measures the feature; the second row excludes that overlap.
- **Sounds-list share.** The gain in the last row is partly by construction, since the lane orders songs by
  audio plus sounds score.
- **UK-rap stratum.** It was chosen by name and Wikidata labels, and only about half of it is UK rap in the
  strict sense. Its top 10 changes less than its top 20: rap seeds have many strong melody and sample matches,
  and those lengthen the head.

**8PM.** Frozen capture of the shipped setting, all languages; scene country is inferred from the graph. The live API
returned the same list before the language filter went on.

- **More UK scene.** UK-scene artists in the top 20 went from 3 to 12, and other Nemzzz songs from one (#20) to
  two (#13, #14).
- **New entries** include Kidwild (#9), AntsLive (#10), Headie One (#11), Central Cee (#12 and #15), Krillz (#17)
  and Blanco (#20).
- **Dropped out:** songs by Eminem, Snoop Dogg, JAY-Z, Metro Boomin, Tee Grizzley and the French rapper Myth
  Syzer, among others.
- **The top six are still US songs.** Each is among the first three of the melody or samples list, so they form
  the protected head (DJ Khaled, Spiff TV, Lil Durk, 21 Savage, 42 Dugg, Future). Lil Man J follows at #7 because
  he stands out in the samples list (z ≈ 3). In rap, such strong sample scores are often shared drum sounds rather
  than samples.

**Latency.** On the local RTX 5080, a catalog seed answered in about 97 ms with the layer on. On the RTX 2070 SUPER
after release, six spot checks answered in 0.17–0.37 s for catalog songs and 0.73 s for an unseen song, within the
run-to-run range seen before release; GPU memory stayed at 6.2 GB.

- An artist whose related list is not in the table costs one Deezer request: 365–560 ms when measured blocking.
- That request now runs in the background while the preview downloads. For two artists not in the table
  (Bugzy Malone, Pa Salieu), the ranking waited 0 ms for it.
- The table is 5.1 MB.

**Limitations.**

- **Gen Alpha with the language gate.** In same-language mode, the layer loses one of 59 Gen Alpha queries
  (0.407 → 0.390), within the limit; with all languages it gains one. Since 2026-11-05 the served list is
  same-language, so Gen Alpha's served HR@10 is 0.390, four queries below the all-language list with the layer
  (0.458); most of that gap is the language gate itself. **Show all languages** restores the all-language list.
- **Famous cross-scene pairs can still drop.** With the top-3 rule, Got To Give It Up → Blurred Lines falls from
  #10 to #19 in Best (#5 in the samples tab).
- **Flow is not measured.** The graph reflects it only through listening habits.
- **Snapshot.** The table should be rebuilt about monthly, because related lists drift and new artists appear.
  Unseen artists are covered by the live lookup.
- **Licensing.** Deezer's terms limit use to personal, non-commercial applications. In research, a CC0-only
  variant (Wikidata country and genre, no Deezer graph) matched the graph variant on agreement beyond Deezer's
  list. It did nothing for 8PM, because Nemzzz and most UK drill artists have no Wikidata item.

## Sounds like v2 and query stems (2026-11-05): not served

A follow-up study asked what would most improve the results people see. Every experiment used development data; the
two strongest candidates then got one confirmation each under rules written before the run. **Both failed and are
not served.** The research workspace (`improve-research-20261104`) holds the scripts and results; the Sounds like v2
code is on the `exp/sounds-v2` branch.

**Where the gap is.** Top 10 for 800 artist-disjoint development seeds (one per artist), other-artist results only:

| list | ListenBrainz lists the artist as similar | the same, beyond Deezer's related list | shares a Wikidata genre | confidently another language |
| --- | ---: | ---: | ---: | ---: |
| melody tab | 0.103 | 0.083 | 0.219 | 0.201 |
| samples tab | 0.147 | 0.114 | 0.298 | 0.173 |
| sounds tab (MuQ-MuLan) | 0.211 | 0.162 | 0.385 | 0.150 |
| Best | 0.127 | 0.098 | 0.265 | 0.179 |
| Best with the scene layer | 0.205 | 0.147 | 0.340 | 0.147 |
| random songs | 0.012 | 0.010 | 0.063 | 0.203 |

For a typical song the catalog holds no cover or sample, so 80 % of Best's top-10 slots carry no calibrated
cover/sample evidence, and the sounds tab agrees better with listeners and genres. Audio evidence cannot pick the
intent song by song: the evidence thresholds fire for 52 % of random seeds against 59 % of cover queries, gating Best
on them cost covers HR@10 19–37 pp on development data, and sounds slots inside Best cost 0.75–4 pp. The two intents
stay in separate lists.

**Sounds like v2.** The candidate list works like this:

- MuQ-MuLan windows pass through a small residual network (512 → 1,024 → 512). It was trained for about 30 s with a
  contrastive loss on pairs of songs by Deezer-related artists.
- Training excluded every scene seed's artist and every benchmark track. The model was chosen on 600 held-out
  training artists, and ListenBrainz was never a training signal.
- The top 200 songs by cosine MaxSim are re-ranked by z(score) + 1 × scene affinity.
- Other-language songs are dropped and each artist keeps one song.

On development seeds it raised ListenBrainz agreement from 0.191 (sounds tab) to 0.315, agreement beyond Deezer's
list from 0.147 to 0.245, and genre agreement from 0.367 to 0.467, with other-language results at 0 %. A network
trained only on Wikidata genres also raised agreement beyond Deezer's list (+0.025), so the gain is not just a copy
of Deezer's graph. On 446 MagnaTagATune odd-one-out triplets (human judgments, other audio) it matched raw MuQ-MuLan
(0.744 vs 0.752). The production implementation reproduced the research list (ListenBrainz 0.3147 vs 0.3146).

One-time confirmation on the 799 test seeds, against Spicetify's current Sounds like list:

| pre-registered check | result | verdict |
| --- | --- | --- |
| agreement beyond Deezer's list ≥ +0.05, CI above 0 | +0.114 [0.095, 0.133] | pass |
| genre agreement ≥ +0.05, CI above 0 | +0.147 [0.122, 0.173] | pass |
| other-language results ≤ 2 % | 0 % | pass |
| distinct artists ≥ 9.5 | 9.99 | pass |
| MagnaTagATune at most 2 pp below raw MuQ-MuLan | −0.8 pp | pass |
| catalog coverage at most 0.5 pp below raw MuQ-MuLan | 7.59 % vs 7.56 % | pass |
| v4 pure-sonic targets in the top 50, at most one fewer than the sounds tab | 1 of 44 vs 7 of 44 | **fail** |

With the language filter applied to both lists, v2 still beat the website's sounds tab: ListenBrainz +0.091
[0.076, 0.106], genre +0.081 [0.061, 0.101]. But the pairs that critics and listeners cite as sounding alike often
cross scenes and eras, and the projection loses them before any re-rank (7 → 2 of the 44 in the top 50). It also
did not help the case that started this work: 8PM's projected top 200 holds only 15 UK-scene songs, and v2's top 20
has two other-artist UK-scene songs, against four for the sounds tab and seven for Spicetify's list. v2 is not
served, and the test seeds and v4 pairs are now spent for sounds-like lists.

**Query stems in the samples channel.** Hybrid Demucs (`HDEMUCS_HIGH_MUSDB_PLUS`) separates the query preview. Its
instrumental stem (drums, bass and other) is embedded with SampleID as served and searched against the unchanged
samples index. The samples list becomes the union of the mixture and instrumental lists, ordered by the larger
per-query z-score, and Best and the scene layer read that list. Vocal stems hurt covers and were dropped.

- **Samples channel alone** (260 development queries, full catalog): forward-sample HR@10 0.115 → 0.150
  (+3.5 pp [0.8, 6.2]).
- **Inside Best with the scene layer** (development): it passed the strict gate in both language modes. Samples
  HR@10 rose by +1.61 pp [0.50, 2.80] with all languages and +1.36 pp [0.24, 2.61] in same-language mode.

The one-time held-out confirmation had to pass in both modes:

| mode | samples HR@10 / AP, pp | covers HR@10 / AP, pp | Gen Alpha hits in the top 10 (59) | verdict |
| --- | --- | --- | --- | --- |
| all languages | +1.34 [0.26, 2.45] / +0.60 [0.17, 1.06] | −0.17 / −0.18 | 27 → 26 | pass |
| same language | +1.34 [0.27, 2.46] / +0.68 [0.22, 1.15] | −0.09 / −0.49 [−0.85, −0.13] | 23 → 21 | **fail** |

The served list is same-language, so stems are not served. Serving them would mean precomputing instrumental
windows for the catalog (about 15 GPU-hours on the RTX 5080, 8.8 GB) and separating unseen songs live, which needed
about 1.2–1.4 GB of extra VRAM on the 5080; that is not measured on the 2070.

**Other findings (development).**

- **Encoders.** None beats MuQ-MuLan on the MagnaTagATune triplets: LAION-CLAP 0.772, MuQ-MuLan 0.752–0.767, MAEST
  0.746, MERT 0.652.
- **Longer audio.** Each doubling of the excerpt adds only about 3.5 pp of samples candidate recall (6–12 pp for
  covers). The terms of other preview sources (Apple, SoundCloud, Spotify) forbid this use.
- **Rap flow.** Beat-relative vocal-onset features separate UK from US rap (AUC 0.73–0.79) but add only 0.01 over
  MuQ-MuLan.
- **Junk uploads.** Derivative uploads (sped up, nightcore, karaoke, "type beat") fill 0.4 % of top-10 slots.
- **Feedback.** It exists only for the legacy Spicetify Sounds view, as list-level ratings: not enough to train a
  ranker.

## Same vibe (2026-11-06)

**Why.** Two Best lists that "made no sense" to a listener:

- **Joy Crookes – *Feet Don't Fail Me Now* (UK soul)** opened with Offset – *Prada Myself*.
- **Childish Gambino – *Heartbeat*** opened with Stegosaurus Rex, Deichkind and an Armin van Buuren remix.

A third list had a great first match: deep dispatch – *Something on My Mind* → Rivo – *Stay Here (Jos & Eli
Remix)*.

On 800 development seeds, 55 % of Best's other-artist top-10 results were "incoherent": they share no Wikidata genre
with the seed and ListenBrainz does not list the artist as similar. The incoherent results sit mostly in the protected
head:

| result | incoherent |
| --- | ---: |
| in the protected head | 62 % |
| everywhere else | 39 % |
| in the seed's sounds top 200 | 26 % |
| found by both melody and samples | 38 % |
| found by only one of the two | 56 % |

Strong cover or sample evidence goes with incoherence, because real covers and samples often cross genres. Agreement
with the sound goes with coherence. The great example is a two-channel match: samples #1 and melody #26, at 125 vs
130 BPM.

**What did not help (development data).** The study is in the research workspace (`patterns-research-20261106`):

- **"Sheet music."** Transcribed chords, melody and bass were matched by key-free chord n-grams, melodic intervals and
  chord DTW. Ranking each cover's positive among 4,000 random decoys:
  - chord n-grams reach the top 10 for 24–28 % of covers, against 81 % for the melody channel (CLEWS);
  - transcribed melodies reach 4.5 %;
  - adding the symbolic score to CLEWS + SampleID lowers hit rates;
  - it recovers 5 of 133 covers that CLEWS misses, none high enough for a catalog-scale top 10;
  - extraction would cost about 17 GPU-hours for the catalog.
- **Shazam-style fingerprints.** These were local implementations of Wang's landmarks
  ([2003](https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf)) and of a pitch- and tempo-tolerant matcher in
  the style of [Panako](https://github.com/JorenSix/Panako). They match identical audio only:
  - sample pairs in the top 10: 4–5 %, against 27–30 % for SampleID;
  - SampleID already ranks every fingerprint hit in its top 10 but one, which it has at #14.
- **Changing Best.**
  - Stricter head evidence (a list's top-3 or threshold song must also stand out with z ≥ 2 within its top 20):
    −4.7 pp incoherence on development data.
  - Protecting only the evidenced songs themselves (E2): −8.0 pp incoherence on development data.
  - E2 was confirmed once on 1,000 fresh seeds: incoherence 0.516 → 0.433 [Δ CI −0.095, −0.073].
  - E2 **failed** the held-out relationship rule: same-language covers HR@10 −1.11 pp [−1.78, −0.51] and AP −1.04 pp,
    against a 1.0 pp limit. Best is unchanged. The held-out relationship sets have now been used once more.
- **Gating Best itself by tempo and sound.** Incoherence fell 0.567 → 0.405 on 200 seeds, but covers HR@1 fell 30 pp:
  the two intents conflict in one list. So the gate became a separate list.

**Definition.** Same vibe ([`samevibe.py`](../src/soundalike/anysong/samevibe.py)) builds its list from:

1. **Best's first 100 songs as served,** keeping, in Best order, the songs that pass two gates:
   - **Tempo:** the same tempo, double or half, within 6 %. An unknown tempo passes.
   - **Sound:** the song is in the seed's sounds list, or its MuQ-MuLan score (maxsim minus the hub penalty) is at
     least τ = 0.378. τ is the median 800th sounds score of the development seeds.
2. **Fill:** first the sounds list's tempo-compatible songs, then the rest of the sounds list, then the rest of Best.

Rules that apply throughout:
- One song per artist.
- The language filter works as in Best.
- Tempos come from Beat This! ([Foscarin et al., 2024](https://github.com/CPJKU/beat_this), MIT licence): a catalog
  table, and the same model live for unseen songs.
- Each song's tempo, score, artist and language are read from its first-channel track.

**Selection.** Six variants were fixed before running: pool 50 or 100; sound gate "either", score only or list only;
tempo tolerance 4, 6 or 8 %. All six met the guards on 800 development seeds: at least 9.5 distinct artists, at most
2 % other-language songs, and a full list for at least 95 % of seeds. The rule picked the lowest incoherence:

| list | incoherent |
| --- | ---: |
| pool 100, "either", 6 % (selected) | 0.400 |
| the other five variants | 0.411–0.446 |
| Sounds like | 0.530 |
| Best | 0.554 |

**Confirmation.** It ran once, on 1,000 fresh seeds: one song per artist, artists in no earlier seed or training set,
and ListenBrainz data collected for them. Served same-language mode; Sounds like with one song per artist and the
language filter:

| list | incoherent | ListenBrainz-similar | beyond Deezer's list | shared genre | in sounds top 200 | tempo-compatible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Same vibe** | **0.390** | **0.320** | **0.251** | **0.492** | 0.785 | 0.983 |
| Sounds like | 0.499 | 0.212 | 0.164 | 0.414 | 1.000 | 0.749 |
| Best | 0.516 | 0.220 | 0.166 | 0.403 | 0.184 | 0.434 |

Both pre-registered conditions held:
- incoherence lower than Sounds like: −0.108 [−0.126, −0.091];
- ListenBrainz agreement beyond Deezer's lists not lower: +0.087 [+0.074, +0.102].

Neither Wikidata nor ListenBrainz feeds Same vibe's gates. Best's pool does pass through the scene layer, which uses
Deezer's related artists and Wikidata countries, so agreement beyond Deezer's lists is the cleanest of these checks.

The 25 v4 development pairs are directional only:

| list | v4 pairs in the top 10 |
| --- | ---: |
| Same vibe | 2 |
| Sounds like | 1 |
| Best | 3 |

By design, matches like Rivo, which sound different but share material, stay in Best and not in Same vibe.

**Parity and serving.**
- **Parity:** the API reproduced the research Same vibe list and Best list for all 800 development seeds in both
  language modes (1,600 lists each).
- **Catalog seeds:** each needs exact MuQ-MuLan scores for up to 400 candidates (about 2 ms on the RTX 5080) and a
  tempo lookup in a 1.8 MB table.
- **Unseen seeds on the RTX 2070 SUPER:**
  - the tempo costs 31 ms of GPU time plus 72 ms of decoding (median of 40 previews);
  - the model holds 158 MiB of VRAM;
  - all 40 tempos match the table.

**Limitations.**
- τ belongs to the 2026-11-01 index and must be recomputed when the index changes.
- Same vibe reads Best's order, so a change to Best needs a new confirmation.
- "Incoherent" is a proxy built from genre tags and co-listening, not listening tests.

## Sounds like on the GPU engine (2026-11-06)

**Why.** Spotify's and the main page's Sounds like tab still came from the older `dual_sonic64_guardrail`
service. That service only answers for songs inside its fixed 272,853-track library, so a song outside it
(Shoreline Mafia – *HEAT STICK*, Vegyn – *The Path Less Travelled*) got "isn't in the hosted library" while Best
matches and Same vibe analysed the same song live. Listeners also found Same vibe's songs closer in sound than that
tab's.

**What.** `/similar` now also returns `sounds`: the sonic (MuQ-MuLan) channel's songs in order, language-filtered like
Best, one song per artist, each with its catalog tempo. Both clients show it as Sounds like. The older service only
answers when the matcher can't (and the local companion keeps its own list). This is the "website sounds tab" list
that was measured in the Sounds like v2 study, and there it beat the older list on development and on held-out
seeds. One song per artist and other-artist results only, top 10:

| list | seeds | ListenBrainz | beyond Deezer's list | shares a genre | other language |
| --- | --- | ---: | ---: | ---: | ---: |
| older Sounds like | 800 development | 0.167 | 0.126 | 0.309 | 16.8 % |
| GPU sound list | 800 development | 0.191 | 0.147 | 0.367 | 15.3 % |
| older Sounds like, language gate | 800 development | 0.189 | 0.142 | 0.330 | 0 % |
| GPU sound list, language gate (as served) | 800 development | 0.214 | 0.163 | 0.389 | 0 % |
| older Sounds like | 799 held-out | 0.156 | 0.124 | 0.313 | 17.2 % |
| GPU sound list | 799 held-out | 0.191 | 0.151 | 0.353 | 15.4 % |
| older Sounds like, language gate | 799 held-out | 0.177 | 0.141 | 0.335 | 0 % |
| GPU sound list, language gate (as served) | 799 held-out | 0.211 | 0.169 | 0.380 | 0 % |

The held-out numbers come from the Sounds like v2 confirmation run, which measured both lists as comparators; this
switch was not its pre-registered question, so they are evidence rather than a confirmation. Same vibe is a different
list (Best's candidates gated by tempo and sound); on its own fresh-seed test it scored higher still (ListenBrainz
0.320).

**Parity.** The production list (`api.AnySong` on the full index with the language, scene and tempo tables)
reproduced the research list's development metrics exactly: ListenBrainz 0.214, beyond Deezer's list 0.163, genre
0.389, other language 0 on the 800 development seeds. Lists average 19.8 songs (a seed whose language is rare in the
catalog can get fewer), and 99.5 % of their top-10 rows carry a tempo.

**Limitations.** The list has no feedback prompt yet (that prompt and its server still accept only the older
methods). It recommends only from the 269,278-track catalog, like the other hosted lists.

## Catalog growth to about 1M tracks (pre-registered 2026-11-07)

**Status.** These rules were committed before the grown index existed and before any of its results were seen.
Results are appended below this section unchanged.

**Change under test.** About 730,000 songs added by `anysong.crawl` (popular songs of the catalog's artists, then
related artists, most-followed first) and embedded with the catalog's models and settings. Parity on 256 catalog
songs re-embedded through the same pipeline: SampleID and MuQ-MuLan bit-identical, CLEWS window cosine
≥ 0.9999997, language top-1 100 %, tempo within 0.5 BPM for 252 of 254. The index uses the compressed stage 1
above. Both conditions are served by the same API code with the same settings; only the catalog differs.

**Labels.** ListenBrainz similar-artist hits, Deezer related lists, Wikidata genres and the language table are
rebuilt to cover the grown catalog, and both conditions are scored with those same labels. Without that, every new
artist would count as a miss: a recommended artist only matches ListenBrainz by MusicBrainz ID or by name among
catalog artists.

**Seeds.** The 799 held-out seeds decide; the 800 development seeds are for diagnosing and fixing problems. No
setting is tuned on held-out seeds.

**Must pass (all), else not deployed:**

1. **Lists users see.** Best matches, Same vibe and Sounds like as served (language gate on, top 10, other-artist
   songs, one per artist), paired by seed, grown minus current:
   - ListenBrainz hit rate: the 95 % paired-bootstrap interval of the difference has its lower bound at −0.010 or
     above.
   - Shares a genre: the mean difference is −0.02 or above.
   - Other-language rows: up by no more than 1 point.
   - Derivative uploads (karaoke, sped-up, slowed, tribute and the like, as in the research filters): up by no more
     than 1 point.
2. **Relationship benchmarks.** Any-tab HR@10 may drop by at most 0.03 on covers (held-out), samples (held-out) and
   samples reverse (held-out), and by at most 0.10 on Gen Alpha (test, 59 queries). More decoys can only push
   labelled matches down. When the catalog grew from 25k to 271k, covers lost 0.063 and Gen Alpha 0.238.
3. **Exactness.** Against a brute-force exact search, the served top 20 is identical for at least 99 % of 300
   random tracks per channel.
4. **Serving on the RTX 2070 SUPER.**
   - Peak VRAM of the API process at most 7.6 GB of 8.
   - p95 latency for an unseen song through the API at most 1.0 s.
   - Startup under 2 minutes.
   - `serve_check` parity: re-embedded catalog tracks rank themselves first.

If rule 1 passes and only rule 2 fails, the release waits for a decision with the evidence: how many lost hits
were displaced by unlabelled versions of the same work (same normalized title, different artist).

**Reported, not gated:**
- Share of each held-out seed's ListenBrainz-similar artists present in the catalog (coverage).
- Share of top-10 rows that are new songs.
- The same list metrics on development seeds.

### Result: the unrestricted grown catalog failed (2026-11-08)

Run once as registered, on the 889,277-song index: the 269,278 catalog songs plus the 620,000 crawled songs
embedded by then. Rules 1 and 2 failed and rule 3 passed. Rule 4 was not run, and nothing was deployed.

Rule 1 on the 799 held-out seeds, ListenBrainz hit rate (current → grown, difference with its 95 % interval):

| List | Current | Grown | Difference |
|---|---:|---:|---|
| Best matches | 0.215 | 0.119 | −0.096 (−0.108 to −0.085) |
| Same vibe | 0.314 | 0.193 | −0.120 (−0.134 to −0.107) |
| Sounds like | 0.211 | 0.113 | −0.098 (−0.110 to −0.086) |

The other rule-1 checks passed:
- Shared genre: −0.012, −0.014 and −0.009.
- Other-language rows: unchanged.
- Derivative uploads: down by up to 0.2 points.

Rule 2 (any-tab HR@10):

| Benchmark | Current | Grown | Change | Verdict |
|---|---:|---:|---:|---|
| Covers | 0.709 | 0.653 | −0.056 | fail |
| Samples | 0.162 | 0.135 | −0.027 | pass |
| Samples reverse | 0.170 | 0.144 | −0.026 | pass |
| Gen Alpha | 0.508 | 0.339 | −0.170 | fail |

Rule 3: the served top 20 matched brute force for all 300 queries in every channel.

Reported measures:
- Coverage: the share of held-out seeds' ListenBrainz-similar artists in the catalog rose from 75.7 % to 99.6 %
  (492 seeds with ListenBrainz data).
- New songs: crawled songs made up 60 %, 56 % and 63 % of the three lists' top 10.

**Diagnosis on development seeds only.** ListenBrainz hit rate of Best-list rows, by how far the row's artist is
from the seed's artist in Deezer's related graph:

| Rows | Related | Two steps | Farther |
|---|---:|---:|---:|
| Catalog songs, current index | 0.633 | 0.369 | 0.052 |
| Crawled songs, grown index | 0.329 | 0.128 | 0.004 |

Crawled songs from farther away filled about 45 % of the rows (2,139 rows over 800 seeds) and were almost never
hits. The crawl reaches long-tail artists whose audio is close to the seed's, but their listeners are not the seed's
listeners. ListenBrainz also knows less about long-tail artists, so part of this gap is missing labels. Crawled
songs by the seed's related artists were hits a third of the time.

### Amendment: catalog tiers (2026-11-08)

Written before any held-out, fresh or out-of-catalog result of this change.

**Change under test.** The same grown index and tables, served with `--core-ids` (`anysong.tiers`):
- A crawled song is a candidate only for seeds by its own artist, or by an artist whose own Deezer related list
  names its artist.
- The 269,278 catalog songs stay candidates for every seed.

The rule was chosen on development seeds and has no tuned parameters. The mask is applied before the index picks
the candidates it scores exactly, so lists keep their length. The same code without `--core-ids` reproduced the
first run's current lists exactly for all 800 development and 799 held-out seeds.

Development seeds, current → gated, as design data:

| List | Current | Gated | 95 % interval of the difference |
|---|---:|---:|---|
| Best matches | 0.210 | 0.215 | −0.000 to +0.011 |
| Same vibe | 0.322 | 0.326 | −0.004 to +0.011 |
| Sounds like | 0.213 | 0.226 | +0.007 to +0.018 |

Shared genre, other-language rows and derivative uploads are each within 0.01. Crawled songs make up 4 %, 7 % and
4 % of the three lists' top 10.

**Decision.** All four rules apply as registered, with two adjustments:
- Rule 2 runs `eval_index --core-ids`, and rule 4 runs against the gated service.
- The held-out seeds were already scored once, for the ungated build, so rule 1 must also pass on 1,000 fresh
  seeds. These are artist-disjoint catalog seeds drawn for the pattern research on 2026-11-06
  (`fresh_seeds.json`) and never used for the catalog question. Their ListenBrainz labels were fetched the same
  way.

**Reported, not gated: seeds outside the current catalog.** Development, held-out and fresh seeds are all catalog
songs, so they cannot show what the crawl adds for songs the catalog lacks. These seeds are selected as follows:
1. Take artists with no catalog song, at least five songs in the grown index and a MusicBrainz ID.
2. Shuffle them with `random.Random(20261108)`.
3. Draw one song per artist uniformly, and keep the first 800.

Their ListenBrainz similar artists are fetched the same way as the fresh set's. "Current" is the live API path for
an unseen song: its preview windows, read from the embedding shards, ranked against the 269k index, with the
related artists Deezer lists. "Gated" serves the song from the grown index. Scoring is as in rule 1.

### Result: catalog tiers passed rules 1 to 3 (2026-11-08)

Run once on each set after the amendment was committed (c70e110).

Rule 1, ListenBrainz hit rate (current → gated, difference with its 95 % interval; it must stay at −0.010 or
above):

| Seeds | List | Current | Gated | Difference |
|---|---|---:|---:|---|
| Held-out (799) | Best matches | 0.215 | 0.217 | +0.002 (−0.003 to +0.007) |
| | Same vibe | 0.314 | 0.316 | +0.002 (−0.004 to +0.008) |
| | Sounds like | 0.211 | 0.217 | +0.006 (+0.002 to +0.011) |
| Fresh (1,000) | Best matches | 0.212 | 0.219 | +0.007 (+0.002 to +0.012) |
| | Same vibe | 0.316 | 0.321 | +0.004 (−0.002 to +0.011) |
| | Sounds like | 0.210 | 0.215 | +0.006 (+0.001 to +0.011) |

The other rule-1 checks passed on both sets:
- Shared genre: from +0.002 to +0.012.
- Other-language rows: unchanged.
- Derivative uploads: unchanged.

Crawled songs made up 3 % to 7 % of the top 10. (Erratum to the amendment: on development seeds, Sounds like's
shared genre rose by 0.013, not by less than 0.01.)

Rule 2 (any-tab HR@10, with `--core-ids`):

| Benchmark | Current | Gated | Change | Limit |
|---|---:|---:|---:|---:|
| Covers | 0.709 | 0.707 | −0.002 | −0.03 |
| Samples | 0.162 | 0.165 | +0.003 | −0.03 |
| Samples reverse | 0.170 | 0.168 | −0.001 | −0.03 |
| Gen Alpha | 0.508 | 0.508 | 0.000 | −0.10 |

Rule 3: with the masks on both sides, the served top 20 and top 100 matched brute force for all 300 random tracks
in every channel.

**Reported, not gated: seeds outside the current catalog.** 800 seeds, drawn from 5,236 eligible artists:

| List | Current (unseen song) | Gated | Difference |
|---|---:|---:|---|
| Best matches | 0.072 | 0.138 | +0.066 (+0.056 to +0.077) |
| Same vibe | 0.103 | 0.186 | +0.084 (+0.070 to +0.098) |
| Sounds like | 0.076 | 0.139 | +0.063 (+0.051 to +0.075) |

Further changes for these seeds:
- Other-language rows fell from 37 %, 33 % and 38 % to none. The current API cannot tell the language of an unseen
  song by an artist outside the catalog, so its language filter stays off; the grown catalog has labels for every
  crawled song.
- Shared genre rose by 0.035, 0.051 and 0.043.
- Crawled songs made up 22 %, 25 % and 18 % of the top 10.

Sounds like lists were slightly shorter: 9.7 rows in the top 10 on average.

### Rule 4 on the RTX 2070 SUPER: latency over the limit, not deployed (2026-11-08)

Two deploys with automatic rollback, both rolled back.

The first deploy's checks:
- `serve_check` parity: all 64 re-embedded tracks per channel ranked themselves first; window cosine at least
  0.999999.
- `serve_check` VRAM: 5,445 MiB loaded, 6,797 MiB peak.
- `serve_check` serial unseen p95: 847 ms, against 555 ms for the 269k index.
- Startup: 45 s, with the tiers active.

Its latency check through the API was invalid. It sent 40 requests back to back from one address and 32 got
HTTP 429 from the API's own rate limit. It also requested songs by id, which adds a Deezer track lookup that the
original serving measurement did not include.

**Corrected method.** As in the serving measurement above, each song is searched, then requested from `/similar`,
and only that request is timed. Requests are paced under the rate limits. The 40 songs are taken from crawl chunk 30,
which neither index holds. Both services were measured the same way on the same songs:

| Unseen song | Current (269k) | Gated grown |
|---|---:|---:|
| p50 | 0.932 s | 0.997 s |
| p95 | 1.125 s | 1.046 s |
| max | 1.416 s | 1.192 s |
| peak VRAM | 6,215 MiB | 6,435 MiB |

The gated grown catalog missed the 1.0 s p95 limit by 46 ms, so the deploy script rolled it back. The current
service misses the limit too. The limit was set from the 610 to 670 ms measured before Same vibe's tempo tracking
and the scene layer's related-artist lookup were added.

Medians per stage:
- **Search:** 201 ms on the current index, 413 ms on the grown one. The samples channel went from 133 to 310 ms.
- **Related artists:** on the current service, the API waited a median 137 ms (up to 616 ms) after the search for
  Deezer's list. On the grown index, the list had arrived by the time the longer search ended.

**Serving fix found during the attempts.** In the first deploy, the API fetched the seed's related artists from
Deezer before searching, for the tiers. The fetch runs in a background thread and took about 880 ms while the GPU
work ran, against about 300 ms on its own (probably contention for Python's GIL; not verified). Serving now gates
with the related lists stored in the scene table, as the evaluation did, so the search never waits for Deezer. 150
development and 150 out-of-catalog seeds reproduced the evaluated gated lists exactly.

**Decision (2026-11-08, after the results above).** The project owner changed rule 4's latency limit for this
release to "unseen-song p95 no higher than the current service's, measured the same way". The current service
measured 1.125 s and the gated grown catalog 1.046 s. The original 1.0 s limit predates Same vibe and the scene
layer, and the current service no longer meets it either. The deploy re-measures against 1.125 s, with the other
rule-4 checks and automatic rollback unchanged.

**Deployed (2026-11-08).** The third deploy passed every rule-4 check against that limit:

| Check | Result |
|---|---|
| Startup | 44 s, catalog tiers active |
| Unseen-song latency, 40 songs | p50 0.99 s, p95 1.032 s, max 1.087 s |
| Peak VRAM | 6,435 MiB |

Checks through the public URL:
- A catalog song answered in 0.62 s.
- A crawled song by an artist new to the catalog answered from the index in 0.57 s; the previous index had to
  fetch and embed it live.
- Every crawled song in the returned lists was by the seed's artist or one of its related artists.
- Fixed seeds captured before and after the swap kept most of their top 10; zero to five songs changed per list.

### The full crawl, about 1M songs (pre-registered 2026-11-08, before any result)

**Change under test.** All 730,016 crawled songs (the crawl reached its target at 21:54), indexed with the same
settings:
- **Catalog:** 999,293 songs, the 269,278 core songs plus the crawl's. Catalog tiers are unchanged (same core ids).
- **Tables:** the scene table, languages and tempo are rebuilt from the full crawl, including Wikidata facts for
  13,207 new artists. The hub penalty is rebuilt.
- **Serving:** stage 1 may run as an fp16 GEMM for channels where that is measured faster on the RTX 2070 SUPER and
  exact (`--half-stage1`, decided before rules 1 to 3 run, which then use the served setting).

**Baselines.** Both the served gated 889k catalog and the original 269k catalog. Every rule must pass against both.

**Must pass (all), else not deployed:**

1. **Lists users see,** on the 799 held-out seeds (used twice before) and the 1,000 fresh seeds (used once
   before). Nothing is tuned on either set. The ListenBrainz hit rate's paired 95 % interval must have its lower
   bound at −0.010 or above. Limits on mean change:
   - Shared genre: −0.02 or above.
   - Other-language rows: up by at most 1 point.
   - Derivative uploads: up by at most 1 point.

   Labels are rebuilt for the bigger catalog and score all three conditions.
2. **Relationship benchmarks** (`eval_index --core-ids`, served stage-1 setting): any-tab HR@10 may drop by at most
   0.03 on covers, samples and samples reverse (held-out) and by at most 0.10 on Gen Alpha.
3. **Exactness** with the tier masks and the served stage-1 setting: the served top 20 matches brute force for at
   least 99 % of 300 random tracks per channel.
4. **Serving on the RTX 2070 SUPER:**
   - `serve_check` parity: every re-embedded catalog track ranks itself first.
   - Startup under 2 minutes.
   - Peak VRAM at most 7,248 MiB.
   - Unseen-song p95 no higher than the served catalog's (the owner's "no slower than now"). Both are measured in the
     same deploy session, before and after the swap, on the same 40 songs. These are taken from 50 random
     never-crawled frontier artists, one top track each, so neither catalog holds them. Each song is searched,
     then requested; only that request is timed. The crawl is stopped during both measurements.

**Half-precision stage 1 decision rule.** fp16 is enabled for a channel only if:
- On the RTX 2070 SUPER, its stage-1 time per row is at least 10 % lower than fp32's (median of repeated runs on a
  slice of the served index).
- On the RTX 5080 at 999k, its top-20 exactness (300 random tracks) is no lower than fp32's.

**Half-precision stage 1, measured.** RTX 2070 SUPER, a 65,536-row slice of each channel of the served 889k index,
median of 40 runs per setting, per million tracks, while the API kept serving:

| Channel | fp32 | fp16 | Change | Stage-1 top-100 overlap |
|---|---:|---:|---:|---:|
| Melody (8-bit) | 53.1 ms | 56.5 ms | +6 % | 1.00 |
| Samples (4-bit) | 256.3 ms | 196.4 ms | −23 % | 0.99 |
| Sound (8-bit) | 16.0 ms | 18.3 ms | +14 % | 0.99 |

Stage 2, the exact re-scoring of the candidates read from disk, took 9.5, 69.5 and 8.8 ms. By the rule, fp16
qualifies for samples only, pending the exactness comparison at 999k.

### Result: the full crawl passed rules 1 to 3 (2026-11-09)

Run once, after the pre-registration (e07ab48) and the fp16 commit (e550a83), with the served setting: samples'
stage 1 in fp16, melody and sound in fp32. At 999k on the RTX 5080, samples' top-20 exactness was 1.000 in fp16
and in fp32, so fp16 for samples meets its decision rule.

Rule 1, ListenBrainz hit rate (paired difference with its 95 % interval; the lower bound must be −0.010 or above
against both baselines):

| Seeds | List | 269k | 889k (served) | 999k | Against 889k | Against 269k |
|---|---|---:|---:|---:|---|---|
| Held-out (799) | Best matches | 0.215 | 0.217 | 0.217 | +0.001 (−0.003 to +0.004) | +0.002 (−0.003 to +0.008) |
| | Same vibe | 0.314 | 0.316 | 0.317 | +0.001 (−0.002 to +0.004) | +0.003 (−0.003 to +0.010) |
| | Sounds like | 0.211 | 0.217 | 0.218 | +0.000 (−0.002 to +0.003) | +0.007 (+0.002 to +0.012) |
| Fresh (1,000) | Best matches | 0.212 | 0.219 | 0.220 | +0.001 (−0.002 to +0.004) | +0.008 (+0.002 to +0.013) |
| | Same vibe | 0.316 | 0.321 | 0.321 | +0.001 (−0.002 to +0.003) | +0.005 (−0.002 to +0.012) |
| | Sounds like | 0.210 | 0.215 | 0.216 | +0.001 (−0.001 to +0.003) | +0.007 (+0.002 to +0.012) |

The other rule-1 checks passed against both baselines:
- Shared genre: −0.003 to 0.000 against 889k, +0.002 to +0.011 against 269k.
- Other-language rows: unchanged.
- Derivative uploads: changes between −0.0005 and +0.0001.

Crawled songs made up 3 % to 8 % of the top 10 (3 % to 7 % at 889k).

**Reported, not gated.** Among recommended artists that Deezer does not list as related to the seed, the
ListenBrainz hit rate barely moved against 889k (largest change −0.001). Against 269k it fell in Same vibe on the
fresh seeds, by 0.009 (−0.016 to −0.002). The 889k catalog already had this drop (−0.008, −0.015 to −0.001); its
result above did not report it.

Rule 2 (any-tab HR@10, with `--core-ids` and samples in fp16):

| Benchmark | 269k | 889k | 999k | Against 889k | Against 269k | Limit |
|---|---:|---:|---:|---:|---:|---:|
| Covers | 0.709 | 0.707 | 0.707 | 0.000 | −0.002 | −0.03 |
| Samples | 0.162 | 0.165 | 0.162 | −0.003 | 0.000 | −0.03 |
| Samples reverse | 0.170 | 0.168 | 0.168 | 0.000 | −0.001 | −0.03 |
| Gen Alpha | 0.508 | 0.508 | 0.508 | 0.000 | 0.000 | −0.10 |

Rule 3: with the tier masks, the served top 10, 20 and 100 matched brute force for all 300 random tracks in every
channel, with samples in fp16 and in fp32.

### Rule 4, first deploy: rolled back on parity (2026-11-09)

The deploy script measured the served 889k catalog first: 40 of 48 candidate songs answered live (8 were not
returned by the search), p50 1.003 s, p95 1.065 s, max 1.129 s, 6,435 MiB. `serve_check` on the 999k index then
reported:
- Window cosine at least 0.999999 in every channel; self-score difference 1.3e-5 on average.
- VRAM: 5,737 MiB loaded, 7,141 MiB peak.
- Serial unseen p95: 807 ms (847 ms for the 889k index).
- Self-rank: 64 of 64 first in melody and samples, **63 of 64 in sound**.

The script rolled back on the sound-channel parity before the swap; the 889k service came back healthy.

**Cause.** The one track was The Clash's "Straight to Hell (Remastered)" (core catalog, Deezer 69962767). The core
catalog also holds "Straight to Hell" (69505565), the same recording on another release: their stored windows have
cosine 0.998. With the hub penalty (alpha 0.5), the other release scores 0.5408 and the track itself 0.5371, so it
ranks second from its stored windows too. It ranks second in the 269k and 889k indexes as well.

**How often.** Self-rank from the stored windows, with the served settings, for 1,000 random core songs and 1,000
crawled songs in each index:

| Index | Melody | Samples | Sound | Crawled songs, all channels |
|---|---:|---:|---:|---:|
| 269k | 999 | 998 | 997 | (none in the index) |
| 889k (served) | 999 | 998 | 997 | 1,000 |
| 999k | 999 | 998 | 998 | 1,000 |

Every track not first, in every index, sat behind another release of the same piece by the same artist with
identical or near-identical windows (cosine 0.996 to 1.0 in sound and samples; a perfect melody score in both
releases). Examples: Chopin's waltz Op. 69 No. 2 by Ingrid Fliter under two titles, and "Glass: Metamorphosis
Four" by Valentina Lisitsa, once titled "Glass: Glass: Metamorphosis Four". With identical windows, the lower
Deezer ID wins the tie. At these rates (2 to 3 tracks per 1,000 core songs, none in 1,000 crawled songs), a
64-track draw includes such a track in about 17 % of deploys of the 269k index, whose draws are all core songs,
and 3 % to 6 % for the grown indexes. By default, lists drop other releases of the seed with the same song key;
releases under other titles can appear. This is a property of the catalog, not of the serving path, which the
check exists to test.

`serve_check` now also ranks each sampled track from its stored windows on the serving GPU and reports
`rank_agree`, the share of tracks whose re-embedded rank equals their stored rank, plus the ids of any track not
first.

**Amendment (2026-11-09, owner's decision, before the second deploy).** Rule 4's parity check becomes: every
re-embedded track ranks exactly where its stored windows rank it on the serving GPU (`rank_agree` 1.0 in every
channel), and at least 62 of the 64 rank themselves first. At the measured rates, three or more of 64 in one
channel has a chance well under 1 in 10,000. The rest of rule 4 is unchanged: startup under 2 minutes, peak VRAM at most 7,248 MiB, and
unseen-song p95 no higher than the served catalog's, measured again in the same session on the same songs, with
automatic rollback. The deploy does not start during the nightly backup (02:45 to 04:00).

**Deployed (2026-11-09).** The second deploy passed every rule-4 check:

| Check | Result |
|---|---|
| Parity | `rank_agree` 1.0 in every channel; all 64 first in melody and samples, 63 in sound (the same remaster, second from its stored windows too) |
| Startup | 52 s, catalog tiers active |
| Unseen-song latency, 40 songs | p50 0.972 s, p95 1.024 s, max 1.087 s; the 889k catalog, just before: p50 1.009 s, p95 1.078 s, max 1.155 s |
| Peak VRAM | 6,727 MiB (6,435 MiB for the 889k catalog) |

Checks through the public URL:
- Two catalog seeds and three songs by artists new in this build answered from the index in 0.53 to 0.87 s.
- Two seeds outside the catalog answered live in 1.07 and 2.42 s.
- 29 crawled songs appeared in those lists, each by the seed's artist or one of its related artists.
- Six fixed seeds captured before and after the swap changed zero to two songs per top-10 list. Their requests took
  a median 384 ms after the swap and 500 ms before. The samples search went from 347 to 269 ms, in line with the
  fp16 benchmark, but the fp32 stages were faster too. The crawl ran during the first capture and not the second,
  so this comparison is not controlled.

## Ranking for the lists users see (pre-registered 2026-11-09, before any held-out result)

**Why.** A test of the served 999k catalog on 2026-11-09 found lists that often made little sense:
- 1,018 songs from Deezer's charts of 40 countries and 23 genres.
- 180 song pairs, each checked against a Wikipedia article.
- The owner listening in Spotify.

For seeds in a thinly covered language, songs without a language label filled the lists: K-pop, trance, film
scores, and English songs with no label. Hebrew seeds got 70 % such rows.

Best matches put up to seven weak audio matches above any song from the seed's scene, because a top-3 place in the
melody or samples list counted as evidence whatever its score. Only 41 % of results for UK chart seeds were by UK
artists. Copies and covers of the seed sometimes topped or filled the lists. Sounds like had the weakest ListenBrainz
agreement of the three lists.

**Change under test.** The index, the tables and the channel lists are unchanged. Everything below happens in
`AnySong._view` and the defaults of `SceneParams`:
1. **Language gate.** A listed song without a confident language label takes its artist's majority language
   (`LanguageTable.artist_language`, until now used only for seeds).
2. **Copies of the seed.** Another artist's version of the seed's song (same `song_key` title) scoring at least
   0.55 in the samples list is dropped as a copy of the seed's recording. Of the remaining versions, at most 3 stay.
3. **Best matches.** Evidence by score only (`rank_evidence` 0). After the three-song audio head, the scene lane
   takes every slot (`pattern` "S"). The continuation boost goes from 2.0 to 4.0.
4. **Sounds like.** Songs move up by their artist's scene affinity: 1 / (5 + rank) × (1 + 8 × affinity). The list
   still draws only from the 200 nearest songs in sound.

Song lookups (artist spellings, bracketed titles, a normalized second search, ISRC) were fixed in the same release.
They change which recording answers a title, not how lists rank, and were checked separately. The 1,018 sweep songs
and the pair songs were resolved again:
- 983 sweep songs resolved as before.
- 6 changed, each to the version asked for (HUNTR/X "Golden" had resolved to "Golden (Acapella)").
- 8 lookups that had failed now succeed.

**Selection.** 33 variants were compared on development data only:
- The scene layer's 800 development seeds.
- The 676 catalog songs of the chart sweep (development data from now on).
- The development and validation splits of the relationship benchmarks.

Lists came from the served code path with the GPU search cached (`grow/qa/devharness.py`). On that data, against
the served lists:

| | Best matches | Same vibe | Sounds like |
|---|---:|---:|---:|
| ListenBrainz hit rate | +0.102 | +0.028 | +0.119 |
| Far-genre share | −0.082 | −0.021 | −0.060 |

Best matches HR@10 changed by −0.019 on covers, −0.010 on samples and −0.013 on samples reverse.

Variants were rejected for these reasons:
- **A one-song head** gained more (Best matches +0.126) but lost 0.034 on samples reverse.
- **A copy rule at 0.45** cost 0.024 on covers.
- **A rank-based copy rule** cost 0.128 on covers, since faithful covers top both the melody and samples lists.

**Baselines.** The served 999k lists (`gated1m`), computed before this change and stored.

**Seeds.**
- Rule 1: the 799 held-out seeds (used four times before) and the 1,000 fresh seeds (used three times before).
- Rule 2: the relationship benchmarks' held-out split, plus Gen Alpha.

**Must pass (all), else not deployed:**
1. **Lists users see.** Best matches, Same vibe and Sounds like, each on both seed sets (`gate_eval compare
   --labels build1m`):
   - ListenBrainz hit rate: the paired 95 % interval's lower bound at −0.010 or above.
   - Shared genre: mean change −0.02 or above.
   - Other-language rows: up by at most 1 point.
   - Derivative uploads: up by at most 1 point.
   - Rows shown in the top 20: down by at most 1.0 on average. The language gate shortens some lists.
2. **Relationship benchmarks.**
   - Any-tab HR@10 may drop by at most 0.03 on covers, samples and samples reverse, and by at most 0.10 on Gen Alpha.
     The channel lists are unchanged, so these should not move.
   - Best matches HR@10, as served, may drop by at most 0.03 on covers, samples and samples reverse.
3. **It must help.** On both seed sets, the ListenBrainz hit rate of Best matches and of Sounds like must rise,
   with the paired 95 % interval's lower bound above 0.
4. **Serving on the RTX 2070 SUPER.** Rule 4 of the 999k catalog with the amended parity check: startup under
   2 minutes, peak VRAM at most 7,248 MiB, and unseen-song p95 no higher than the served service's, measured in the
   same session on the same songs, with automatic rollback.

Exactness (rule 3 before) is untouched: the index and its search do not change.

### Result: rules 1 to 3 passed (2026-11-09)

Run once, after the pre-registration (64664c5), on the served 999k index (samples' stage 1 in fp16). All 41
checks passed (`grow/qa/verdict_v2.py`).

Rules 1 and 3, top 10 with one song per artist (paired 95 % intervals):

| Seeds | List | ListenBrainz hit rate | Change | Shared genre | Rows in the top 20 |
|---|---|---|---|---|---|
| Held-out (799) | Best matches | 0.217 → 0.322 | +0.105 (+0.093 to +0.118) | 0.367 → 0.452 | 20.00 → 19.96 |
| | Same vibe | 0.317 → 0.341 | +0.025 (+0.019 to +0.031) | 0.454 → 0.477 | 19.99 → 19.93 |
| | Sounds like | 0.218 → 0.331 | +0.113 (+0.101 to +0.125) | 0.386 → 0.472 | 19.80 → 19.52 |
| Fresh (1,000) | Best matches | 0.220 → 0.315 | +0.095 (+0.084 to +0.105) | 0.405 → 0.485 | 20.00 → 19.93 |
| | Same vibe | 0.321 → 0.344 | +0.023 (+0.016 to +0.029) | 0.496 → 0.517 | 20.00 → 19.94 |
| | Sounds like | 0.216 → 0.331 | +0.115 (+0.104 to +0.125) | 0.425 → 0.500 | 19.84 → 19.67 |

Other-language rows stayed at zero. Derivative uploads changed by −0.0008 to +0.0003.

**Reported, not gated.**
- The scene layer reads Deezer's related-artist graph, which may share signal with ListenBrainz. Among recommended
  artists that Deezer does not list as related to the seed, the ListenBrainz hit rate also rose:

  | List | Held-out | Fresh |
  |---|---|---|
  | Best matches | +0.080 | +0.067 |
  | Same vibe | +0.023 | +0.023 |
  | Sounds like | +0.096 | +0.087 |

  Every interval excludes zero.
- Distinct artists in the top 10 fell slightly: Sounds like 9.97 → 9.88 (held-out) and 9.97 → 9.92 (fresh); the
  other lists by about 0.01.

Rule 2, HR@10 on the held-out split:

| Benchmark | Any tab | Best matches as served | Limit |
|---|---|---|---|
| Covers | 0.707 → 0.707 | 0.681 → 0.662 (−0.019, 1,170 queries) | −0.03 |
| Samples | 0.162 → 0.162 | 0.135 → 0.130 (−0.005, 747) | −0.03 |
| Samples reverse | 0.168 → 0.168 | 0.155 → 0.137 (−0.018, 695) | −0.03 |
| Gen Alpha (test) | 0.508 → 0.508 | 0.373 → 0.356 (−0.017, 59; not gated) | −0.10 (any tab) |

The held-out costs match development (−0.019, −0.010, −0.013).

### Rule 4: rolled back on latency (2026-11-09)

**First attempt: stopped before the swap.** The baseline came from the service's result cache. The service still
held the same 40 songs from the 999k deploy, so the baseline p95 was 0.029 s. The run was stopped during
`serve_check`; the served code never changed. `measure_unseen.py` now ignores cached answers, and the service was
restarted to clear its cache.

**Second attempt: rolled back on latency.** It passed parity: `rank_agree` 1.0 in every channel, with the Clash
remaster second in sound as before. It started in 52 s with the catalog tiers active. Then it rolled back on
latency:

| 40 unseen songs | p50 | p95 | max | Server total, median |
|---|---|---|---|---|
| Served code, 04:53 (baseline) | 0.970 s | 0.992 s | 1.130 s | 943.9 ms |
| New code, 05:04 | 0.972 s | 1.022 s | 1.089 s | 944.8 ms |
| Served code, 01:00 (999k deploy) | 0.972 s | 1.024 s | 1.087 s | 943.3 ms |

Per song, the new code took a median 8 ms longer (server total +6 ms). The change runs after the search, and the
extra time sat in stages it does not touch (MuQ-MuLan embedding +4.9 ms).

The same code on the same songs measured 0.992 s and 1.024 s p95 in two sessions four hours apart (paired median
difference +0.3 ms). The p95 of 40 songs moves by about 30 ms between runs. A rule with no margin therefore fails
code that changes nothing on this path about half the time.

The service went back to the served code automatically. Peak VRAM was 6,727 MiB throughout.

**Amendment (2026-11-09, owner's decision, after the rolled-back attempt and before the next).** This release changes
nothing before or during the search. For it, the latency check becomes two conditions on the same 40 unseen songs,
measured in the same session with cold caches:
- The paired median of the server's total time (new code minus served code) is at most +25 ms.
- p95 is at most the served code's p95 plus 50 ms. The margin is above the 32 ms spread between two runs of
  identical code.

The rest of rule 4 is unchanged:
- Parity as amended above.
- Startup under 2 minutes.
- Peak VRAM at most 7,248 MiB.
- Automatic rollback.
- No deploy during the nightly backup.

The rolled-back attempt would have passed this check (+6 ms; 1.022 s against 1.042 s). The amendment was therefore
decided after seeing a result, and the deploy measures again.

**Deployed (2026-11-09).** The third attempt passed every rule-4 check:

| Check | Result |
|---|---|
| Parity | `rank_agree` 1.0 in every channel; all 64 first in melody and samples, 63 in sound (the Clash remaster) |
| Startup | 57 s, catalog tiers active |
| Unseen-song latency, 40 songs | p50 0.976 s, p95 1.015 s, max 1.118 s. Served code just before: p50 0.960 s, p95 1.005 s, max 1.096 s. Limit 1.055 s |
| Paired median server time | +13.0 ms (limit +25 ms) |
| Peak VRAM | 6,727 MiB |

The extra time sat in stages the change does not touch: MuQ-MuLan +4.1 ms, and each of the three searches +1.2 to
+2.5 ms. Both measurements of the new code ran right after `serve_check`, which loads the GPU for several minutes;
the served code's never did. That order may explain why those stages were slower both times.

Checks through the public URL:
- "Bitches Brew" by "✝✝✝ (Crosses)", a 404 before, resolved to "††† (Crosses)".
- "Where Is The Love?" by "Black Eyed Peas" with its ISRC, a 404 before, resolved from the catalog.
- The same ISRC alone still returns 404: Deezer maps it to an old release with no preview in any country.
- The Crosses ISRC alone resolved through Deezer's ISRC lookup.
- Nemzzz "8PM": Kidwild, Central Cee, Headie One, Nemzzz himself, Digga, Krillz, Blanco and AntsLive now fill
  places 4 to 12 of Best matches; before, the first of them (Headie One) was 8th and Central Cee 12th. Sounds like
  opens with Kidwild, Central Cee, LeoStayTrill and Krillz.
- Brent Faiyaz "Paper Soldier": the d3puu copy is gone from first place.

## Listener-aware ranking (pre-registered 2026-11-09, before any held-out result)

**Why.** After the ranking change above, about a third of the other artists in the lists were ones ListenBrainz
lists as similar to the seed's artist: Best matches 0.32, Sounds like 0.35. Ordering, not retrieval, held this back.
On 825 development seeds:
- The songs the search already returns hold a median of 24 listener-similar artists. Perfect reordering would reach
  0.93; within Sounds like's 200 nearest songs by sound, 0.83.
- 73 % of Best's top-10 rows were by artists unrelated to the seed's artist on Deezer. Those rows were hits 26 % of
  the time. Rows by Deezer-related artists were hits 61–76 % of the time.
- For some seeds the scene's songs are not retrieved at all. All 20 of Roar's Deezer-related artists have catalog
  songs, but only 5 appear among the 600 songs the search returns for "I Can't Handle Change".

**Change under test** (`listener.py`; `AnySong._listener` and `_listener_lists`):
1. **Candidates.** The channel lists, plus scene candidates. For each artist in the seed artist's Deezer
   neighbourhood, the 3 songs closest in sound are added, chosen from at most 150 of its catalog songs. They are scored
   like the sonic channel (window cosine minus the hub penalty) and gated by the catalog tiers.
2. **Listener model.** A gradient-boosted tree model estimates whether listeners of the seed's artist also play the
   candidate's artist. It has 100 trees of depth at most 6 (`ranker/listener-v1.npz`, sha256 `7ad156a6…24a7`).
   - Features: the served sound score; ranks and scores in the three channels; Deezer graph overlap, related-list
     direction and neighbourhood; region, genre and scene affinity; tempo gap; whether the language is known.
   - Not used: popularity, MuLan mood scores, release year.
   - Training: 434,481 rows from the 823 development seeds (scene `dev` and chart sweep). Labels are ListenBrainz
     similar artists, which are never a feature.
3. **Best matches.** The audio head is unchanged: at least 3 songs, extended over cover/sample evidence. The rest
   follow z(listener logit) + 0.5 z(sound score) + 0.25 z(melody/samples fusion score), then the artist caps.
4. **Sounds like.** The sonic list by z(sound score) + 2 z(listener logit).
5. **Same vibe.** Its rule is unchanged, applied to the new Best and Sounds like orders; scene candidates can enter.
6. **Own artist.** Songs by the seed's own artist get the median listener score, so sound alone places them.
7. **Fallback.** Without a known seed artist, or with `scene=0`, the lists are as before.

**Selection** on development data only, with out-of-fold models (5 folds grouped by seed artist), through the served
code path (`grow/qa/ranker_dev.py`):

| List | ListenBrainz hit rate | Shared genre | Far-genre rows |
|---|---|---|---|
| Best matches | 0.317 → 0.398 (+0.081) | +0.031 | −0.031 |
| Same vibe | 0.355 → 0.473 (+0.119) | +0.059 | −0.044 |
| Sounds like | 0.346 → 0.465 (+0.119) | +0.071 | −0.057 |

Best matches HR@10 on the development benchmarks changed by −0.009 on covers, −0.005 on samples and −0.007 on samples
reverse. Artist variety did not drop: the distinct-artist share of all top-10 rows went 0.590 → 0.590 (Best) and
0.577 → 0.554 (Sounds like). Sounds like's mean sound score of the top 10 went 0.453 → 0.441.

Variants were rejected for these reasons:
- **Popularity features** gained another 0.085, but halved the variety of artists (top-50 share 8 % → 16–20 %).
- **A one-song head** cost 0.061 on covers; **no head** cost 0.178.
- **Fusion weight 0** cost 0.025 on covers, too close to the limit. **Weight 0.5** gave no listener gain.
- **MuLan zero-shot mood scores** carried no information beyond the sound score (mood agreement was about +0.83 for
  nearly every sonic candidate).

**Baseline.** The served lists (`gated1m_v2`, computed on 2026-11-09 at 04:25 with the served code).

**Seeds.**
- Rules 1 and 3: the 799 held-out seeds (used five times before) and the 1,000 fresh seeds (used four times before).
- Rule 2: the relationship benchmarks' held-out split.

**Must pass (all), else not deployed:**
1. **Lists users see.** Best matches, Same vibe and Sounds like, each on both seed sets (`gate_eval compare --labels
   build1m`):
   - ListenBrainz hit rate: the paired 95 % interval's lower bound at −0.010 or above.
   - Shared genre: mean change −0.02 or above.
   - Other-language rows and derivative uploads: each up by at most 1 point.
   - Rows shown in the top 20: down by at most 1.0 on average.
2. **Relationship benchmarks.** Best matches HR@10 as served on the held-out split may drop by at most 0.03 on covers,
   samples and samples reverse. The channel lists (the any tab) are not touched by this change and are not
   re-measured.
3. **It must help, without narrowing.**
   - On both seed sets, the ListenBrainz hit rate of Best matches, Same vibe and Sounds like must rise, with the paired
     95 % interval's lower bound above 0.
   - For Best matches and Sounds like, on both sets: the distinct-artist share of all top-10 rows may fall by at most
     0.05, and the 50 most-recommended artists' share may rise by at most 0.03.
   - Sounds like's mean sound score of the top 10 may fall by at most 0.02.
4. **Serving on the RTX 2070 SUPER**, with automatic rollback:
   - Parity as amended (`rank_agree` 1.0, at least 62 of 64 first).
   - Startup under 2 minutes; peak VRAM at most 7,248 MiB.
   - 40 unseen songs and 40 catalog songs, measured in the same session on the same songs with cold result caches:
     the paired median server time may rise by at most 100 ms, and p95 may be at most the served p95 plus 200 ms.
     The added step took a median 16–31 ms and a p95 of 29–95 ms per seed on the RTX 5080; the 2070 host's CPU and
     storage are slower.

**Reported, not gated:**
- The ListenBrainz hit rate among artists Deezer does not list as related.
- Same vibe's sound score.
- Best matches HR@10 on the other benchmarks.

ListenBrainz's similar-artist data (labs endpoint) is probably an early-2023 snapshot, so newer artists cannot count as
hits.

### Result: rules 1 to 3 passed (2026-11-09)

Run once, after the pre-registration (4ca3c8c), with `ranker/listener-v1.npz`. All 49 checks passed
(`grow/qa/verdict_lr.py`). The benchmark step was interrupted once at 2,000 of 2,739 queries and re-run from the
start; it computes features only, so nothing was chosen in between.

Rules 1 and 3, top 10 with one song per artist (paired 95 % intervals):

| Seeds | List | ListenBrainz hit rate | Change | Shared genre | Rows in the top 20 |
|---|---|---|---|---|---|
| Held-out (799) | Best matches | 0.322 → 0.408 | +0.085 (+0.073 to +0.099) | +0.042 | 19.96 → 19.98 |
| | Same vibe | 0.341 → 0.451 | +0.110 (+0.096 to +0.124) | +0.079 | 19.93 → 19.96 |
| | Sounds like | 0.331 → 0.456 | +0.125 (+0.111 to +0.137) | +0.092 | 19.52 → 19.52 |
| Fresh (1,000) | Best matches | 0.315 → 0.406 | +0.091 (+0.079 to +0.103) | +0.024 | 19.93 → 20.00 |
| | Same vibe | 0.344 → 0.466 | +0.122 (+0.110 to +0.134) | +0.052 | 19.94 → 19.97 |
| | Sounds like | 0.331 → 0.454 | +0.123 (+0.111 to +0.134) | +0.066 | 19.67 → 19.67 |

Other-language rows stayed at zero. Derivative uploads changed by −0.0008 to +0.0002.

Rule 3, variety and sound:

| Seeds | List | Distinct-artist share of top-10 rows | Top-50 artists' share | Sound score of the top 10 |
|---|---|---|---|---|
| Held-out | Best matches | 0.641 → 0.679 | 0.066 → 0.052 | |
| | Sounds like | 0.634 → 0.652 | 0.059 → 0.054 | 0.450 → 0.438 (−0.013) |
| Fresh | Best matches | 0.602 → 0.625 | 0.070 → 0.061 | |
| | Sounds like | 0.598 → 0.597 | 0.068 → 0.061 | 0.454 → 0.442 (−0.012) |

Rule 2, Best matches HR@10 on the held-out split:

| Benchmark | Before | After | Limit |
|---|---|---|---|
| Covers (1,170 queries) | 0.662 | 0.650 (−0.012) | −0.03 |
| Samples (747) | 0.130 | 0.130 (0.000) | −0.03 |
| Samples reverse (695) | 0.137 | 0.138 (+0.001) | −0.03 |

**Reported, not gated.**
- **ListenBrainz hit rate among artists Deezer does not list as related to the seed.**

  | List | Held-out | Fresh |
  |---|---|---|
  | Best matches | 0.229 → 0.202, −0.027 (−0.041 to −0.013) | 0.224 → 0.206, −0.018 (−0.029 to −0.005) |
  | Same vibe | +0.068 | +0.086 |
  | Sounds like | +0.124 | +0.114 |

  Best's gain comes from placing more Deezer-related artists, which are mostly listener-similar. Among the others,
  fewer are. Deezer's related lists are themselves co-listening data, so part of Best's gain is two co-listening
  sources agreeing. Same vibe and Sounds like improved beyond the related artists too.
- Same vibe's sound score of the top 10: 0.432 → 0.416 (held-out), 0.435 → 0.419 (fresh).
- Best matches HR@10 on the other benchmarks:

  | Benchmark | Queries | Before | After |
  |---|---|---|---|
  | Gen Alpha | 59 | 0.356 | 0.288 |
  | SampleID | 43 | 0.186 | 0.186 |
  | V4 | 19 | 0.158 | 0.105 |

  The Gen Alpha drop is the largest regression in this release.
- Crawled songs in the top 10: 11 % to 16 % (7 % to 8 % before). They come only from the seed artist's own Deezer
  related list, as the catalog tiers allow.
- The added step took a median 17 ms and a p95 of 34 ms per held-out benchmark seed on the RTX 5080.

### Rule 4: deployed (2026-11-09)

The deploy (`deploy_lr.sh`) passed every check on the first attempt:

| Check | Result | Limit |
|---|---|---|
| Parity | `rank_agree` 1.0 in every channel; all 64 first in melody and samples, 63 in sound (the Clash remaster) | as amended |
| Startup | 82 s, catalog tiers and ranker active (52 to 57 s in earlier deploys) | under 120 s |
| 40 unseen songs | paired median server time +18 ms; p95 1.043 → 1.070 s | +100 ms; 1.243 s |
| 40 catalog songs | paired median server time +31 ms; p95 0.467 → 0.512 s | +100 ms; 0.667 s |
| Peak VRAM | 6,727 MiB, unchanged | 7,248 MiB |

On the RTX 2070 SUPER the listener step itself took a median 28 ms for unseen songs and 40 ms for catalog songs
(p95 35 and 61 ms). The searches did not slow down.

Checks through the public URL and the LAN:
- Roar "I Can't Handle Change": Sounds like opens with salvia palth, dandelion hands, Black Box Recorder, Elita and
  Joy Again, where the served list had Ty Segall, Cursive and Sufjan Stevens' "Come on! Let's Boogey to the Elf Dance".
- Nemzzz "8PM": Sounds like holds only UK artists in its first eight.

**Found after deploying, not covered by the gates.** The gates read one song per artist. Best matches now places an
artist's two songs (the per-artist cap) next to each other more often:

| Best matches, raw top 10 | Before | After |
|---|---|---|
| Distinct artists | 9.10 (held-out), 9.13 (fresh) | 8.52, 8.59 |
| Back-to-back pairs by one artist, per list | 0.26 | 0.59 (held-out), 0.57 (fresh) |

Same vibe and Sounds like hold one song per artist and are unaffected.

## Spacing an artist's songs in Best matches (pre-registered 2026-11-09, before any held-out result)

**Why.** See above: with the listener ranker, Best matches often places an artist's two songs back to back.

**Change under test** (`scene.arrange_artists`, used only with the listener ranker). The audio head and the
per-artist caps are unchanged. After the head:
- No artist appears twice in a row (`BEST_ARTIST_GAP` 2).
- Each artist appears at most once in the first 10 places (`BEST_ARTIST_ONCE` 10). Head songs count.

An entry that breaks a rule waits for the first place where it does not, ahead of later entries. Every artist's
first song therefore keeps its order. With one song per artist, Best's top 10 is the same as before, or extended
where the old list had fewer than 10 artists in its top 20.

**Selection** on the development seeds, as for the ranker (`ranker_dev.py`, out-of-fold):

| Variant | Distinct artists in Best's top 10 as shown | Back-to-back pairs in the top 10, per list |
|---|---|---|
| Served (caps only) | 8.57 | 0.60 |
| Never twice in a row | 8.62 | 0.03 |
| At least 3 places apart | 8.69 | 0.03 |
| No back-to-back and once in the top 10 (chosen) | 9.89 | 0.03 |

- Every variant left the one-per-artist ListenBrainz, genre and sound metrics unchanged.
- Best HR@10 on the development benchmarks moved by at most +0.001.
- The remaining 0.03 pairs sit inside the audio head.

**Baseline.** The served lists (`gated1m_lr`). **Seeds.** The 799 held-out and 1,000 fresh seeds; the benchmarks'
held-out split.

**Must pass (all), else not deployed:**
1. **Lists users see.** On both seed sets:
   - Every seed's one-per-artist Best top 10 equals the served one, or extends it.
   - Sounds like is identical.
   - Same vibe: the ListenBrainz hit rate's paired 95 % lower bound is −0.010 or above. Rows shown in its top 20 fall
     by at most 1.0.
2. **Relationship benchmarks.** Best matches HR@10 on held-out covers, samples and samples reverse falls by at most
   0.01 each against the served ranker.
3. **It must help.** On both seed sets, back-to-back pairs by one artist in Best's top 10 as shown are at most 0.10
   per list, and the number of distinct artists there rises.
4. **Serving.** Rule 4 of the listener ranking, unchanged:
   - parity as amended;
   - startup under 2 minutes; peak VRAM at most 7,248 MiB;
   - 40 unseen and 40 catalog songs in one session: paired median server time at most +100 ms, p95 at most the served
     p95 plus 200 ms;
   - automatic rollback.

## Serving on the RTX 2070 SUPER

The served API runs on an 8 GB RTX 2070 SUPER in the homelab. `serve_check` and `robustness` ran there against the
full index before it went live.

- **Fit.** 5.4 GB of VRAM after loading models and index, 6.3 GB peak while serving. The process uses about
  1.5 GB of RAM. The rest of its 6 GB memory cap is page cache for the memory-mapped full-dimensional windows
  (13.6 GB on disk), which the kernel trims as needed. Startup takes 21.5 s: models 15 s, index 5 s, warm-up
  1.2 s.
- **Parity.** 64 indexed tracks per channel were re-embedded live on the 2070. Window cosine to the offline vector
  was at least 0.999999, and every track ranked itself first.
- **Unseen queries.** All 5,348 benchmark query previews were embedded live and searched as if new. Every
  benchmark's live AP and hit rates matched the offline numbers within 0.0001, and the top-10 overlap with the
  offline ranking was at least 0.995. A song nobody has indexed gets the same quality as a catalog song.
- **Latency.** Per uncached query, serially: decode 71 ms; embeddings 41 ms melody, 82 ms samples, 157 ms
  sound; search 37, 109 and 20 ms; 517 ms in total (p95 555 ms). Through the API, a catalog song answers in
  about 170 ms and an unseen song in 610–670 ms.

### Compressed stage 1 (2026-11-07)

A catalog of about a million tracks does not fit the 2070's stage 1 in fp16. The API process holds 6,162 MiB, of
which the 269k projections are 2.19 GB, so models and working memory take about 3.9 GB of the 8 GB. At the default
dimensions, fp16 projections need 8.45 GB per million tracks. The test: 300 random catalog tracks per channel were
searched with a compressed stage 1 and the live re-rank (default candidate depth, hub penalty as served). Each
served top-k was compared with an exact brute-force ranking of all 269,278 tracks; the numbers are the share of the
exact top-k found.

| channel | stage 1 | GB per 1M tracks | top 10 | top 20 | top 100 | worst query, top 10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| melody | fp16, 256 dims (live) | 1.54 | 1.000 | 1.000 | 1.000 | 1.0 |
| melody | int8, per-dimension scale, clipped at the 99.99th percentile | 0.77 | 0.997 | 0.997 | 0.999 | **0.0** |
| melody | int8, per-dimension scale, no clipping | 0.77 | 1.000 | 1.000 | 0.997 | 1.0 |
| melody | int8, per-window scale | 0.77 | 1.000 | 1.000 | 1.000 | 1.0 |
| samples | fp16, 384 dims (live) | 6.14 | 1.000 | 1.000 | 1.000 | 1.0 |
| samples | int8, 256 dims | 2.05 | 1.000 | 1.000 | 1.000 | 1.0 |
| samples | int4, 384 dims, per-window scale | 1.54 | 1.000 | 1.000 | 1.000 | 1.0 |
| samples | int8, 128 dims | 1.02 | 0.996 | 0.995 | 0.979 | 0.8 |
| sound | fp16, 128 dims (live) | 0.77 | 1.000 | 1.000 | 1.000 | 1.0 |
| sound | int8, per-window scale | 0.38 | 1.000 | 1.000 | 1.000 | 1.0 |
| sound | int4, per-dimension scale, clipped at the 99.9th percentile | 0.19 | 0.986 | 0.980 | 0.961 | 0.5 |

- **Clipping:** clipping the scale at a percentile lost one melody query's entire top 10. A few extreme melody tracks
  (the hub table's minimum is −16,705) carry their information in the outlying values.
- **Per-window scaling:** an unclipped per-dimension scale avoids that loss, but those same tracks stretch it and
  coarsen everyone else: 12 % relative error against 1.2 % with one scale per window.
- **4-bit:** fine for samples, whose 8×8 window pairs leave a wide margin. With a per-dimension scale it lost
  matches in sound, whose scores sit in a narrow band; sound stays at 8 bits (4-bit with a per-window scale was not
  tried there).
- **Served setting:** melody int8 at 256 dimensions, samples int4 at 384, sound int8 at 128, built by `index_build
  --proj-bits` and served by `TwoStageIndex`:
  - 2.69 GB per million tracks, plus 0.06 GB of window scales.
  - Through that code, the top 10 and top 20 were identical to brute force for all 300 queries in every channel.
  - At top 100, one melody query and one sound query differed by one song. The difference was the same at double
    the candidate depth, so it was not a missed candidate.
- **Caveat:** this was measured on 269k tracks, where neighbourhoods are sparser than in a bigger catalog. Repeat
  the comparison at the new size before serving it.

### Robustness

200 random catalog previews, each edited and embedded live like an unseen query. A hit means the original song
ranks first. One track's preview is only 1 s long, so its 10 s excerpt is empty and it was skipped (n = 199).

| edit | melody HR@1 / @10 | samples HR@1 / @10 | sound HR@1 / @10 |
| --- | --- | --- | --- |
| unchanged | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| sped up 1.25× | 0.59 / 0.74 | 1.00 / 1.00 | 0.12 / 0.26 |
| slowed 0.85× | 0.66 / 0.78 | 1.00 / 1.00 | 0.22 / 0.37 |
| +2 semitones | 0.76 / 0.90 | 1.00 / 1.00 | 0.23 / 0.44 |
| −2 semitones | 0.78 / 0.88 | 1.00 / 1.00 | 0.29 / 0.46 |
| 10% faster, same pitch | 1.00 / 1.00 | 0.99 / 1.00 | 0.25 / 0.45 |
| noise at 10 dB SNR | 1.00 / 1.00 | 1.00 / 1.00 | 0.35 / 0.52 |
| phone band | 0.87 / 0.94 | 0.81 / 0.92 | 0.01 / 0.05 |
| 10 s excerpt | 0.94 / 0.97 | 1.00 / 1.00 | 1.00 / 1.00 |

The samples channel finds sped-up, slowed and pitched versions almost perfectly, so nightcore and slowed edits
land on the original there. Melody tolerates tempo and noise but
loses some ground when the pitch or speed shifts. The sound channel describes the recording rather than the song:
any change to speed, pitch or bandwidth moves it, and a phone-quality recording is unrecognisable to it.

### Known pairs

24 famous relationships, run through the live API before (25k research index) and after (full index). Ranks are
the best position of an expected song in any tab's top 50 (song view); "–" means not in the top 50. Every expected
song is in the full index, so each miss is a model miss, not a catalog gap. The two new hits are songs that were
not in the 25k index. "Live" queries were embedded on the fly because the query song is not in the catalog.

| query → expected | 25k | 271k | tab |
| --- | --- | --- | --- |
| Blurred Lines → Got To Give It Up | 1 | 2 | samples |
| Got To Give It Up → Blurred Lines | 1 | 5 | samples |
| Cupid (Twin Ver.) → Gen Alpha group | 1 | 3 | melody |
| One Of Your Girls → Gen Alpha group | 3 | 4 | melody |
| APT. → Gen Alpha group | 19 | – | melody |
| Mo Money Mo Problems → I'm Coming Out | 4 | 45 | samples |
| Viva La Vida → If I Could Fly | 35 | – | melody |
| Walk This Way (Run-DMC, live) → Aerosmith | – | 3 | samples |
| Under Pressure (live) → My Chemical Romance cover | – | 1 | melody |

The other 15 pairs were missed by both indexes:
- Every Breath You Take ↔ I'll Be Missing You;
- Stronger ↔ Harder, Better, Faster, Stronger;
- Super Freaky Girl ↔ Super Freak;
- Old Town Road → 34 Ghosts IV;
- Shape of You → No Scrubs;
- Stay With Me → I Won't Back Down;
- Thinking Out Loud → Let's Get It On;
- the live queries Gangsta's Paradise, I Got a Woman, Ice Ice Baby and Why Can't We Live Together;
- Johnny Cash's Hurt → the Nine Inch Nails original.

Top-1 hits went from 3 to 1, top-10 hits from 5 to 6, and top-50 hits stayed at 7. That matches the benchmarks: with
11 times more songs, true matches slip a few places, and newly indexed songs let two new pairs land. Most
famous interpolations and lawsuits remain out of reach. Likely reasons, not checked case by case: the shared hook
falls outside one of the two 30-second previews, or the resemblance is a groove rather than shared audio.

With the [scene layer](#scene-layer-2026-11-04), the any-tab counts are unchanged (1, 6 and 7). Best's own top 10
holds the expected song for 5 of the 24 pairs, against 6 without the layer:

- **Got To Give It Up → Blurred Lines** falls from #10 to #19 (#5 in the samples tab).
- **The other pairs keep their places.** The first release, before the top-3 rule, had also dropped Cupid's Gen
  Alpha match from #5 to #11.

## Limitations

- **Previews only.** Each song is heard through a 30-second Deezer preview. Shared material elsewhere in the song
  is invisible.
- **Indexed catalog only.** Results can only be songs in the index: 269k songs until 2026-11-08, 889k on 2026-11-08,
  999k since 2026-11-09, with crawled songs limited to the seed's scene. Any song can be the query.
- **Small sets.** v4 has 19 held-out and 35 validation queries, so one query moves HR by 3–5 points. Gen Alpha is a
  single group of 65 songs, and its members are not independent queries.
- **Gen Alpha was seen once.** It was looked at when choosing the melody hub setting (see
  [Hub penalty](#hub-penalty)), so it is not fully blind for that one decision.
- **Known pairs are demonstrations.** They were picked by hand to be recognisable, not sampled.
- **Scene metrics are partly circular.** ListenBrainz and Deezer's related lists both come from listening data, so
  agreement partly re-measures the feature; see [Scene layer](#scene-layer-2026-11-04).
