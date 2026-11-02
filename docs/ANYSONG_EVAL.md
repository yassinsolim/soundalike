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

### What did not help: query expansion

Query expansion re-scores each candidate by mixing its similarity to the query with its average similarity to the
query's top n results (n ∈ {1, 2, 3, 5, 8}, mixing weight λ ∈ {0.5, 1, 2}, 20 variants per channel). On development
splits it added at most 0.005 to melody covers AP over the hub-penalised baseline, and every variant lowered HR@10 (by
0.003 to 0.040). Every samples variant lowered both AP and HR@10. For sound, AP rose by up to 0.026 with HR@10
unchanged, measured on only 31 queries. It is not used.

The tooling for these studies (re-ranking, fusion sweep and paired bootstrap) lives outside the package; `eval_index`
reports the shipped `best` list directly.

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

## Limitations

- **Previews only.** Each song is heard through a 30-second Deezer preview. Shared material elsewhere in the song
  is invisible.
- **Frozen catalog.** Results can only be songs in the 269k index. Any song can be the query.
- **Small sets.** v4 has 19 held-out and 35 validation queries, so one query moves HR by 3–5 points. Gen Alpha is a
  single group of 65 songs, and its members are not independent queries.
- **Gen Alpha was seen once.** It was looked at when choosing the melody hub setting (see
  [Hub penalty](#hub-penalty)), so it is not fully blind for that one decision.
- **Known pairs are demonstrations.** They were picked by hand to be recognisable, not sampled.
