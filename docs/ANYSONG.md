# Any-song (beta)

`/anysong` on soundalike.yassin.app takes any Deezer track, including one that has never been indexed, and finds
catalog songs that share its **melody**, a **sample/loop** or its overall **sound**. Unseen songs are fetched
and embedded live on a GPU; everything else is a lookup.

| Channel | Model | Finds | Score |
| --- | --- | --- | --- |
| `melody` (Same melody) | CLEWS DVI | covers, interpolations, reused tunes | negative distance |
| `samples` (Samples & loops) | SampleID | shared samples, loops, chopped sections | cosine |
| `sonic` (Sounds like) | MuQ-MuLan | production, instrumentation, mood | cosine |

Scores are only comparable within one channel and one query; the page shows relative bars, not percentages.
Everything runs on 30-second Deezer previews, so a relationship outside the preview can be missed.

## Architecture

```
browser ── Cloudflare ── tunnel ── CT 408 Caddy :8080 ─┬─ /api/anysong/*  ─▶ CT 409 :8790  anysong.api (RTX 2070 SUPER)
                                                        └─ everything else ─▶ vercel_gateway (webapp/, /anysong page)
```

- [`webapp/anysong/index.html`](../webapp/anysong/index.html) is a self-contained page (inline CSS/JS, strict CSP in
  [`vercel.json`](../webapp/vercel.json): same-origin `connect-src`, images and audio from `*.dzcdn.net` only).
  Previews are resolved through the existing `/api/preview`.
- [`anysong/api.py`](../src/soundalike/anysong/api.py) serves `GET /api/anysong/search?q=&limit=` (Deezer search
  plus `in_catalog`), `GET /api/anysong/similar?id=&k=` and `GET /healthz`.
- [`anysong/serve.py`](../src/soundalike/anysong/serve.py) holds the two-stage index: a PCA projection of every
  window in VRAM ranks candidates, then the full-dimensional windows (memory-mapped `full.npy`) rerank them.
  A catalog song is queried with its stored windows (no audio work); an unseen song is downloaded, decoded and run
  through all three encoders, and the result is cached in memory.
- Rate limits are token buckets per client: `/similar` 20/min (burst 5), `/search` 60/min (burst 15). Cached
  `/similar` results are free. Clients are keyed by `--client-ip-header` (`CF-Connecting-IP`) with IPv6 grouped
  per /64; only enable that flag while the port is reachable solely through the proxy chain. Excess concurrent
  live work returns 503 (`--max-pending`), which the page retries with backoff.

Errors are `{ok:false, error}` with 400 (bad parameters), 404 (unknown track / no preview), 422 (undecodable
preview), 429 (rate limit), 502 (Deezer), 503 (busy) or 504 (timed out waiting for an identical request).

## Building an index

Extract window embeddings for the frozen catalog with
[`embed_run`](../src/soundalike/anysong/embed_run.py) (`clews_dvi`, `sampleid`, `muq_mulan`; shards are resumable
`part-NNNN.npz` files with `ids`, `track`, `win`), then:

```sh
python -m soundalike.anysong.index_build --emb emb_full --out index_full \
  --ids catalog_ids.txt --dims melody=256 samples=384 sonic=128 \
  --meta catalog_meta.npz --max-song-tracks 100
```

Each channel directory holds `basis.npy`, `proj.npy` (fp16), `full.npy`, `ids.npy`, `norms.npy` and `meta.json`
(retained PCA energy, candidate counts). `--max-song-tracks` leaves out any song key (as the API groups
results) with more than that many tracks. In the frozen catalog this removes only 2,031 *A State Of Trance*
radio-show segments, which otherwise surface as one "song" in many unrelated result lists (see
[ANYSONG_EVAL.md](ANYSONG_EVAL.md)). `meta.json` records what was dropped. Then add the hub penalty:

```sh
python -m soundalike.anysong.hub_build --index index_full --meta catalog_meta.npz
```

This writes `hub.npy` per channel and records `hub` in `meta.json`. Serving subtracts `alpha * h(track)`
([CSLS](https://arxiv.org/abs/1710.04087)), where `h` is the track's mean similarity to its 10 nearest other
songs. That stops a few "sounds like everything" tracks from crowding every result list. Defaults are `alpha =
0.5` for samples and sonic, and 0 (off) for melody, because a family of songs sharing one melody is a genuine
dense cluster that the penalty would push down. The reasons are in [ANYSONG_EVAL.md](ANYSONG_EVAL.md).
`--hub-alpha` on `api` and `eval_index` overrides the recorded value; 0 disables it.

Checks before serving an index:

- `python -m soundalike.anysong.eval_index --index index_full --meta catalog_meta.npz --bench bench --exact 256
  --hubs 2000 --out eval.json` scores every benchmark split per channel and as the page's "any tab". It also
  compares the two-stage ranking with brute force (`--exact`) and measures hubness (`--hubs`). Choose settings
  on `development` splits; report `held_out`/`test`/`validation`.
- `python -m soundalike.anysong.serve_check --index <dir> --previews <dir> --bench <dir> --out report.json`
  re-embeds a sample of indexed tracks live to check parity with the offline index, and reports VRAM, latency
  and benchmark recall.
- `python -m soundalike.anysong.robustness --n 200` takes catalog previews and edits them (sped up, slowed,
  ±2 semitones, tempo change, noise, phone band, a 10 s excerpt). It embeds each edit live, like an unseen
  query, and checks whether the original song still ranks first.

On the 8 GB 2070 SUPER, stop the API before running `serve_check` or `robustness`. The full 269k index fits:
5.4 GB of VRAM loaded, 6.3 GB peak. If a bigger index does not fit, lower `samples` to 256.

## Deployment (homelab)

GPU host CT 409 (`/opt/soundalike`): `src/`, `venv/`, checkpoints under `ckpt/`, upstream code under `ext/`,
offline Hugging Face cache in `hf/`, indexes under `index/` with `index/current` → the live index and
`index/catalog_meta.npz` for titles. The service is
[`deploy/homelab/soundalike-anysong.service`](../deploy/homelab/soundalike-anysong.service) (user `anysong`,
`MemoryMax=6G`, enabled at boot; startup loads and warms the models and index, about 22 s with the full 269k-track
index). The process itself uses about 1.5 GB of RAM; the rest of the cap is page cache for the memory-mapped
windows, which the kernel trims.

Swapping the index:

```sh
scp -r index_full root@ct409:/opt/soundalike/index/full-YYYYMMDD
scp index_full.sha256 root@ct409:/opt/soundalike/index/full-YYYYMMDD.sha256
cd /opt/soundalike/index/full-YYYYMMDD && sha256sum -c ../full-YYYYMMDD.sha256 && chmod -R go-w .
systemctl stop soundalike-anysong
ln -sfn full-YYYYMMDD /opt/soundalike/index/current
systemctl start soundalike-anysong && journalctl -u soundalike-anysong -f   # wait for "ready in ..."
curl -s 192.168.1.47:8790/healthz
```

Keep the previous index directory until the new one has been checked; rolling back is the same symlink swap.

Public routing lives on CT 408: `"proxy_paths": {"/api/anysong/*": "192.168.1.47:8790"}` in the soundalike entry of
`/etc/web/sites.json`, applied with `web-caddy-gen`. The API then listens on the LAN address
(`ANYSONG_HOST=192.168.1.47`), and the Proxmox firewall for CT 409 only admits port 8790 from CT 408, which keeps
`CF-Connecting-IP` trustworthy. The page itself ships with the normal `main` deploy.

## Limitations

- Coverage is limited to the frozen catalog (about 271k tracks with previews, 269k after dropping radio-show
  segments); famous sources missing from it cannot be returned, however good the match.
- Preview-only audio: shared material outside the 30-second clip is invisible.
- Benchmarks are small for some relations (see the held-out notes in the README); treat the channels as beta.
