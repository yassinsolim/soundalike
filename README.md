# Soundalike

Find songs that sound like a track you already enjoy.

[Try the web app](https://soundalike.yassin.app) |
[Install it in Spotify](integrations/spicetify/README.md) |
[Help evaluate matches](https://soundalike.yassin.app/evaluate) |
[Report a bug](https://github.com/yassinsolim/soundalike/issues/new?labels=bug&title=Bug%3A%20) |
[Request a feature](https://github.com/yassinsolim/soundalike/issues/new?labels=enhancement&title=Feature%20request%3A%20)

I started Soundalike as a small university project and kept rebuilding it as
Spotify's APIs changed and the catalog grew. Today it is an open-source music
recommender built around audio similarity. It compares precomputed audio
representations and measured features such as tempo, dynamics, and bass
balance. It does not use your listening history as a ranking signal.

The public web app and Spicetify extension use a catalog of 272,853 tracks.
The repository also includes a local app, command-line tools, smaller bundled
indexes, and the research code used to test new ranking ideas.

I am still improving the recommendations. If you try it, specific examples of
great or poor matches are always useful.

![Soundalike results inside Spotify](docs/spicetify-results.png)

## Choose how you want to use it

### Inside Spotify

This is the easiest way to use Soundalike regularly.

1. Install [Spicetify](https://spicetify.app/docs/getting-started/).
2. Open Marketplace in Spotify.
3. Search for **Soundalike** under Extensions and install it.
4. Restart Spotify if Marketplace asks you to.
5. Right-click a track and select **Find soundalikes**.

Results open on their own Spotify page. You can play a result, open its album,
use Spotify's normal track menu, or search for another set of matches.

Marketplace installs receive signed runtime updates when Spotify starts. The
loader verifies the release signature, immutable commit URL, SHA-256 hash, and
browser integrity value before it runs an update. If verification or download
fails, it uses the last verified version.

If you installed Soundalike before the signed updater was introduced, or if
**Find soundalikes** is missing, remove the extension from Marketplace, install
it once more, and restart Spotify. Later runtime updates should not require
another reinstall.

See the [Spicetify guide](integrations/spicetify/README.md) for privacy details,
manual installation, supported Spotify builds, and troubleshooting.

### In a browser

Open <https://soundalike.yassin.app>.

You do not need an account to search the hosted catalog. Spotify login is
optional and is only used when you ask the site to save a playlist. Spotify
handles the authorization through OAuth; Soundalike never receives your
password.

![Soundalike web app](docs/soundalike-results.png)

### As a local app or command-line tool

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/yassinsolim/soundalike.git
cd soundalike
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or activate it on macOS and Linux:

```bash
source .venv/bin/activate
```

Install the basic command-line tools:

```bash
python -m pip install -e .
soundalike similar --title "Blinding Lights" -n 10
```

Install the audio and model dependencies for the local web app:

```bash
python -m pip install -e ".[ml]"
soundalike fetch-index
soundalike serve
```

The local app opens at <http://127.0.0.1:8787>. It accepts a title and artist
or a Deezer track link. Spotify track links work after you complete the Spotify
setup.

For commands that read your Spotify library or use Last.fm, follow
[SETUP.md](SETUP.md).

## What the Spotify extension does

The extension keeps the model ranking and Spotify-specific checks separate:

1. It sends the selected track title and artist to the Soundalike service.
2. The service retrieves candidates from the audio index and applies the
   production ranking policy.
3. Spotify resolves the returned tracks inside the desktop client.
4. If Spotify has a known lyrics language for the seed, candidates are shown
   only after the same language is confirmed.

The visible ranking includes a few practical safeguards:

- Audio similarity defines the candidate pool.
- Popularity signals cannot pull a distant track into that pool.
- Large tempo and dynamics mismatches receive a small penalty.
- Remix, live, acoustic, and other version families are collapsed so one
  recording does not occupy several visible slots.
- A different song by the same artist is still allowed when it ranks well.
- Low-confidence Spotify matches are never played automatically.

Strict language filtering can return fewer than 20 songs. A missing lyrics
language is treated as unknown, not as proof that a song is in another
language.

The extension caches successful recommendations and resolved Spotify metadata
for seven days. Failed language checks and unresolved Spotify searches are not
cached, so temporary failures can recover.

## Privacy

For a hosted recommendation, Soundalike receives:

- the seed title and artist;
- the requested result count and ranking options.

The extension does not send your Spotify password, access token, library,
listening history, or artwork. Lyrics-language and related-artist lookups happen
inside your authenticated Spotify client.

After results settle, the extension may show a short **Good**, **Mixed**, or
**Off** survey. Nothing is submitted until you press **Send feedback**. A
feedback record contains the seed, the rows you were shown, model policy
labels, your selected reasons, an optional note, and random anonymous
deduplication values. Do not put personal information in the note.

The feedback endpoint does not receive your Spotify identity, credentials,
tokens, library, listening history, hidden candidates, or a private storage
URL.

## How recommendations are built

The production ranker uses two compact audio representations:

- an EfficientNet-derived representation for timbre and texture;
- a calibrated CLAP representation for broader acoustic similarity.

The system combines them, limits contextual reordering to the top 1,000
audio-qualified candidates, applies quality and pacing safeguards, and returns
the final list. The hosted and local numpy ranking paths have parity tests to
keep their ordering consistent.

The current production method is `dual_sonic64_guardrail` with ranking policy
`model-quality-v1`. On the frozen 20-pair pure-sonic benchmark, Recall@50 is
0.10 compared with 0.05 for the earlier production baseline. That benchmark is
small and should not be read as proof that every listener will prefer every
result. Listening feedback remains the more useful signal for subjective
quality.

For the full engineering history, failed experiments, benchmark design, and
resource measurements, read:

- [Case study](docs/CASE_STUDY.md)
- [V3 research log](docs/V3_RESEARCH.md)
- [Full-track research notes](docs/FULLTRACK_AUDIO.md)

The full-track MTG-Jamendo work is research tooling. It is not used to acquire
Spotify audio, and it is not presented as the production commercial-catalog
model.

## Known limitations

- The hosted catalog is large but finite. A Spotify track can exist without
  being present in Soundalike's audio index.
- Audio similarity is subjective. Some seeds work better than others.
- Strict language filtering may hide otherwise strong instrumental or
  no-lyrics candidates.
- The Vercel fallback may be slower on its first request after being idle.
- Spicetify cannot patch every Spotify distribution. In particular, Windows
  Store and Linux Snap builds are not supported by Spicetify.
- The basic content engine uses a much smaller bundled CSV and is separate from
  the hosted audio index.

If a result looks wrong, please report it. Specific examples are much easier to
fix than a general "recommendations are bad" report.

## Feedback, bugs, and feature requests

Before opening a new issue, check the
[existing issues](https://github.com/yassinsolim/soundalike/issues) in case
someone has already reported it.

- [Report a bug](https://github.com/yassinsolim/soundalike/issues/new?labels=bug&title=Bug%3A%20)
- [Request a feature](https://github.com/yassinsolim/soundalike/issues/new?labels=enhancement&title=Feature%20request%3A%20)
- [Ask a question](https://github.com/yassinsolim/soundalike/issues/new?labels=question&title=Question%3A%20)

For recommendation-quality reports, include:

- where you used Soundalike: Spotify, the hosted site, or the local app;
- the seed title and artist;
- the results that felt wrong;
- what seemed wrong, such as tempo, mood, vocals, language, or instrumentation;
- a screenshot if the problem is visual.

For software bugs, also include your operating system, Spotify and Spicetify
versions when relevant, the steps to reproduce the problem, and the complete
error message. Remove access tokens, credentials, and other personal
information before posting.

You can also help by completing a few comparisons in the public
[listening evaluator](https://soundalike.yassin.app/evaluate).

## Development

Install the development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the Python tests:

```bash
python -m pytest
```

Run the web and Spicetify tests:

```bash
npm --prefix webapp install
npm --prefix webapp test
```

Keep changes focused and add tests for behavior changes. For a large feature or
ranking experiment, open an issue first so the approach and evaluation plan can
be discussed before implementation.

The main directories are:

```text
src/soundalike/          Python package, CLI, audio tools, and ranking code
webapp/                  Hosted app, API handlers, evaluator, and Node tests
integrations/spicetify/  Spotify extension, signed updater, and setup guides
deploy/homelab/          Always-on service, monitoring, deployment, and rollback
benchmarks/              Frozen evaluation inputs
docs/                    Research notes, results, and the case study
tests/                   Python tests
```

## Catalog maintenance

The production index has no genre field, so the coverage audit uses reviewed
artist groups as coverage proxies. It does not claim to classify the catalog by
genre.

Audit a local index:

```bash
python -m soundalike.ml.coverage_audit \
  --index /path/to/deepvibe_index.npz \
  --output coverage-audit.json
```

Preview a bounded, targeted catalog crawl:

```bash
python -m soundalike.ml.grow_broad \
  --targeted-plan coverage-audit.json \
  --max-artists 10 \
  --max-tracks 40 \
  --max-api-calls 80 \
  --dry-run
```

Targeted mode does not add the broad default seeds and will not run without
finite artist, track, and API-call limits.

Deployment and rollback instructions are in [webapp/DEPLOY.md](webapp/DEPLOY.md).

## Project history

The repository began as `spotify_program.py`, a first-year university project
that printed statistics from a static CSV. That file remains in the repository
as a record of where the project started.

## License

Soundalike is available under the [MIT License](LICENSE).
