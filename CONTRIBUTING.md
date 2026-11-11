# Contributing to Soundalike

Thanks for helping. Bug reports, bad-match examples, listening comparisons,
documentation fixes and code changes are all welcome.

## Ways to help

- **Report a bad match.** Use the
  [bad match form](https://github.com/yassinsolim/soundalike/issues/new?template=bad_match.yml).
  A seed song, the results that felt wrong and a sentence on why they felt
  wrong are the most useful feedback the project gets.
- **Report a bug** or **suggest a feature** with the other
  [issue forms](https://github.com/yassinsolim/soundalike/issues/new/choose).
- **Rate comparisons** in the
  [listening evaluator](https://soundalike.yassin.app/evaluate).
- **Open a pull request.** Small fixes can go straight to a PR. For larger
  changes, new features or ranking experiments, open an issue first so the
  approach can be agreed before you spend time on it.

Search the [existing issues](https://github.com/yassinsolim/soundalike/issues)
before opening a new one.

## Project layout

```text
src/soundalike/          Python package, CLI, audio tools and ranking code
webapp/                  Hosted app, API handlers, evaluator and Node tests
integrations/spicetify/  Spotify extension, signed updater and setup guides
mobile/                  iOS and Android companion app
deploy/homelab/          Always-on service, monitoring, deployment and rollback
benchmarks/              Frozen evaluation inputs
docs/                    Research notes, results and the case study
tests/                   Python tests
```

## Set up

You need Git, Python 3.9 or newer (CI uses 3.12) and Node.js 24.

```bash
git clone https://github.com/<your-username>/soundalike.git
cd soundalike
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
npm --prefix webapp ci
npm --prefix mobile ci       # only if you work on the mobile app
```

The `ml`, `anysong` and `fulltrack` extras in `pyproject.toml` are only needed
for model training and research scripts. Tests that need them skip themselves
when they are not installed.

## Run the checks

These are the same checks CI runs on every pull request:

```bash
python -m pytest                 # Python
npm --prefix webapp test         # web app, API handlers and Spicetify extension
npm --prefix mobile run typecheck
npm --prefix mobile test         # mobile app
```

To try the hosted web app locally, run `python webapp/dev_server.py` and open
<http://127.0.0.1:8788/>.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Keep each pull request to one change. Unrelated fixes belong in separate PRs.
3. Add or update tests when behaviour changes.
4. Run the checks above.
5. Open the pull request and fill in the template. Link the issue it closes,
   for example `Closes #12`.

CI must pass before a pull request can be merged. The maintainer reviews every
PR and squash-merges it, so the PR title becomes the commit message on `main`.
Write it in the imperative, for example "Add a skip button to results".

First-time contributors' workflow runs wait for maintainer approval. That is
normal and only needs to happen once.

## Area notes

### Spicetify extension

The extension is `integrations/spicetify/soundalike.js`. To test a change in
Spotify, install your copy manually as described in
[integrations/spicetify/SETUP.md](integrations/spicetify/SETUP.md). Disable the
Marketplace copy while testing so you know which version is running.

Do not change these in a pull request; the maintainer handles them when
publishing a signed release:

- `RUNTIME_SEMANTIC_VERSION` in `soundalike.js`;
- `integrations/spicetify/releases/stable.json`;
- `integrations/spicetify/bootstrap.js` and the commit pinned in `manifest.json`.

Merged extension changes reach users with the next signed release. The process
is in [integrations/spicetify/RELEASE.md](integrations/spicetify/RELEASE.md).

Include before and after screenshots for visual changes.

### Ranking and recommendation changes

Listening examples are welcome, but a change to how results are ranked needs
measured evidence as well:

- Tune on development data only. Use the held-out split once, to confirm the
  final setting, and say how many configurations you tried.
- Report the benchmark, the metric and the change against the current setting,
  with a confidence interval or a paired comparison where possible.
- Include examples that got worse as well as examples that got better.

[docs/ANYSONG_EVAL.md](docs/ANYSONG_EVAL.md) describes the benchmarks, the
metrics and how earlier changes were tested.

### Web app and API

The hosted app and its API handlers live in `webapp/`. Do not change response
formats that the Spicetify extension or the mobile app depend on without
updating them in the same pull request.

## Security and privacy

- Never commit secrets: `.env` files, API keys, tokens, webhook URLs or
  signing keys. Use `.env.example` for new settings.
- Remove access tokens and personal information from logs and screenshots
  before posting them.
- Report security problems privately through
  [GitHub's private vulnerability reporting](https://github.com/yassinsolim/soundalike/security/advisories/new),
  not in a public issue.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE).
