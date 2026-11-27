import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "webapp"
# /anysong is now a redirect stub; the any-song search runs on the home page.
PAGE = (WEB / "anysong" / "index.html").read_text(encoding="utf-8")
HOME = (WEB / "index.html").read_text(encoding="utf-8")
SCRIPT = (WEB / "search.js").read_text(encoding="utf-8")


def _config():
    config = json.loads((WEB / "site.json").read_text(encoding="utf-8"))
    rewrites = {r["source"]: r["destination"] for r in config["rewrites"]}
    redirects = {r["source"]: r for r in config.get("redirects", [])}
    routes = {
        item["source"]: {h["key"]: h["value"] for h in item["headers"]}
        for item in config["headers"]
    }
    return rewrites, redirects, routes


def test_anysong_redirects_home_and_keeps_strict_csp():
    rewrites, redirects, routes = _config()
    assert "/anysong" not in rewrites and "/anysong/" not in rewrites
    for source in ("/anysong", "/anysong/"):
        assert redirects[source] == {"source": source, "destination": "/", "permanent": False}
    # The stub page keeps the old page's strict policy.
    assert routes["/anysong"] == routes["/anysong/(.*)"]
    csp = dict(
        (d.split(" ", 1) + [""])[:2] for d in (p.strip() for p in routes["/anysong"]["Content-Security-Policy"].split(";"))
    )
    assert csp["default-src"] == "'none'"
    assert csp["script-src"] == "'unsafe-inline'"
    assert csp["connect-src"] == "'self'"
    assert csp["frame-ancestors"] == "'none'"
    assert csp["base-uri"] == "'none'"
    assert csp["form-action"] == "'none'"


def test_anysong_stub_forwards_old_links_to_the_home_page():
    # Hosts that ignore site.json redirects still serve this file, so it
    # forwards ?id=<id>&tab=<tab> and #id=<id>&tab=<tab> links itself.
    script = '<script>location.replace("/" + location.search + location.hash);</script>'
    assert script in PAGE
    assert '<noscript><meta http-equiv="refresh" content="0; url=/"></noscript>' in PAGE
    assert PAGE.index(script) < PAGE.index("<noscript>")
    assert '<meta name="robots" content="noindex">' in PAGE
    assert '<link rel="canonical" href="https://soundalike.yassin.app/">' in PAGE
    assert '<a href="/">soundalike home page</a>' in PAGE
    assert not re.search(r"<script[^>]+src=", PAGE, re.I)
    assert "fetch(" not in PAGE


def test_home_page_search_uses_same_origin_any_song_api():
    assert '<script src="/search.js"></script>' in HOME
    assert "`/api/anysong/search?limit=8&q=${encodeURIComponent(value)}`" in SCRIPT
    assert "fetch(`/api/anysong/similar?${params}`" in SCRIPT
    assert "/api/preview?id=" in SCRIPT
    assert re.search(r"fetch\(\s*[`'\"]https?:", SCRIPT) is None
    assert (
        'parsed.hostname === "dzcdn.net" || parsed.hostname.endsWith(".dzcdn.net")'
    ) in SCRIPT
    for legacy in ("/api/search?", "/api/recommend", "/api/stats", "spicetify_recommend"):
        assert legacy not in SCRIPT
        assert legacy not in HOME


def test_main_page_no_longer_links_to_anysong():
    assert 'href="/anysong"' not in HOME
    assert "<strong>Try any song</strong>" not in HOME
    assert ".promos{display:grid;grid-template-columns:repeat(2," in HOME


def test_home_page_footer_links_evaluation_in_new_tab_safely():
    link = re.search(r'<a href="([^"]+ANYSONG_EVAL\.md)"([^>]*)>', HOME)
    assert link
    assert link.group(1).startswith("https://github.com/yassinsolim/soundalike/")
    assert 'target="_blank"' in link.group(2)
    assert 'rel="noopener"' in link.group(2)


def test_anysong_language_switch_covers_every_tab():
    assert 'params.set("lang", lang);' in SCRIPT
    assert '<div class="language-filter" id="language" role="status">' in SCRIPT
    assert '<button id="language-toggle" type="button" aria-pressed="false" hidden>' in SCRIPT
    assert '"Show all languages"' in SCRIPT and '"Show same language"' in SCRIPT
    # display:flex would otherwise beat the hidden attribute.
    assert ".language-filter[hidden]{display:none}" in HOME
    # The server's hidden count is for the fused list, not main or Listeners also play, so no tab shows a count:
    # they only say what they leave out.
    assert '"all languages" : "other languages left out"' in SCRIPT
    assert "Number(query.hidden)" not in SCRIPT and "other-language matches in the top 20" not in SCRIPT
    assert "hidden_sounds" not in SCRIPT


def test_anysong_best_caption_puts_covers_first_then_the_model():
    assert (
        "\"Likely covers, samples and shared melodies first, then other artists' songs a model ranks closest, "
        "from five signals: sound, scene, playlists, melody and samples. "
        'Turn on Audio only to rank by sound alone."'
    ) in SCRIPT
    # Servers whose main list follows sound and the listener ranker.
    assert (
        '"Likely covers, samples and shared melodies first, then the songs that sound most alike, '
        "nudged toward artists this song's listeners also play. "
        'Turn on Audio only to rank by sound alone."'
    ) in SCRIPT
    # Servers without the main list send Best matches with listener picks in it.
    assert (
        '"Likely covers, samples and shared melodies first, then similar-sounding songs '
        "by artists this song's listeners also play, marked Listeners also play. "
        'Turn on Audio only to leave those out."'
    ) in SCRIPT
    # Audio only leaves the listeners out, so its caption does too.
    assert (
        '"Likely covers, samples and shared melodies first, then the songs that sound most alike."'
    ) in SCRIPT


def test_anysong_has_the_spotify_extensions_three_tabs():
    keys = re.findall(r'key: "(\w+)",\s*label: "([^"]+)"', SCRIPT)
    assert keys == [("best", "Best matches"), ("listeners", "Listeners also play"), ("melody", "Melody match")]
    # Best matches read the main list, or Best matches from a server without it; Melody match is the "best" list.
    assert 'if (key === "melody") return "best";' in SCRIPT
    assert 'Array.isArray(results.main) && results.main.length ? "main" : "best"' in SCRIPT
    # Melody match says how the list is built, and with Audio only that it is audio alone.
    assert (
        '"Likely covers, samples and shared melodies first, from the melody and sample matches fused into one '
        "ranking, then similar-sounding songs by artists this song's listeners also play, marked Listeners also "
        'play. Turn on Audio only to leave those out."'
    ) in SCRIPT
    assert (
        '"Likely covers, samples and shared melodies: the melody and sample matches fused into one ranking, '
        'by audio alone."'
    ) in SCRIPT
    # Links to the dropped tabs, and unknown tabs, open Best matches; a tab the page has opens itself.
    assert (
        "const tabKey = key =>\n"
        '    key === null ? null : resultTabs.some(entry => entry.key === key) ? key : "best";'
    ) in SCRIPT.replace("\r\n", "\n")
    # The tab chosen last is kept for the next visit.
    assert 'const tabStoreKey = "soundalike:tab:v1";' in SCRIPT
    assert "if (updateHash) saveTab(key);" in SCRIPT
    # A new policy keeps cached answers from before the consensus body out.
    assert 'const bestMatchesPolicy = "main-consensus-v4";' in SCRIPT
    assert "best-vibe-sounds" not in SCRIPT
    # Audio only and artists without listener data say why Listeners also play is empty.
    assert (
        "\"Audio only is on, and this list comes from what this song's listeners also play. "
        'Turn Audio only off to see it."'
    ) in SCRIPT
    assert (
        "\"We don't know what this artist's listeners play yet, so there's no list here. "
        "Best matches still has this song's closest matches.\""
    ) in SCRIPT


def test_match_a_part_is_offered_only_by_servers_that_match_parts():
    # The control reads query.parts from the normal answer: no request of its own, hidden without it.
    assert "const parts = ((picked.body || {}).query || {}).parts;" in SCRIPT
    assert "healthz" not in SCRIPT
    assert 'if (part) params.set("part", partText(part));' in SCRIPT
    # A server without it answers for the whole preview (or refuses the span and is asked again without it).
    assert "if (!(body.query || {}).part) picked.part = null;" in SCRIPT
    assert 'if (params.has("part") && (response.status === 400 || response.status === 404)) {' in SCRIPT
    # The part is in the share link, and a newly picked song starts from its whole preview.
    assert "const hash = `#id=${id}&tab=${tab}${part ? `&part=${partText(part)}` : \"\"}`;" in SCRIPT
    assert "if (!writeHash(id, true)) load({ id });" in SCRIPT
    # The copy is short, marked beta and claims nothing about quality.
    assert '"Soundalike hears a 30-second preview. Pick the part of it you care about."' in SCRIPT
    assert '<span class="beta">New · beta</span>' in SCRIPT
    assert 'Match a part <span class="beta">beta</span>' in SCRIPT
    assert '"All three tabs follow the part. Listeners also play still starts from the whole song\'s listeners; ' \
        'only how close each song sounds follows the part."' in SCRIPT
    # Spans sit on a 5-second grid and run 10 to 20 seconds; the timeline works with touch and the keyboard.
    assert "const partStepSeconds = 5;" in SCRIPT and "const partLongestSeconds = 20;" in SCRIPT
    assert "track.onpointerdown = onPartPointerDown;" in SCRIPT
    assert "touch-action:none" in HOME
    # display rules would otherwise beat the hidden attribute.
    assert ".part-chip[hidden]{display:none}" in HOME and ".part-head[hidden]{display:none}" in HOME
    # The waveform is read from Deezer's CDN only, like the covers.
    assert 'const url = safeImg((preview && preview.preview) || "");' in SCRIPT
