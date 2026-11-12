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


def test_anysong_language_switch_covers_both_tabs():
    assert 'params.set("lang", lang);' in SCRIPT
    assert '<div class="language-filter" id="language" role="status">' in SCRIPT
    assert '<button id="language-toggle" type="button" aria-pressed="false" hidden>' in SCRIPT
    assert '"Show all languages"' in SCRIPT and '"Show same language"' in SCRIPT
    # display:flex would otherwise beat the hidden attribute.
    assert ".language-filter[hidden]{display:none}" in HOME
    # The server's hidden count is for the fused list, not main or Listeners also play, so neither tab shows a count:
    # both only say what they leave out.
    assert '"all languages" : "other languages left out"' in SCRIPT
    assert "Number(query.hidden)" not in SCRIPT and "other-language matches in the top 20" not in SCRIPT
    assert "hidden_sounds" not in SCRIPT


def test_anysong_best_caption_puts_covers_first_then_the_listeners():
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


def test_anysong_has_the_spotify_extensions_two_tabs():
    keys = re.findall(r'key: "(\w+)",\s*label: "([^"]+)"', SCRIPT)
    assert keys == [("best", "Best matches"), ("listeners", "Listeners also play")]
    # Best matches read the main list, or Best matches from a server without it.
    assert 'Array.isArray(results.main) && results.main.length ? "main" : "best"' in SCRIPT
    # Links to the old tabs, and unknown tabs, open Best matches.
    assert (
        'const tabKey = key => (key === null ? null : key === "listeners" ? "listeners" : "best");'
    ) in SCRIPT
    # A new policy keeps cached five-tab answers out of the two tabs.
    assert 'const bestMatchesPolicy = "main-listeners-v1";' in SCRIPT
    assert "best-vibe-sounds" not in SCRIPT
    # Audio only and artists without listener data say why Listeners also play is empty.
    assert (
        "\"Audio only is on, and this list comes from what this song's listeners also play. "
        'Turn Audio only off to see it."'
    ) in SCRIPT
    assert (
        "\"We don't know what this artist's listeners play yet, so there's no list here. "
        'Best matches still has the songs that sound most alike."'
    ) in SCRIPT
