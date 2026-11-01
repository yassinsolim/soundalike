import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "webapp"
PAGE = (WEB / "anysong" / "index.html").read_text(encoding="utf-8")


def _headers():
    config = json.loads((WEB / "vercel.json").read_text(encoding="utf-8"))
    rewrites = {r["source"]: r["destination"] for r in config["rewrites"]}
    routes = {
        item["source"]: {h["key"]: h["value"] for h in item["headers"]}
        for item in config["headers"]
    }
    return rewrites, routes


def test_anysong_routes_and_strict_csp():
    rewrites, routes = _headers()
    assert rewrites["/anysong"] == rewrites["/anysong/"] == "/anysong/index.html"
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
    assert csp["img-src"] == "data: https://dzcdn.net https://*.dzcdn.net"
    assert csp["media-src"] == "https://dzcdn.net https://*.dzcdn.net"


def test_anysong_page_is_self_contained_and_uses_same_origin_api():
    assert not re.search(r"<script[^>]+src=", PAGE, re.I)
    assert not re.search(r"<link[^>]+stylesheet", PAGE, re.I)
    assert "/api/anysong/search?" in PAGE
    assert "/api/anysong/similar?" in PAGE
    assert "/api/preview?id=" in PAGE
    assert re.search(r"fetch\(\s*[`'\"]https?:", PAGE) is None
    assert 'u.hostname === "dzcdn.net" || u.hostname.endsWith(".dzcdn.net")' in PAGE


def test_main_page_links_to_anysong():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert 'href="/anysong"' in html
    assert ".promos{display:grid;grid-template-columns:repeat(3," in html


def test_anysong_footer_links_evaluation_in_new_tab_safely():
    link = re.search(r'<a href="([^"]+ANYSONG_EVAL\.md)"([^>]*)>', PAGE)
    assert link
    assert link.group(1).startswith("https://github.com/yassinsolim/soundalike/")
    assert 'target="_blank"' in link.group(2)
    assert 'rel="noopener"' in link.group(2)
