"""Focused homepage promotion and accessibility contracts."""
from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "webapp" / "index.html").read_text(encoding="utf-8")


class _ActionParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_actions = False
        self.actions_depth = 0
        self.current_link: dict[str, object] | None = None
        self.links: list[dict[str, object]] = []
        self.navigation_label: str | None = None
        self.hidden_depth = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = dict(attrs)
        if tag == "nav" and values.get("aria-label") == "Soundalike actions":
            self.in_actions = True
            self.actions_depth = 1
            self.navigation_label = values["aria-label"]
            return
        if not self.in_actions:
            return
        if tag == "nav":
            self.actions_depth += 1
        if values.get("aria-hidden") == "true":
            self.hidden_depth += 1
        if tag == "a":
            self.current_link = {"attrs": values, "text": []}

    def handle_endtag(self, tag: str) -> None:
        if not self.in_actions:
            return
        if tag == "a" and self.current_link is not None:
            self.links.append(self.current_link)
            self.current_link = None
        if self.hidden_depth and tag == "span":
            self.hidden_depth -= 1
        if tag == "nav":
            self.actions_depth -= 1
            if self.actions_depth == 0:
                self.in_actions = False

    def handle_data(self, data: str) -> None:
        if self.current_link is not None and self.hidden_depth == 0:
            self.current_link["text"].append(data)


def test_homepage_promotes_spicetify_and_evaluator_with_native_links():
    parser = _ActionParser()
    parser.feed(HTML)

    assert parser.navigation_label == "Soundalike actions"
    assert len(parser.links) == 2
    links = {
        " ".join(link["text"]).strip(): link["attrs"] for link in parser.links
    }
    spotify = next(
        attrs for text, attrs in links.items() if "Use inside Spotify" in text
    )
    evaluator = next(
        attrs for text, attrs in links.items() if "Help improve the model" in text
    )
    assert "Spicetify extension" in next(
        text for text in links if "Use inside Spotify" in text
    )
    assert spotify["href"].endswith("/integrations/spicetify")
    assert spotify["target"] == "_blank"
    assert "noopener" in spotify["rel"].split()
    assert evaluator["href"] == "/evaluate"


def test_homepage_search_accessibility_and_wiring_remain_present():
    assert '<main class="wrap">' in HTML
    assert "</main>" in HTML
    assert (
        'id="q" aria-label="Search for a song" role="combobox" '
        'aria-autocomplete="list" aria-controls="ac" aria-expanded="false"'
    ) in HTML
    assert 'id="ac" role="listbox" aria-label="Song suggestions"' in HTML
    assert '<div id="out" aria-live="polite"></div>' in HTML
    assert '<script src="/search.js"></script>' in HTML
    assert '$("#q").addEventListener("input",onType)' in HTML
    assert 'fetch("/api/search?q="' in HTML
    assert 'fetch("/api/recommend"' in HTML


def test_marketplace_description_pins_the_immutable_update_bootstrap():
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["description"] == (
        "Find same-language audio matches in Spotify with native playback and "
        "optional anonymous feedback."
    )
    assert manifest["main"] == "integrations/spicetify/bootstrap.js"
    assert manifest["branch"] == "38ca29ca9ec760dc40e58a567ca6aaff632ae306"
