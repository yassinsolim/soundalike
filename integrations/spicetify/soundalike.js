// soundalike — Spicetify extension
// Adds a right-click "Find soundalikes" item to any track in the Spotify
// desktop client. It asks the local soundalike server (run `soundalike serve`)
// for songs that *sound* like the one you clicked, and shows them in a modal.
//
// Install (requires the standalone Spotify from spotify.com — the Microsoft
// Store build cannot be patched by Spicetify):
//   1. spicetify config-dir            # find your config folder
//   2. copy soundalike.js into  <config>/Extensions/
//   3. spicetify config extensions soundalike.js
//   4. spicetify apply
//   5. run `soundalike serve` in a terminal, then right-click any song.

(function soundalike() {
  const SERVER = "http://127.0.0.1:8787";

  // Wait until the Spicetify APIs we need are ready.
  if (!(window.Spicetify && Spicetify.ContextMenu && Spicetify.Platform && Spicetify.URI)) {
    setTimeout(soundalike, 400);
    return;
  }

  const onlyTracks = (uris) =>
    Array.isArray(uris) && uris.length === 1 && uris[0].includes(":track:");

  async function findSoundalikes(uris) {
    const id = uris[0].split(":track:")[1];
    Spicetify.showNotification("Finding soundalikes…");
    let data;
    try {
      const res = await fetch(`${SERVER}/api/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: `https://open.spotify.com/track/${id}`,
          n: 20,
          diversity: 0.15,
        }),
      });
      data = await res.json();
    } catch (e) {
      Spicetify.showNotification(
        "soundalike server not reachable — run `soundalike serve`.", true);
      return;
    }
    if (!data || !data.ok) {
      Spicetify.showNotification((data && data.error) || "No match found.", true);
      return;
    }
    showModal(data);
  }

  function showModal(data) {
    const s = data.seed, v = data.vibe;
    const wrap = document.createElement("div");
    wrap.style.cssText = "font:14px system-ui,sans-serif;color:#e8eaed";

    const tags = [v.tempo, v.dynamics, v.low_end, v.tone]
      .map((t) => `<span style="background:#1a1f27;border:1px solid #2a313c;border-radius:8px;padding:3px 8px;margin-right:6px;font-size:12px">${esc(t)}</span>`)
      .join("");
    const rows = data.results
      .map(
        (x, i) => `
      <div class="sa-row" data-q="${esc(x.title + " " + x.artist)}"
        style="display:flex;align-items:center;gap:12px;padding:8px 6px;border-radius:8px;cursor:pointer">
        <div style="width:20px;color:#8b93a1;text-align:right">${i + 1}</div>
        <div style="flex:1;min-width:0">
          <div style="font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${esc(x.title)}</div>
          <div style="color:#8b93a1;font-size:12.5px">${esc(x.artist)}</div>
        </div>
        <div style="color:#1db954;font-size:12px">&#9656; play</div>
      </div>`
      )
      .join("");

    wrap.innerHTML = `
      <div style="margin-bottom:6px;color:#8b93a1">Sounds like <b style="color:#e8eaed">${esc(s.title)}</b> &mdash; ${esc(s.artist)}</div>
      <div style="margin-bottom:12px">${tags}</div>
      <div style="max-height:52vh;overflow:auto">${rows}</div>`;

    wrap.querySelectorAll(".sa-row").forEach((el) => {
      el.onmouseenter = () => (el.style.background = "#171c24");
      el.onmouseleave = () => (el.style.background = "transparent");
      el.onclick = () => {
        Spicetify.Platform.History.push(`/search/${encodeURIComponent(el.dataset.q)}`);
        Spicetify.PopupModal.hide();
      };
    });

    Spicetify.PopupModal.display({
      title: "\u25C8 soundalike",
      content: wrap,
      isLarge: true,
    });
  }

  function esc(str) {
    return String(str || "").replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  new Spicetify.ContextMenu.Item(
    "Find soundalikes",
    findSoundalikes,
    onlyTracks,
    "enhance" // Spicetify built-in icon
  ).register();

  console.log("[soundalike] extension loaded — right-click a track to try it.");
})();
