// soundalike — Spicetify extension
// Adds a right-click "Find soundalikes" item to any track in the Spotify
// desktop client. It prefers the optional local soundalike server and otherwise
// uses the hosted library, then shows the results in a Spotify-style modal.
//
// Install (requires a patchable desktop Spotify app): follow this directory's
// README for Marketplace or manual setup.

(function soundalike() {
  const LOCAL_SERVER = "http://127.0.0.1:8787";
  const HOSTED_SERVER = "https://soundalike.yassin.app";
  const LOCAL_PROBE_TIMEOUT_MS = 800;
  const HOSTED_TIMEOUT_MS = 65000;
  const LOCAL_STATUS_TTL_MS = 30000;
  let localStatus = { available: false, checkedAt: 0 };
  let nativeContextChain;
  let warnedAboutNativeMenus = false;

  // Wait until the Spicetify APIs we need are ready.
  if (!(
    window.Spicetify &&
    Spicetify.ContextMenu &&
    Spicetify.GraphQL?.Definitions?.getTrack &&
    Spicetify.GraphQL?.Request &&
    Spicetify.Platform &&
    Spicetify.ReactJSX?.jsx &&
    Spicetify.URI
  )) {
    setTimeout(soundalike, 400);
    return;
  }
  if (window.__soundalikeContextMenuItem) return;

  const onlyTracks = (uris) =>
    Array.isArray(uris) && uris.length === 1 && uris[0].includes(":track:");

  async function fetchWithTimeout(url, options, timeoutMs) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    try {
      return await fetch(url, { ...options, signal: controller.signal });
    } finally {
      clearTimeout(timeout);
    }
  }

  async function localServerReady() {
    if (Date.now() - localStatus.checkedAt < LOCAL_STATUS_TTL_MS) {
      return localStatus.available;
    }
    let available = false;
    try {
      const response = await fetchWithTimeout(
        `${LOCAL_SERVER}/health`,
        { cache: "no-store" },
        LOCAL_PROBE_TIMEOUT_MS
      );
      const health = response.ok ? await response.json() : null;
      available = health?.ok === true;
    } catch {
      available = false;
    }
    localStatus = { available, checkedAt: Date.now() };
    return available;
  }

  async function postRecommendations(server, payload, timeoutMs, allowErrorResponse = false) {
    const options = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    };
    const response = timeoutMs
      ? await fetchWithTimeout(`${server}/api/recommend`, options, timeoutMs)
      : await fetch(`${server}/api/recommend`, options);
    let result;
    try {
      result = await response.json();
    } catch {
      throw new Error(`Recommendation service returned HTTP ${response.status}.`);
    }
    if (!response.ok && !allowErrorResponse) {
      throw new Error(
        result?.error || `Recommendation service returned HTTP ${response.status}.`
      );
    }
    if (!response.ok && !result?.error) {
      throw new Error(`Recommendation service returned HTTP ${response.status}.`);
    }
    return result;
  }

  async function requestRecommendations(payload) {
    if (await localServerReady()) {
      try {
        return {
          data: await postRecommendations(LOCAL_SERVER, payload),
          source: "local",
        };
      } catch (error) {
        localStatus = { available: false, checkedAt: Date.now() };
        console.warn("[soundalike] Local engine failed; using hosted library.", error);
      }
    }
    Spicetify.showNotification(
      "Using hosted Soundalike — the first request after idle can take about 30 seconds.",
      false,
      10000
    );
    return {
      data: await postRecommendations(
        HOSTED_SERVER,
        payload,
        HOSTED_TIMEOUT_MS,
        true
      ),
      source: "hosted",
    };
  }

  async function findSoundalikes(uris) {
    const id = uris[0].split(":track:")[1];
    Spicetify.showNotification("Finding soundalikes…");
    let data;
    let seedTrack;
    try {
      const metadata = await Spicetify.GraphQL.Request(
        Spicetify.GraphQL.Definitions.getTrack,
        { uri: `spotify:track:${id}` }
      );
      seedTrack = metadata?.data?.trackUnion;
      const artist = seedTrack?.firstArtist?.items?.[0]?.profile?.name;
      if (!seedTrack?.name || !artist) {
        throw new Error("Spotify did not return track metadata.");
      }
      const recommendation = await requestRecommendations({
        query: `${seedTrack.name} — ${artist}`,
        n: 20,
        diversity: 0.15,
      });
      data = recommendation.data;
      if (data && typeof data === "object") {
        data.__soundalikeSource = recommendation.source;
      }
    } catch (e) {
      Spicetify.showNotification(
        `soundalike failed: ${e?.message || "recommendation service unavailable"}`, true);
      return;
    }
    if (!data || !data.ok) {
      Spicetify.showNotification((data && data.error) || "No match found.", true);
      return;
    }
    const modal = showModal(data, seedTrack);
    hydrateSpotifyRows(data.results, modal).catch((error) => {
      console.error("[soundalike] Spotify result enrichment failed", error);
    });
  }

  async function findSpotifyTrack(result) {
    const response = await Spicetify.GraphQL.Request(
      Spicetify.GraphQL.Definitions.searchModalResults,
      {
        searchTerm: `${result.title} ${result.artist}`,
        offset: 0,
        limit: 5,
        numberOfTopResults: 5,
        includeAudiobooks: false,
        includeAuthors: false,
      }
    );
    const hits = response?.data?.searchV2?.topResultsV2?.itemsV2 || [];
    const ranked = hits
      .map((hit) => hit?.item?.data)
      .filter((track) => track?.__typename === "Track")
      .map((track) => ({ track, score: spotifyMatchScore(track, result) }))
      .sort((a, b) => b.score - a.score);
    const match = ranked[0]?.score >= 6 ? ranked[0].track : null;
    if (!match) return null;
    try {
      const details = await Spicetify.GraphQL.Request(
        Spicetify.GraphQL.Definitions.getTrack,
        { uri: match.uri }
      );
      return mergeSpotifyTrackDetails(match, details?.data?.trackUnion);
    } catch (error) {
      console.warn(
        `[soundalike] Spotify action metadata lookup failed for ${match.uri}`,
        error
      );
      return match;
    }
  }

  function mergeSpotifyTrackDetails(searchTrack, details) {
    if (details?.__typename !== "Track" || details.uri !== searchTrack.uri) {
      return searchTrack;
    }
    const artists = [
      ...(details.firstArtist?.items || []),
      ...(details.otherArtists?.items || []),
    ];
    return {
      ...searchTrack,
      ...details,
      albumOfTrack: {
        ...searchTrack.albumOfTrack,
        ...details.albumOfTrack,
      },
      artists: {
        items: artists.length ? artists : searchTrack.artists?.items || [],
      },
    };
  }

  function spotifyMatchScore(track, result) {
    const expectedTitle = normalizeLabel(result.title);
    const actualTitle = normalizeLabel(track.name);
    const expectedArtist = normalizeLabel(result.artist);
    const actualArtists = (track.artists?.items || [])
      .map((item) => normalizeLabel(item?.profile?.name))
      .filter(Boolean);
    if (!expectedTitle || !expectedArtist) return 0;
    let score = 0;
    if (actualTitle === expectedTitle) {
      score += 4;
    } else if (actualTitle.includes(expectedTitle) || expectedTitle.includes(actualTitle)) {
      score += 2;
    }
    if (actualArtists.some((artist) => artist === expectedArtist)) {
      score += 4;
    } else if (actualArtists.some(
      (artist) => artist.includes(expectedArtist) || expectedArtist.includes(artist)
    )) {
      score += 2;
    }
    return score;
  }

  function normalizeLabel(value) {
    return String(value || "")
      .normalize("NFKD")
      .replace(/\p{M}/gu, "")
      .toLocaleLowerCase()
      .replace(/[^\p{L}\p{N}]+/gu, " ")
      .trim();
  }

  function artworkUrl(track) {
    const sources = track?.albumOfTrack?.coverArt?.sources || [];
    const url = (
      sources.find((source) => source.width === 64)?.url ||
      sources.find((source) => source.width === 300)?.url ||
      sources[0]?.url ||
      ""
    );
    return url.replace(
      "https://i.scdn.co/image/",
      "https://image-cdn-fa.spotifycdn.com/image/"
    );
  }

  function findNativeContextChain() {
    if (nativeContextChain) return nativeContextChain;
    const registry = Spicetify.Platform?.Registry;
    if (!registry || typeof document.querySelector !== "function") return null;

    const selectors = [
      '[data-testid="now-playing-bar"]',
      '[data-testid="topbar-content"]',
      "main",
      "[data-testid]",
    ];
    for (const selector of selectors) {
      const element = document.querySelector(selector);
      const fiberKey = element && Object.keys(element).find(
        (key) => key.startsWith("__reactFiber$") || key.startsWith("__reactInternalInstance$")
      );
      let fiber = fiberKey ? element[fiberKey] : null;
      let collecting = false;
      const contextChain = [];
      while (fiber) {
        if (fiber.tag === 10) {
          const value = fiber.memoizedProps?.value;
          if (
            !collecting &&
            typeof value?.isDesktop === "boolean" &&
            typeof value?.isWeb === "boolean" &&
            value?.ui
          ) {
            collecting = true;
          }
          if (collecting && fiber.elementType) {
            contextChain.push({ Provider: fiber.elementType, value });
          }
        }
        fiber = fiber.return;
      }
      if (
        contextChain.some(({ value }) => value === registry) &&
        contextChain.some(({ value }) => value === Spicetify._platform) &&
        contextChain.some(({ value }) => (
          value?.store &&
          value?.subscription &&
          typeof value.store.getState === "function"
        ))
      ) {
        nativeContextChain = contextChain;
        return nativeContextChain;
      }
    }
    return null;
  }

  function nativeTrackMenu(track) {
    if (!track?.uri) return null;
    const React = Spicetify.React;
    const components = Spicetify.ReactComponent;
    const contextChain = findNativeContextChain();
    if (!(
      React?.createElement &&
      React?.Suspense &&
      components?.RightClickMenu &&
      components?.TrackMenu &&
      contextChain
    )) {
      if (!warnedAboutNativeMenus) {
        warnedAboutNativeMenus = true;
        console.warn(
          "[soundalike] Native Spotify track menus are unavailable; direct playback remains enabled."
        );
      }
      return null;
    }

    const artists = (track.artists?.items || [])
      .map((artist) => ({
        type: "artist",
        name: artist?.profile?.name,
        uri: artist?.uri,
      }))
      .filter((artist) => artist.name && artist.uri);
    const menuProps = {
      uri: track.uri,
      canAddToQueue: true,
    };
    if (track.albumOfTrack?.uri) menuProps.albumUri = track.albumOfTrack.uri;
    if (artists.length) menuProps.artists = artists;

    let menu = React.createElement(components.TrackMenu, menuProps);
    for (const { Provider, value } of contextChain) {
      menu = React.createElement(Provider, { value }, menu);
    }
    return React.createElement(
      React.Suspense,
      {
        fallback: React.createElement(
          "div",
          { className: "sa-menu-loading" },
          "Loading track actions..."
        ),
      },
      menu
    );
  }

  function spotifyArtists(track, fallback) {
    const artists = (track?.artists?.items || [])
      .map((item) => item?.profile?.name)
      .filter(Boolean);
    return artists.length ? artists.join(", ") : fallback;
  }

  async function activateResult(result, track) {
    if (track?.uri && typeof Spicetify.Player?.playUri === "function") {
      try {
        await Spicetify.Player.playUri(track.uri);
      } catch (error) {
        console.error(`[soundalike] Could not play ${track.uri}`, error);
        Spicetify.showNotification(`Could not play ${result.title}.`, true);
      }
      return;
    }
    Spicetify.Platform.History.push(
      `/search/${encodeURIComponent(`${result.title} ${result.artist}`)}`
    );
    Spicetify.PopupModal.hide();
  }

  function renderResultRow(row, result, index, track = null) {
    const artist = spotifyArtists(track, result.artist);
    const image = track ? artworkUrl(track) : "";
    row.dataset.q = `${result.title} ${result.artist}`;
    if (track?.uri) row.dataset.uri = track.uri;

    const React = Spicetify.React;
    if (!(React?.createElement && Spicetify.ReactDOM?.createRoot)) {
      const artistNode = row.querySelector(".sa-artist");
      if (artistNode) artistNode.textContent = artist;
      if (image) {
        const img = document.createElement("img");
        img.src = image;
        img.alt = "";
        img.loading = "lazy";
        row.querySelector(".sa-cover")?.replaceChildren(img);
      }
      row.onclick = () => activateResult(result, track);
      return;
    }

    if (!row.__soundalikeRoot) {
      row.__soundalikeRoot = Spicetify.ReactDOM.createRoot(row);
    }
    const activate = () => activateResult(result, track);
    const child = React.createElement(
      "div",
      {
        className: "sa-row-content",
        role: "button",
        tabIndex: 0,
        title: track?.uri
          ? `Play ${result.title} by ${artist}`
          : `Search Spotify for ${result.title} by ${result.artist}`,
        "aria-label": track?.uri
          ? `Play ${result.title} by ${artist}`
          : `Search Spotify for ${result.title} by ${result.artist}`,
        onClick: activate,
        onKeyDown: (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            activate();
          }
        },
      },
      React.createElement("div", { className: "sa-rank" }, index + 1),
      React.createElement(
        "div",
        { className: "sa-cover", "aria-hidden": "true" },
        image
          ? React.createElement("img", { src: image, alt: "", loading: "lazy" })
          : React.createElement("span", null, "\u266B")
      ),
      React.createElement(
        "div",
        { className: "sa-meta" },
        React.createElement("div", { className: "sa-title" }, result.title),
        React.createElement("div", { className: "sa-artist" }, artist)
      ),
      React.createElement(
        "div",
        { className: "sa-open", "aria-hidden": "true" },
        track?.uri ? "\u25B6" : "\u203A"
      )
    );
    const menu = nativeTrackMenu(track);
    row.__soundalikeRoot.render(
      menu
        ? React.createElement(Spicetify.ReactComponent.RightClickMenu, { menu }, child)
        : child
    );
  }

  async function hydrateSpotifyRows(results, modal) {
    if (!Spicetify.GraphQL.Definitions.searchModalResults) {
      console.warn("[soundalike] Spotify artwork lookup is unavailable in this client.");
      return;
    }
    let cursor = 0;
    let enriched = 0;
    async function worker() {
      while (cursor < results.length) {
        const index = cursor++;
        try {
          const track = await findSpotifyTrack(results[index]);
          const row = modal.querySelector(`.sa-row[data-index="${index}"]`);
          if (!track || !row) continue;
          renderResultRow(row, results[index], index, track);
          enriched++;
        } catch (error) {
          console.warn(
            `[soundalike] Spotify metadata lookup failed for ${results[index].title}`,
            error
          );
        }
      }
    }
    const workerCount = Math.min(4, results.length);
    await Promise.all(Array.from({ length: workerCount }, worker));
    console.log(
      `[soundalike] added Spotify metadata to ${enriched}/${results.length} rows.`
    );
  }

  function showModal(data, seedTrack) {
    const s = data.seed, v = data.vibe;
    const wrap = document.createElement("div");
    const seedImage = artworkUrl(seedTrack);
    const source = data.__soundalikeSource === "local"
      ? "LOCAL ENGINE"
      : "HOSTED LIBRARY";
    wrap.className = "sa-results";

    const tags = [v.tempo, v.dynamics, v.low_end, v.tone]
      .map((t) => `<span class="sa-tag">${esc(t)}</span>`)
      .join("");
    const rows = data.results
      .map(
        (x, i) => `
      <div class="sa-row" data-index="${i}" data-q="${esc(x.title + " " + x.artist)}">
        <div class="sa-rank">${i + 1}</div>
        <div class="sa-cover" aria-hidden="true"><span>♫</span></div>
        <div class="sa-meta">
          <div class="sa-title">${esc(x.title)}</div>
          <div class="sa-artist">${esc(x.artist)}</div>
        </div>
        <div class="sa-open">›</div>
      </div>`
      )
      .join("");

    wrap.innerHTML = `
      <style>
        .sa-results{font:14px var(--encore-body-font-stack,system-ui,sans-serif);color:var(--spice-text,#fff)}
        .sa-seed{display:flex;align-items:center;gap:12px;margin:0 0 10px}
        .sa-seed img,.sa-seed-fallback{width:56px;height:56px;border-radius:4px;object-fit:cover;background:#282828}
        .sa-seed-fallback{display:grid;place-items:center;color:#b3b3b3;font-size:22px}
        .sa-kicker{color:var(--spice-subtext,#b3b3b3);font-size:12px;margin-bottom:2px}
        .sa-seed-title{font-size:18px;font-weight:700;line-height:1.25}
        .sa-seed-artist{color:var(--spice-subtext,#b3b3b3);margin-top:2px}
        .sa-tags{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px}
        .sa-tag{background:rgba(255,255,255,.08);border-radius:999px;padding:4px 9px;color:var(--spice-subtext,#b3b3b3);font-size:12px}
        .sa-list{max-height:52vh;overflow:auto;padding-right:4px}
        .sa-list-head,.sa-row-content{display:grid;grid-template-columns:28px 44px minmax(0,1fr) 28px;align-items:center;column-gap:12px}
        .sa-list-head{height:28px;color:var(--spice-subtext,#b3b3b3);font-size:12px;border-bottom:1px solid rgba(255,255,255,.1);margin-bottom:4px}
        .sa-row{border-radius:6px}
        .sa-row-content{min-height:56px;padding:4px 6px;border-radius:6px;cursor:pointer;outline:none}
        .sa-row-content:hover,.sa-row-content:focus-visible{background:rgba(255,255,255,.1)}
        .sa-rank{text-align:right;color:var(--spice-subtext,#b3b3b3);font-variant-numeric:tabular-nums}
        .sa-cover{width:44px;height:44px;border-radius:4px;background:#282828;display:grid;place-items:center;color:#727272;overflow:hidden}
        .sa-cover img{width:100%;height:100%;object-fit:cover}
        .sa-meta{min-width:0}
        .sa-title{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
        .sa-artist{color:var(--spice-subtext,#b3b3b3);font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:2px}
        .sa-open{color:var(--spice-subtext,#b3b3b3);font-size:24px;text-align:center}
        .sa-row-content:hover .sa-title,.sa-row-content:hover .sa-open,.sa-row-content:focus-visible .sa-title,.sa-row-content:focus-visible .sa-open{color:var(--spice-text,#fff)}
        .sa-menu-loading{padding:8px 12px;color:var(--spice-subtext,#b3b3b3);white-space:nowrap}
      </style>
      <div class="sa-seed">
        ${seedImage
          ? `<img src="${esc(seedImage)}" alt="">`
          : `<div class="sa-seed-fallback" aria-hidden="true">♫</div>`}
        <div>
          <div class="sa-kicker">SOUNDS LIKE · ${source}</div>
          <div class="sa-seed-title">${esc(s.title)}</div>
          <div class="sa-seed-artist">${esc(s.artist)}</div>
        </div>
      </div>
      <div class="sa-tags">${tags}</div>
      <div class="sa-list-head"><div>#</div><div></div><div>Title</div><div></div></div>
      <div class="sa-list">${rows}</div>`;

    wrap.querySelectorAll(".sa-row").forEach((row, index) => {
      renderResultRow(row, data.results[index], index);
    });

    Spicetify.PopupModal.display({
      title: "\u25C8 soundalike",
      content: wrap,
      isLarge: true,
    });
    return wrap;
  }

  function esc(str) {
    return String(str || "").replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  window.__soundalikeContextMenuItem = new Spicetify.ContextMenu.Item(
    "Find soundalikes",
    findSoundalikes,
    onlyTracks,
    "enhance" // Spicetify built-in icon
  );
  window.__soundalikeContextMenuItem.register();

  console.log("[soundalike] extension loaded — right-click a track to try it.");
})();
