(() => {
  const input = document.querySelector("#q");
  if (!input) return;

  input.removeEventListener("input", onType);

  const cache = new Map();
  const maxCacheEntries = 64;
  const primaryRecommendationServer = "https://soundalike-api.yassin.app";
  const fallbackRecommendationServer = "https://soundalike.yassin.app";
  const primaryRecommendationTimeoutMs = 5000;
  const fallbackRecommendationTimeoutMs = 65000;
  let timer = null;
  let controller = null;
  let requestSequence = 0;

  const normalize = value =>
    value.trim().toLocaleLowerCase().replace(/\s+/g, " ");

  function remember(query, items) {
    if (cache.has(query)) cache.delete(query);
    cache.set(query, items);
    if (cache.size > maxCacheEntries) {
      cache.delete(cache.keys().next().value);
    }
  }

  function show(items) {
    acItems = items;
    acSel = -1;
    renderAc();
  }

  function cachedPrefix(query) {
    const tokens = query.split(" ");
    const candidates = [...cache.entries()]
      .filter(([cached]) => query.startsWith(cached))
      .sort((left, right) => right[0].length - left[0].length);
    for (const [, items] of candidates) {
      const filtered = items.filter(item => {
        const text = normalize(`${item.title} ${item.artist}`);
        return tokens.every(token => text.includes(token));
      });
      if (filtered.length) return filtered;
    }
    return [];
  }

  async function requestSuggestions(value, query, sequence) {
    controller = new AbortController();
    const activeController = controller;
    try {
      const response = await fetch(
        `/api/search?q=${encodeURIComponent(value)}`,
        { signal: activeController.signal }
      );
      if (!response.ok) throw new Error(`search returned ${response.status}`);
      const body = await response.json();
      const items = body.results || [];
      remember(query, items);
      if (
        sequence === requestSequence &&
        normalize(input.value) === query &&
        document.activeElement === input
      ) {
        show(items);
      }
    } catch (error) {
      if (error.name !== "AbortError" && sequence === requestSequence) {
        hideAc();
      }
    } finally {
      if (controller === activeController) controller = null;
    }
  }

  function onTypeCached() {
    const value = input.value.trim();
    const query = normalize(value);
    clearTimeout(timer);
    requestSequence += 1;
    if (controller) controller.abort();

    if (query.length < 2) {
      hideAc();
      return;
    }

    if (cache.has(query)) {
      show(cache.get(query));
      return;
    }

    const immediate = cachedPrefix(query);
    if (immediate.length) show(immediate);
    const sequence = requestSequence;
    timer = setTimeout(
      () => requestSuggestions(value, query, sequence),
      100
    );
  }

  input.addEventListener("input", onTypeCached);

  async function fetchRecommendations(server, item, timeoutMs) {
    const params = new URLSearchParams({
      query: `${item.title} — ${item.artist}`,
      n: "20",
      diversity: "0.15",
      v: "2",
    });
    const requestController = new AbortController();
    const timeout = window.setTimeout(() => requestController.abort(), timeoutMs);
    try {
      const response = await fetch(
        `${server}/api/spicetify_recommend?${params}`,
        { cache: "default", signal: requestController.signal }
      );
      const body = await response.json();
      if (!response.ok && response.status !== 400 && response.status !== 422) {
        throw new Error(body.error || `recommendation returned ${response.status}`);
      }
      return body;
    } finally {
      window.clearTimeout(timeout);
    }
  }

  async function requestRecommendations(item) {
    try {
      return await fetchRecommendations(
        primaryRecommendationServer,
        item,
        primaryRecommendationTimeoutMs
      );
    } catch (error) {
      console.warn(
        "[soundalike] Primary recommendation service failed; using Vercel.",
        error
      );
      return fetchRecommendations(
        fallbackRecommendationServer,
        item,
        fallbackRecommendationTimeoutMs
      );
    }
  }

  const bestMatchesTimeoutMs = 30000;
  const bestMatchesRetries = 2;
  // Best matches and Same vibe come from one any-song response.
  const resultTabs = [
    {
      key: "best",
      label: "Best matches",
      blurb:
        "Likely covers, samples and shared melodies first, then similar-sounding songs by related artists from the same scene.",
    },
    {
      key: "vibe",
      label: "Same vibe",
      blurb:
        "Songs from these matches that share this song's tempo and overall sound, one per artist.",
    },
    {
      key: "sounds",
      label: "Sounds like",
      blurb: "Songs with the same overall sound: low end, dynamics and tone.",
    },
  ];
  let pickSequence = 0;
  let shownPick = null;

  class RequestError extends Error {
    constructor(status, message) {
      super(message);
      this.status = status;
    }
  }

  const pause = ms => new Promise(resolve => window.setTimeout(resolve, ms));

  const songKey = song =>
    `${song.title || ""} ${song.artist || ""}`
      .normalize("NFKD")
      .replace(/[\u0300-\u036f]/g, "")
      .toLocaleLowerCase()
      .replace(/[^\p{L}\p{N}]+/gu, " ")
      .trim();

  const spotifySearchUrl = (title, artist) =>
    `https://open.spotify.com/search/${encodeURIComponent(`${title} ${artist}`.trim())}`;

  async function fetchBestMatches(item, isCurrent, lang) {
    const params = new URLSearchParams({
      title: item.title,
      artist: item.artist,
      k: "20",
      lang,
    });
    for (let attempt = 0; ; attempt += 1) {
      const requestController = new AbortController();
      const timeout = window.setTimeout(
        () => requestController.abort(),
        bestMatchesTimeoutMs
      );
      let response;
      let body = {};
      try {
        response = await fetch(`/api/anysong/similar?${params}`, {
          signal: requestController.signal,
        });
        try {
          body = await response.json();
        } catch {}
      } catch (error) {
        throw new RequestError(error.name === "AbortError" ? 504 : 0, "");
      } finally {
        window.clearTimeout(timeout);
      }
      // 503 means every analysis slot is taken; a short wait usually frees one.
      if (response.status === 503 && attempt < bestMatchesRetries && isCurrent()) {
        await pause(1500 * (attempt + 1));
        continue;
      }
      if (!response.ok || !body.ok) {
        throw new RequestError(response.status, body.error || "");
      }
      return body;
    }
  }

  async function requestSoundsLike(item) {
    let body;
    try {
      body = await requestRecommendations(item);
    } catch {
      throw new RequestError(0, "Server error.");
    }
    if (!body.ok) throw new RequestError(0, body.error || "No match.");
    return body;
  }

  function bestErrorText(error) {
    const status = error && error.status;
    if (status === 400) return "Best matches need both a title and an artist.";
    if (status === 404) {
      return "This song isn't on Deezer with a preview, so it has no best matches.";
    }
    if (status === 429) {
      return "You're going a bit fast. Wait a few seconds and try again.";
    }
    if (status === 503) return "The matcher is busy right now. Try again in a moment.";
    if (status === 504) return "That took too long. Try again.";
    if (status === 502) return "Deezer didn't respond. Try again in a moment.";
    if (status === 422) return "We couldn't decode this song's preview.";
    return "Best matches are unavailable right now. Try again in a moment.";
  }

  function anysongRows(body, list) {
    const rows = ((body.results || {})[list] || []).filter(x =>
      Number.isInteger(x.id)
    );
    const scores = rows.map(x => Number(x.score) || 0);
    const high = Math.max(...scores);
    const low = Math.min(...scores);
    return rows.map((x, index) => {
      const title = x.title || `Deezer track ${x.id}`;
      const artist = x.artist || "";
      const why =
        x.ranks && typeof x.ranks === "object"
          ? Object.entries(x.ranks)
              .filter(([, rank]) => Number.isInteger(rank))
              .map(([list, rank]) => `${list} #${rank}`)
              .join(" · ")
          : "";
      return {
        id: String(x.id),
        title,
        artist,
        why,
        // Scores are only comparable within one list, so bars are relative to it.
        width:
          high > low
            ? Math.round(25 + (75 * (scores[index] - low)) / (high - low))
            : 100,
        url: spotifySearchUrl(title, artist),
      };
    });
  }

  function soundsRows(body) {
    const rows = body.results || [];
    const top = Math.max(...rows.map(x => Number(x.neural_sim) || 0), 0.001);
    return rows.map(x => ({
      id: String(x.deezer_id),
      title: x.title || "",
      artist: x.artist || "",
      why: "",
      width: Math.round((100 * (Number(x.neural_sim) || 0)) / top),
      url: x.spotify_url || spotifySearchUrl(x.title || "", x.artist || ""),
    }));
  }

  const rowHtml = (x, index) =>
    `<li class="row" data-id="${esc(x.id)}">` +
    `<button class="pv" type="button" aria-label="Preview ${esc(x.title)} by ${esc(x.artist)}" title="Preview" data-id="${esc(x.id)}">▶</button>` +
    `<div class="rank">${index + 1}</div>` +
    `<div class="meta"><div class="t">${esc(x.title)}</div><div class="a">${esc(x.artist)}${
      x.why ? `<span class="why"> · ${esc(x.why)}</span>` : ""
    }</div></div>` +
    `<div class="bar"><i style="width:${x.width}%"></i></div>` +
    `<a class="play" href="${esc(x.url)}" target="_blank" rel="noopener">Open in Spotify ▸</a></li>`;

  const entryFor = picked => picked[picked.tab === "vibe" ? "best" : picked.tab];

  function seedTags(picked) {
    const entry = entryFor(picked);
    if (entry.status !== "ready") return [];
    if (picked.tab !== "sounds") {
      const query = entry.body.query || {};
      const total = (entry.body.timings_ms || {}).total;
      const bpm = typeof query.bpm === "number" ? Math.round(query.bpm) : 0;
      return [
        query.source !== "live"
          ? "in our library"
          : entry.body.cached ? "heard live earlier" : "heard live just now",
        picked.tab === "vibe" && bpm > 0 ? `${bpm} BPM` : "",
        !entry.body.cached && total ? `matched in ${(total / 1000).toFixed(1)} s` : "",
      ].filter(Boolean);
    }
    const vibe = entry.body.vibe || {};
    return [vibe.low_end, vibe.dynamics, vibe.tone].filter(Boolean);
  }

  function blurbText(picked) {
    const tab = resultTabs.find(candidate => candidate.key === picked.tab);
    const entry = entryFor(picked);
    const seed = entry.status === "ready" && entry.body.seed;
    if (picked.tab === "sounds" && seed && songKey(seed) !== songKey(picked.seed)) {
      return `${tab.blurb} Based on “${seed.title}” by ${seed.artist}, the closest song in that library.`;
    }
    return tab.blurb;
  }

  function footText(picked) {
    const entry = entryFor(picked);
    if (picked.tab === "best") {
      return "Best matches combine the melody and samples rankings. ▶ = 30-second preview.";
    }
    if (picked.tab === "vibe") {
      return "Same vibe comes from the same matcher as Best matches. ▶ = 30-second preview.";
    }
    const size = entry.status === "ready" && Number(entry.body.library_size);
    return size
      ? `Matched from a ${size.toLocaleString()}-track library. ▶ = 30-second preview.`
      : "▶ = 30-second preview.";
  }

  function languageName(code) {
    try {
      return new Intl.DisplayNames(["en"], { type: "language" }).of(code);
    } catch {
      return code.toUpperCase();
    }
  }

  // Best matches and Same vibe hide confidently other-language songs unless
  // the listener asks for all languages. Sounds like is not filtered. The
  // hidden count is Best's own, so Same vibe only says what it leaves out.
  function renderLanguage(picked) {
    const box = document.querySelector("#language");
    const button = document.querySelector("#language-toggle");
    if (!box || !button) return;
    box.hidden =
      (picked.tab !== "best" && picked.tab !== "vibe") || picked.best.status === "failed";
    if (box.hidden) return;
    if (picked.best.status !== "ready") {
      button.setAttribute("aria-pressed", String(picked.lang === "all"));
      button.textContent = picked.lang === "all" ? "Show same language" : "Show all languages";
      return;
    }
    const query = picked.best.body.query || {};
    const code = query.language || "unknown";
    const all = query.language_mode === "all";
    const hidden = Math.max(0, Number(query.hidden) || 0);
    document.querySelector("#language-status").textContent =
      code === "unknown"
        ? "Language uncertain; showing all languages."
        : picked.tab === "vibe"
          ? `${languageName(code)} · ${all ? "all languages" : "other languages left out"}`
          : `${languageName(code)} · ${
              hidden
                ? `${hidden} other-language ${hidden === 1 ? "match" : "matches"} ${
                    all ? "in" : "hidden from"
                  } the top 20`
                : "no other-language matches in the top 20"
            }`;
    button.hidden = code === "unknown";
    button.setAttribute("aria-pressed", String(all));
    button.textContent = all ? "Show same language" : "Show all languages";
  }

  function toggleLanguage(picked) {
    if (picked.best.status !== "ready") return;
    picked.lang = (picked.best.body.query || {}).language_mode === "all" ? "same" : "all";
    const sequence = ++picked.languageSequence;
    picked.best = { status: "pending" };
    renderTab(picked);
    const isCurrent = () =>
      shownPick === picked && picked.languageSequence === sequence;
    fetchBestMatches(picked.item, isCurrent, picked.lang)
      .then(
        body => ({ status: "ready", body }),
        error => ({ status: "failed", error })
      )
      .then(entry => {
        if (picked.languageSequence !== sequence) return;
        picked.best = entry;
        renderTab(picked);
      });
  }

  function markPlaying(list) {
    if (playingId == null || audio.paused) return;
    list.querySelectorAll(".pv").forEach(button => {
      if (button.dataset.id !== String(playingId)) return;
      button.textContent = "⏸";
      button.classList.add("playing");
      button.closest("li.row").classList.add("playing");
    });
  }

  function resetListActions(count) {
    document.querySelector("#copy").disabled = !count;
    const save = document.querySelector("#save");
    if (save.textContent === "Resolving…") return;
    save.textContent = "＋ Save to Spotify";
    save.onclick = savePlaylist;
    save.disabled = !count;
  }

  function markTabs(picked) {
    document.querySelectorAll("#out .tab").forEach(button => {
      const selected = button.dataset.tab === picked.tab;
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
  }

  // Servers without Same vibe send no vibe list, so its tab stays hidden. A
  // language switch keeps the tab while the list reloads.
  function syncVibeTab(picked) {
    if (picked.best.status === "ready") {
      const vibe = (picked.best.body.results || {}).vibe;
      picked.hasVibe = Array.isArray(vibe) && vibe.length > 0;
    }
    const button = document.querySelector("#tab-vibe");
    if (button) button.hidden = !picked.hasVibe;
    if (picked.tab === "vibe" && !picked.hasVibe) {
      picked.tab = "best";
      markTabs(picked);
    }
  }

  function renderTab(picked) {
    const list = document.querySelector("#matches");
    if (shownPick !== picked || !list) return;
    syncVibeTab(picked);
    const entry = entryFor(picked);
    let rows = [];
    if (entry.status === "ready") {
      rows =
        picked.tab === "sounds"
          ? soundsRows(entry.body)
          : anysongRows(entry.body, picked.tab);
    }
    lastResults = rows;
    list.dataset.tab = picked.tab;
    list.setAttribute("aria-labelledby", `tab-${picked.tab}`);
    document.querySelector("#seedtags").innerHTML = seedTags(picked)
      .map(tag => `<span class="tag">${esc(tag)}</span>`)
      .join("");
    document.querySelector("#matchblurb").textContent = blurbText(picked);
    renderLanguage(picked);
    if (entry.status === "pending") {
      list.innerHTML = '<li class="state"><span class="spin"></span> matching…</li>';
    } else if (entry.status === "failed") {
      const message =
        picked.tab === "sounds" ? entry.error.message : bestErrorText(entry.error);
      list.innerHTML = `<li class="state err">${esc(message)}</li>`;
    } else if (!rows.length) {
      list.innerHTML = '<li class="state">No matches in this list.</li>';
    } else {
      list.innerHTML = rows.map(rowHtml).join("");
    }
    list.querySelectorAll(".pv").forEach(button => {
      button.onclick = () => togglePreview(button.dataset.id, button);
    });
    markPlaying(list);
    resetListActions(rows.length);
    document.querySelector("#foot").textContent = footText(picked);
  }

  function selectTab(picked, key) {
    if (picked.tab === key && document.querySelector("#matches")?.dataset.tab === key) {
      return;
    }
    picked.tab = key;
    markTabs(picked);
    renderTab(picked);
  }

  function showPick(picked) {
    const out = document.querySelector("#out");
    const query = picked.best.status === "ready" ? picked.best.body.query || {} : {};
    const legacySeed = picked.sounds.status === "ready" && picked.sounds.body.seed;
    const seed = query.title
      ? { title: query.title, artist: query.artist || "" }
      : legacySeed || { title: picked.item.title, artist: picked.item.artist };
    picked.seed = seed;
    shownPick = picked;
    lastSeed = seed;
    out.innerHTML =
      `<div class="seed"><div><h2>${esc(seed.title)}</h2><div class="art">${esc(seed.artist)}</div><div class="tags" id="seedtags"></div></div>` +
      '<div class="actions"><button class="copy" id="share" type="button">🔗 Share</button><button class="copy" id="copy" type="button">⧉ Copy list</button><button class="save" id="save" type="button">＋ Save to Spotify</button></div></div>' +
      `<div class="tabs" role="tablist" aria-label="Kind of match">${resultTabs
        .map(
          tab =>
            `<button class="tab" type="button" role="tab" id="tab-${tab.key}" data-tab="${tab.key}" aria-controls="matches" aria-selected="false" tabindex="-1">${tab.label}</button>`
        )
        .join("")}</div>` +
      '<div class="language-filter" id="language" role="status" hidden><span id="language-status"></span><button id="language-toggle" type="button" aria-pressed="false" hidden></button></div>' +
      '<p class="blurb" id="matchblurb"></p><ol id="matches" role="tabpanel"></ol>';
    document.querySelector("#copy").onclick = copyList;
    document.querySelector("#share").onclick = shareLink;
    document.querySelector("#language-toggle").onclick = () => toggleLanguage(picked);
    const tabs = [...out.querySelectorAll(".tab")];
    tabs.forEach(button => {
      button.onclick = () => selectTab(picked, button.dataset.tab);
      button.onkeydown = event => {
        if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
        event.preventDefault();
        const shown = tabs.filter(tab => !tab.hidden);
        const step = event.key === "ArrowRight" ? 1 : shown.length - 1;
        const next = shown[(shown.indexOf(button) + step) % shown.length];
        next.focus();
        selectTab(picked, next.dataset.tab);
      };
    });
    try {
      history.replaceState(
        {},
        "",
        "#s=" + encodeURIComponent(`${seed.title} — ${seed.artist}`)
      );
    } catch {}
    selectTab(picked, picked.tab);
  }

  pick = async item => {
    hideAc();
    input.value = `${item.title} — ${item.artist}`;
    const sequence = ++pickSequence;
    const isCurrent = () => sequence === pickSequence;
    const out = document.querySelector("#out");
    shownPick = null;
    out.innerHTML = '<div class="state"><span class="spin"></span> matching…</div>';
    const slow = window.setTimeout(() => {
      const state = out.querySelector(".state");
      if (isCurrent() && !shownPick && state) {
        state.innerHTML =
          '<span class="spin"></span> matching… songs that are new to us are analysed live, which takes a few seconds.';
      }
    }, 1600);
    const picked = {
      item,
      tab: "best",
      seed: null,
      lang: "same",
      languageSequence: 0,
      hasVibe: false,
      best: { status: "pending" },
      sounds: { status: "pending" },
    };
    const settle = (key, promise) =>
      promise.then(
        body => {
          picked[key] = { status: "ready", body };
        },
        error => {
          picked[key] = { status: "failed", error };
        }
      );
    const bestDone = settle(
      "best",
      item.artist
        ? fetchBestMatches(item, isCurrent, picked.lang)
        : Promise.reject(new RequestError(400, ""))
    );
    const soundsDone = settle("sounds", requestSoundsLike(item));
    try {
      await bestDone;
      if (!isCurrent()) return;
      if (picked.best.status !== "ready") {
        await soundsDone;
        if (!isCurrent()) return;
        if (picked.sounds.status !== "ready") {
          out.innerHTML = `<div class="state err">${esc(picked.sounds.error.message)}</div>`;
          return;
        }
        picked.tab = "sounds";
      }
      showPick(picked);
      soundsDone.then(() => {
        if (shownPick === picked && picked.tab === "sounds") renderTab(picked);
      });
    } finally {
      window.clearTimeout(slow);
    }
  };

  const prewarm = () => {
    fetch("/api/search?q=lo&limit=1", { cache: "force-cache" }).catch(() => {});
  };
  if ("requestIdleCallback" in window) {
    window.requestIdleCallback(prewarm, { timeout: 1500 });
  } else {
    window.setTimeout(prewarm, 300);
  }
})();
