(() => {
  const input = document.querySelector("#q");
  const ac = document.querySelector("#ac");
  const out = document.querySelector("#out");
  if (!input || !ac || !out) return;

  // One search bar for any song on Deezer: suggestions come from
  // /api/anysong/search and every tab reads one /api/anysong/similar answer.
  const ID_RE = /^[1-9][0-9]{0,15}$/;
  const suggestDelayMs = 220;
  const maxCacheEntries = 64;
  const bestMatchesTimeoutMs = 30000;
  const bestMatchesRetries = 3;
  const bestMatchesPolicy = "best-vibe-sounds-v3";
  const resultTabs = [
    {
      key: "best",
      label: "Best matches",
      blurb:
        "Likely covers, samples and shared melodies first, then similar-sounding songs by artists this song's listeners also play.",
    },
    {
      key: "vibe",
      label: "Same vibe",
      blurb:
        "Songs from these matches that share this song's tempo and overall sound, one per artist.",
    },
    {
      key: "melody",
      label: "Same melody",
      blurb: "Covers, interpolations and songs that reuse the tune.",
    },
    {
      key: "samples",
      label: "Samples & loops",
      blurb: "Songs that may share a sample, a loop or a chopped-up section.",
    },
    {
      key: "sounds",
      label: "Sounds like",
      blurb:
        "Songs with the same overall sound (instruments, production and mood), favouring artists this song's listeners also play.",
    },
  ];
  // Servers without the fused Sounds like list send only the raw sound channel.
  const sonicBlurb = "A similar overall sound: production, instruments and mood.";
  const DEMO = [
    ["Plastic Love", "Mariya Takeuchi"],
    ["Redbone", "Childish Gambino"],
    ["Only Shallow", "My Bloody Valentine"],
    ["money machine", "100 gecs"],
    ["Chamber of Reflection", "Mac DeMarco"],
    ["Close Eyes", "DVRST"],
  ];
  const SURPRISE = [
    ...DEMO,
    ["Mask Off", "Future"],
    ["Snooze", "SZA"],
    ["So What", "Miles Davis"],
    ["Master of Puppets", "Metallica"],
    ["Last Last", "Burna Boy"],
    ["Ditto", "NewJeans"],
    ["Space Song", "Beach House"],
    ["Nights", "Frank Ocean"],
    ["Sweater Weather", "The Neighbourhood"],
    ["HUMBLE.", "Kendrick Lamar"],
    ["EARFQUAKE", "Tyler, The Creator"],
    ["Windowlicker", "Aphex Twin"],
    ["Reptilia", "The Strokes"],
    ["Tití Me Preguntó", "Bad Bunny"],
    ["Svefn-g-englar", "Sigur Rós"],
    ["Teardrop", "Massive Attack"],
    ["Sicko Mode", "Travis Scott"],
    ["Provenza", "Karol G"],
    ["Bangarang", "Skrillex"],
    ["Doomsday", "MF DOOM"],
  ];

  const cache = new Map(); // normalized query -> Deezer search results
  const cards = new Map(); // Deezer id -> {id, title, artist, cover}
  const resolved = new Map(); // song key -> Deezer id, for chips and #s= links
  let acItems = [];
  let acSel = -1;
  let timer = null;
  let controller = null;
  let requestSequence = 0;
  let tab = "best";
  let loadSequence = 0;
  let currentKey = null;
  let shownPick = null;
  let lastSurprise = -1;

  class RequestError extends Error {
    constructor(status, message) {
      super(message);
      this.status = status;
    }
  }

  const pause = ms => new Promise(resolve => window.setTimeout(resolve, ms));

  const normalize = value =>
    value.trim().toLocaleLowerCase().replace(/\s+/g, " ");

  const songKey = song =>
    `${song.title || ""} ${song.artist || ""}`
      .normalize("NFKD")
      .replace(/[\u0300-\u036f]/g, "")
      .toLocaleLowerCase()
      .replace(/[^\p{L}\p{N}]+/gu, " ")
      .trim();

  const spotifySearchUrl = (title, artist) =>
    `https://open.spotify.com/search/${encodeURIComponent(`${title} ${artist}`.trim())}`;

  // Old any-song links call Sounds like "sonic".
  const tabKey = key => {
    const name = key === "sonic" ? "sounds" : key;
    return resultTabs.some(candidate => candidate.key === name) ? name : null;
  };

  function safeImg(url) {
    try {
      const parsed = new URL(url);
      return parsed.protocol === "https:" &&
        (parsed.hostname === "dzcdn.net" || parsed.hostname.endsWith(".dzcdn.net"))
        ? parsed.href
        : "";
    } catch {
      return "";
    }
  }

  // ---------- search ----------
  function remember(query, items) {
    if (cache.has(query)) cache.delete(query);
    cache.set(query, items);
    if (cache.size > maxCacheEntries) {
      cache.delete(cache.keys().next().value);
    }
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

  function hideAc() {
    ac.style.display = "none";
    acItems = [];
    acSel = -1;
    input.setAttribute("aria-expanded", "false");
    input.removeAttribute("aria-activedescendant");
  }

  function renderAc() {
    if (!acItems.length) {
      hideAc();
      return;
    }
    ac.innerHTML = acItems
      .map((song, index) => {
        const cover = safeImg(song.cover);
        const off = !song.has_preview;
        return (
          `<div class="opt${index === acSel ? " sel" : ""}" id="ac-${index}" role="option" data-i="${index}" aria-selected="${index === acSel}" aria-disabled="${off}">` +
          (cover
            ? `<img class="thumb" src="${esc(cover)}" alt="" loading="lazy">`
            : '<span class="thumb"></span>') +
          `<span class="meta"><span>${esc(song.title)}</span><br><small>${esc(song.artist)}${
            song.album ? ` · ${esc(song.album)}` : ""
          }${off ? " · no preview" : ""}</small></span></div>`
        );
      })
      .join("");
    ac.style.display = "block";
    input.setAttribute("aria-expanded", "true");
    if (acSel >= 0) {
      input.setAttribute("aria-activedescendant", `ac-${acSel}`);
      document.getElementById(`ac-${acSel}`)?.scrollIntoView({ block: "nearest" });
    } else {
      input.removeAttribute("aria-activedescendant");
    }
    ac.querySelectorAll(".opt").forEach(option => {
      option.onmousedown = event => event.preventDefault();
      option.onclick = () => pick(acItems[Number(option.dataset.i)]);
    });
  }

  function show(items) {
    acItems = items;
    acSel = -1;
    renderAc();
  }

  async function requestSuggestions(value, query, sequence) {
    controller = new AbortController();
    const activeController = controller;
    try {
      const response = await fetch(
        `/api/anysong/search?limit=8&q=${encodeURIComponent(value)}`,
        { signal: activeController.signal }
      );
      if (!response.ok) throw new Error(`search returned ${response.status}`);
      const body = await response.json();
      const items = (body.tracks || []).filter(item => ID_RE.test(String(item.id)));
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

  function onType() {
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
      suggestDelayMs
    );
  }

  function onKey(event) {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      if (!acItems.length) return;
      event.preventDefault();
      const count = acItems.length;
      acSel =
        event.key === "ArrowDown" ? (acSel + 1) % count : (acSel - 1 + count) % count;
      renderAc();
    } else if (event.key === "Enter") {
      const item = acSel >= 0 ? acItems[acSel] : acItems.find(song => song.has_preview);
      if (item) {
        event.preventDefault();
        pick(item);
      }
    } else if (event.key === "Escape") {
      hideAc();
    }
  }

  function pick(song) {
    if (!song || !song.has_preview || !ID_RE.test(String(song.id))) return;
    hideAc();
    input.value = `${song.title} — ${song.artist}`;
    input.blur();
    const id = Number(song.id);
    cards.set(id, { id, title: song.title, artist: song.artist, cover: safeImg(song.cover) });
    openId(id);
  }

  // ---------- links ----------
  // #id=<Deezer id>&tab=<tab> is the share link. #s=<title — artist> is the
  // older home-page link, still used by the example chips.
  function writeHash(id, push) {
    const hash = `#id=${id}&tab=${tab}`;
    if (location.hash === hash) return false;
    if (push) location.hash = hash;
    else history.replaceState(null, "", hash);
    return true;
  }

  function openId(id) {
    if (!writeHash(id, true)) load({ id }); // otherwise the hashchange handler loads it
  }

  function openName(title, artist) {
    input.value = `${title} — ${artist}`;
    const known = resolved.get(songKey({ title, artist }));
    if (known) {
      openId(known);
      return;
    }
    const hash = `#s=${encodeURIComponent(`${title} — ${artist}`)}`;
    if (location.hash === hash) load({ title, artist });
    else location.hash = hash;
  }

  function splitSeed(text) {
    for (const separator of [" — ", " – ", " - ", " by "]) {
      const index = text.indexOf(separator);
      if (index > 0) {
        return [text.slice(0, index).trim(), text.slice(index + separator.length).trim()];
      }
    }
    return [text.trim(), ""];
  }

  function readHash() {
    const hash = location.hash || "";
    const named = hash.match(/^#s=(.+)$/);
    if (named) {
      let text = "";
      try {
        text = decodeURIComponent(named[1]);
      } catch {}
      const [title, artist] = splitSeed(text);
      return { target: title ? { title, artist } : null, tab: null };
    }
    const params = new URLSearchParams(hash.slice(1));
    const id = params.get("id") || "";
    return {
      target: ID_RE.test(id) ? { id: Number(id) } : null,
      tab: tabKey(params.get("tab")),
    };
  }

  const targetKey = target => (target.id ? `id:${target.id}` : `s:${songKey(target)}`);

  // /anysong forwards its query string, so /?id=<id>&tab=<tab> opens that song.
  function adoptQueryLink() {
    const params = new URLSearchParams(location.search);
    const id = params.get("id");
    if (id === null) return;
    const wanted = tabKey(params.get("tab"));
    params.delete("id");
    params.delete("tab");
    const rest = params.toString();
    const keepHash = !ID_RE.test(id) || (location.hash && location.hash !== "#");
    const hash = keepHash ? location.hash : `#id=${id}&tab=${wanted || tab}`;
    history.replaceState(null, "", `${location.pathname}${rest ? `?${rest}` : ""}${hash}`);
  }

  function clearResults() {
    loadSequence += 1;
    currentKey = null;
    shownPick = null;
    stopPreview();
    out.innerHTML = "";
    lastResults = [];
    lastSeed = null;
    if (document.activeElement !== input) input.value = "";
  }

  function route() {
    let { target, tab: wanted } = readHash();
    if (wanted && wanted !== tab) selectTab(wanted, false);
    if (target && !target.id) {
      const known = resolved.get(songKey(target));
      if (known) {
        history.replaceState(null, "", `#id=${known}&tab=${tab}`);
        target = { id: known };
      }
    }
    if (target) {
      if (targetKey(target) !== currentKey) load(target);
    } else if (currentKey !== null) {
      clearResults();
    }
  }

  // ---------- results ----------
  async function fetchBestMatches(target, isCurrent, lang) {
    // The policy keeps HTTP-cached responses from before a format change out.
    const params = new URLSearchParams(
      target.id
        ? { id: String(target.id) }
        : { title: target.title, artist: target.artist }
    );
    params.set("k", "20");
    params.set("lang", lang);
    params.set("policy", bestMatchesPolicy);
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

  // A link that names only a title opens the first search hit with a preview.
  async function searchFirst(text) {
    try {
      const response = await fetch(
        `/api/anysong/search?limit=8&q=${encodeURIComponent(text.slice(0, 200))}`
      );
      const body = await response.json();
      const song = (body.tracks || []).find(
        item => item.has_preview && ID_RE.test(String(item.id))
      );
      if (!song) return null;
      const id = Number(song.id);
      cards.set(id, { id, title: song.title, artist: song.artist, cover: safeImg(song.cover) });
      return id;
    } catch {
      return null;
    }
  }

  function errorText(error) {
    const status = error && error.status;
    const message = (error && error.message) || "";
    if (status === 400) return "Pick a song from the suggestions, or type its title and artist.";
    if (status === 404) {
      return message
        ? `${message.charAt(0).toUpperCase()}${message.slice(1)}.`
        : "We couldn't find that song on Deezer.";
    }
    if (status === 429) {
      return "You're going a bit fast. Wait a few seconds and try again.";
    }
    if (status === 503) return "The matcher is busy right now. Try again in a moment.";
    if (status === 504) return "That took too long. Try again.";
    if (status === 502) return "Deezer didn't respond. Try again in a moment.";
    if (status === 422) return "We couldn't decode this song's preview.";
    if (status === 0) return "Network error. Check your connection and try again.";
    return "Something went wrong. Try again.";
  }

  function setState(html, extra = "") {
    out.innerHTML = `<div class="state${extra ? ` ${extra}` : ""}">${html}</div>`;
  }

  async function load(target) {
    const sequence = ++loadSequence;
    const isCurrent = () => sequence === loadSequence;
    currentKey = targetKey(target);
    shownPick = null;
    stopPreview();
    const known = target.id ? cards.get(target.id) : target;
    setState(
      `<span class="spin"></span> Listening to ${
        known && known.title ? esc(known.title) : "the song"
      }…`
    );
    const slow = window.setTimeout(() => {
      if (isCurrent() && !shownPick) {
        setState(
          '<span class="spin"></span> Matching… songs that are new to us are analysed live, which takes a few seconds.'
        );
      }
    }, 1600);
    try {
      let id = target.id || null;
      if (!id && !target.artist) {
        id = await searchFirst(target.title);
        if (!isCurrent()) return;
        if (!id) {
          setState(esc(`We couldn't find “${target.title}” on Deezer.`), "err");
          return;
        }
      }
      const picked = {
        target: id ? { id } : target,
        seed: null,
        lang: "same",
        languageSequence: 0,
        hasVibe: false,
        best: { status: "pending" },
        body: null,
      };
      let body;
      try {
        body = await fetchBestMatches(picked.target, isCurrent, picked.lang);
      } catch (error) {
        if (isCurrent()) setState(esc(errorText(error)), "err");
        return;
      }
      if (!isCurrent()) return;
      picked.best = { status: "ready", body };
      picked.body = body;
      const query = body.query || {};
      if (Number.isInteger(query.id) && query.id > 0) id = query.id;
      const card = (id && cards.get(id)) || {};
      const seed = {
        id,
        title: query.title || card.title || target.title || "",
        artist: query.artist || card.artist || target.artist || "",
        cover: safeImg(query.cover || card.cover || ""),
      };
      picked.seed = seed;
      if (id) {
        picked.target = { id };
        cards.set(id, seed);
        if (!target.id) resolved.set(songKey(target), id);
        currentKey = `id:${id}`;
      }
      if (document.activeElement !== input) {
        input.value = seed.title ? `${seed.title} — ${seed.artist}` : "";
      }
      showPick(picked);
      if (id) writeHash(id, false);
      if (!seed.cover && id) {
        fetchPreview(id).then(preview => {
          const cover = safeImg((preview && preview.cover) || "");
          const box = out.querySelector(".seed .cover");
          if (!cover || shownPick !== picked || !box) return;
          box.innerHTML = `<img alt="" src="${esc(cover)}">`;
          seed.cover = cover;
        });
      }
    } finally {
      window.clearTimeout(slow);
    }
  }

  const soundsList = results =>
    Array.isArray(results.sounds) && results.sounds.length ? "sounds" : "sonic";

  function listRows(body, key) {
    const results = body.results || {};
    const name = key === "sounds" ? soundsList(results) : key;
    const rows = (Array.isArray(results[name]) ? results[name] : []).filter(
      x => Number.isInteger(x.id) && x.id > 0
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
        id: x.id,
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

  const rowHtml = (x, index) =>
    `<li class="row" data-id="${x.id}">` +
    `<button class="pv" type="button" aria-label="Preview ${esc(x.title)} by ${esc(x.artist)}" title="Preview" data-id="${x.id}">▶</button>` +
    `<div class="rank">${index + 1}</div>` +
    `<div class="meta"><button class="t" type="button" data-id="${x.id}" title="Find songs like this">${esc(x.title)}</button><div class="a">${esc(x.artist)}${
      x.why ? `<span class="why"> · ${esc(x.why)}</span>` : ""
    }</div></div>` +
    `<div class="bar" aria-hidden="true"><i style="width:${x.width}%"></i></div>` +
    `<a class="play" href="${esc(x.url)}" target="_blank" rel="noopener" aria-label="Open ${esc(x.title)} by ${esc(x.artist)} in Spotify">Open in Spotify ▸</a></li>`;

  function seedTags(picked) {
    const body = picked.body;
    if (!body) return [];
    const query = body.query || {};
    const total = (body.timings_ms || {}).total;
    const bpm = typeof query.bpm === "number" ? Math.round(query.bpm) : 0;
    const live = query.source === "live";
    return [
      [live ? (body.cached ? "heard live earlier" : "heard live just now") : "in our library", live],
      [tab !== "best" && bpm > 0 ? `${bpm} BPM` : "", false],
      [!body.cached && total ? `matched in ${(total / 1000).toFixed(1)} s` : "", false],
    ].filter(([text]) => text);
  }

  function blurbText(picked) {
    const entry = resultTabs.find(candidate => candidate.key === tab);
    if (tab === "sounds" && picked.body && soundsList(picked.body.results || {}) === "sonic") {
      return sonicBlurb;
    }
    return entry.blurb;
  }

  function languageName(code) {
    const names = {
      de: "German",
      en: "English",
      es: "Spanish",
      fr: "French",
      ja: "Japanese",
      ko: "Korean",
      pt: "Portuguese",
      zh: "Chinese",
    };
    if (names[code]) return names[code];
    try {
      return new Intl.DisplayNames(["en"], { type: "language" }).of(code);
    } catch {
      return code.toUpperCase();
    }
  }

  // Best matches, Same vibe and the fused Sounds like hide confidently
  // other-language songs unless the listener asks for all languages; Same
  // melody, Samples & loops and the raw sound channel are not filtered. The
  // hidden count is Best's own, so the others only say what they leave out.
  function languageFiltered(picked) {
    if (tab === "best" || tab === "vibe") return true;
    return (
      tab === "sounds" &&
      (!picked.body || soundsList(picked.body.results || {}) === "sounds")
    );
  }

  function renderLanguage(picked) {
    const box = document.querySelector("#language");
    const button = document.querySelector("#language-toggle");
    if (!box || !button) return;
    box.hidden = !languageFiltered(picked) || picked.best.status === "failed";
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
        : tab !== "best"
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
    fetchBestMatches(picked.target, isCurrent, picked.lang)
      .then(
        body => ({ status: "ready", body }),
        error => ({ status: "failed", error })
      )
      .then(entry => {
        if (picked.languageSequence !== sequence) return;
        picked.best = entry;
        if (entry.status === "ready") picked.body = entry.body;
        renderTab(picked);
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

  function markTabs() {
    document.querySelectorAll("#out .tab").forEach(button => {
      const selected = button.dataset.tab === tab;
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
    if (tab === "vibe" && !picked.hasVibe) {
      tab = "best";
      markTabs();
      if (picked.seed && picked.seed.id) writeHash(picked.seed.id, false);
    }
  }

  function renderTab(picked) {
    const list = document.querySelector("#matches");
    if (shownPick !== picked || !list) return;
    syncVibeTab(picked);
    const entry = picked.best;
    const rows = entry.status === "ready" ? listRows(entry.body, tab) : [];
    lastResults = rows;
    list.dataset.tab = tab;
    list.setAttribute("aria-labelledby", `tab-${tab}`);
    document.querySelector("#seedtags").innerHTML = seedTags(picked)
      .map(([text, live]) => `<span class="tag${live ? " live" : ""}">${esc(text)}</span>`)
      .join("");
    document.querySelector("#matchblurb").textContent = blurbText(picked);
    renderLanguage(picked);
    if (entry.status === "pending") {
      list.innerHTML = '<li class="state"><span class="spin"></span> matching…</li>';
    } else if (entry.status === "failed") {
      list.innerHTML = `<li class="state err">${esc(errorText(entry.error))}</li>`;
    } else if (!rows.length) {
      list.innerHTML = '<li class="state">No matches in this list.</li>';
    } else {
      list.innerHTML = rows.map(rowHtml).join("");
    }
    list.querySelectorAll(".pv").forEach(button => {
      button.onclick = () => togglePreview(Number(button.dataset.id), button);
    });
    list.querySelectorAll(".t").forEach(button => {
      button.onclick = () => openRow(rows.find(row => row.id === Number(button.dataset.id)));
    });
    syncPlaying();
    resetListActions(rows.length);
  }

  function openRow(row) {
    if (!row) return;
    cards.set(row.id, { id: row.id, title: row.title, artist: row.artist, cover: "" });
    input.value = `${row.title} — ${row.artist}`;
    window.scrollTo({ top: 0, behavior: "smooth" });
    openId(row.id);
  }

  function selectTab(key, updateHash) {
    if (key === tab && document.querySelector("#matches")?.dataset.tab === key) {
      return;
    }
    tab = key;
    if (updateHash && shownPick && shownPick.seed.id) writeHash(shownPick.seed.id, false);
    markTabs();
    if (shownPick) renderTab(shownPick);
  }

  function showPick(picked) {
    const seed = picked.seed;
    const results = picked.body.results || {};
    shownPick = picked;
    lastSeed = seed;
    if (tab === "best" && !Array.isArray(results.best)) tab = "melody";
    out.innerHTML =
      '<section class="seed" aria-label="Selected song">' +
      `<div class="cover">${seed.cover ? `<img alt="" src="${esc(seed.cover)}">` : ""}</div>` +
      `<div class="info"><h2>${esc(seed.title)}</h2><div class="art">${esc(seed.artist)}</div><div class="tags" id="seedtags"></div></div>` +
      '<div class="actions">' +
      (seed.id
        ? `<button class="pv" type="button" data-id="${seed.id}" aria-label="Preview ${esc(seed.title)}" title="Preview">▶</button>`
        : "") +
      '<button class="copy" id="share" type="button">🔗 Share</button><button class="copy" id="copy" type="button">⧉ Copy list</button><button class="save" id="save" type="button">＋ Save to Spotify</button></div></section>' +
      `<div class="tabs" role="tablist" aria-label="Kind of match">${resultTabs
        .map(
          entry =>
            `<button class="tab" type="button" role="tab" id="tab-${entry.key}" data-tab="${entry.key}" aria-controls="matches" aria-selected="false" tabindex="-1">${esc(entry.label)}</button>`
        )
        .join("")}</div>` +
      '<div class="language-filter" id="language" role="status" hidden><span id="language-status"></span><button id="language-toggle" type="button" aria-pressed="false" hidden></button></div>' +
      '<p class="blurb" id="matchblurb"></p><ol id="matches" role="tabpanel"></ol>';
    document.querySelector("#copy").onclick = copyList;
    document.querySelector("#share").onclick = share;
    document.querySelector("#language-toggle").onclick = () => toggleLanguage(picked);
    const seedPreview = out.querySelector(".seed .pv");
    if (seedPreview) seedPreview.onclick = () => togglePreview(seed.id, seedPreview);
    const tabs = [...out.querySelectorAll(".tab")];
    tabs.forEach(button => {
      button.onclick = () => selectTab(button.dataset.tab, true);
      button.onkeydown = event => {
        if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
        event.preventDefault?.();
        const shown = tabs.filter(candidate => !candidate.hidden);
        const step = event.key === "ArrowRight" ? 1 : shown.length - 1;
        const next = shown[(shown.indexOf(button) + step) % shown.length];
        next.focus();
        selectTab(next.dataset.tab, true);
      };
    });
    markTabs();
    renderTab(picked);
  }

  async function share() {
    const button = document.querySelector("#share");
    try {
      if (
        navigator.share &&
        window.matchMedia &&
        window.matchMedia("(pointer: coarse)").matches
      ) {
        await navigator.share({ title: document.title, url: location.href });
        return;
      }
      await navigator.clipboard.writeText(location.href);
      button.textContent = "✓ Link copied";
      window.setTimeout(() => {
        button.textContent = "🔗 Share";
      }, 2000);
    } catch (error) {
      if (error && error.name === "AbortError") return;
      try {
        window.prompt("Copy this link:", location.href);
      } catch {}
    }
  }

  // ---------- previews ----------
  const audio = new Audio();
  audio.preload = "none";
  let playingId = null;
  const previewCache = new Map();

  async function fetchPreview(id) {
    const hit = previewCache.get(id);
    if (hit && Date.now() - hit.at < 5 * 60 * 1000) return hit.data;
    try {
      const response = await fetch(`/api/preview?id=${encodeURIComponent(id)}`);
      const body = await response.json();
      if (!response.ok || !body.ok) return null;
      previewCache.set(id, { at: Date.now(), data: body });
      return body;
    } catch {
      return null;
    }
  }

  function syncPlaying() {
    out.querySelectorAll(".pv").forEach(button => {
      const on =
        playingId !== null && Number(button.dataset.id) === playingId && !audio.paused;
      button.classList.toggle("playing", on);
      button.textContent = on ? "⏸" : "▶";
      button.closest("li.row")?.classList.toggle("playing", on);
    });
  }

  function stopPreview() {
    audio.pause();
    playingId = null;
    syncPlaying();
  }

  async function togglePreview(id, button) {
    if (playingId === id) {
      stopPreview();
      return;
    }
    stopPreview();
    playingId = id;
    button.textContent = "…";
    const preview = await fetchPreview(id);
    if (playingId !== id) return;
    try {
      if (!preview || !preview.preview) throw new Error("no preview");
      audio.src = preview.preview;
      await audio.play();
      syncPlaying();
    } catch {
      if (playingId !== id) return;
      playingId = null;
      syncPlaying();
      button.textContent = "✕";
      window.setTimeout(() => {
        if (button.textContent === "✕") button.textContent = "▶";
      }, 1500);
    }
  }

  audio.addEventListener("ended", stopPreview);
  audio.addEventListener("pause", syncPlaying);

  // ---------- examples ----------
  function surpriseMe() {
    let index;
    do {
      index = Math.floor(Math.random() * SURPRISE.length);
    } while (index === lastSurprise && SURPRISE.length > 1);
    lastSurprise = index;
    openName(SURPRISE[index][0], SURPRISE[index][1]);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function loadChips() {
    const chips = document.querySelector("#chips");
    if (!chips) return;
    chips.innerHTML =
      '<button class="chip surprise" id="surprise" type="button">🎲 Surprise me</button>' +
      DEMO.map(
        (song, index) =>
          `<button class="chip" type="button" data-i="${index}">${esc(song[0])} <b>· ${esc(song[1])}</b></button>`
      ).join("");
    document.querySelector("#surprise").onclick = surpriseMe;
    chips.querySelectorAll("button[data-i]").forEach(button => {
      button.onclick = () => {
        const song = DEMO[Number(button.dataset.i)];
        openName(song[0], song[1]);
      };
    });
  }

  input.addEventListener("input", onType);
  input.addEventListener("keydown", onKey);
  input.addEventListener("blur", () => window.setTimeout(hideAc, 150));
  ac.addEventListener("mousedown", event => event.preventDefault());
  window.addEventListener("hashchange", route);
  loadChips();
  adoptQueryLink();
  route();
})();
