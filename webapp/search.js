(() => {
  const input = document.querySelector("#q");
  const ac = document.querySelector("#ac");
  const out = document.querySelector("#out");
  if (!input || !ac || !out) return;

  // One search bar for any song on Deezer: suggestions come from
  // /api/anysong/search. Three tabs read one /api/anysong/similar answer, as
  // in the Spotify extension: "Best matches" (likely covers, samples and
  // shared melodies, then other artists' songs a model ranks closest from five
  // signals; shown by default), "Listeners also play" (other artists' songs
  // this song's listeners play) and "Melody match" (the melody and sample
  // matches fused into one ranking, the server's "best" list).
  const ID_RE = /^[1-9][0-9]{0,15}$/;
  const suggestDelayMs = 220;
  const maxCacheEntries = 64;
  const bestMatchesTimeoutMs = 30000;
  const bestMatchesRetries = 3;
  const bestMatchesPolicy = "main-consensus-v4";
  // Audio only asks for lists without the listener ranker (scene=0).
  const audioOnlyKey = "soundalike:audio-only:v1";
  // The tab last chosen, kept for later visits.
  const tabStoreKey = "soundalike:tab:v1";
  const listenerPickLabel = "Listeners also play";
  const listenerPickTitle = "Placed by what this song's listeners also play, not by audio alone";
  const audioOnlyTitle = "Rank by audio alone, leaving out what this song's listeners also play";
  const resultTabs = [
    {
      key: "best",
      label: "Best matches",
      blurb:
        "Likely covers, samples and shared melodies first, then other artists' songs a model ranks closest, from five signals: sound, scene, playlists, melody and samples. Turn on Audio only to rank by sound alone.",
      // Servers whose main list follows sound and the listener ranker.
      nudgedBlurb:
        "Likely covers, samples and shared melodies first, then the songs that sound most alike, nudged toward artists this song's listeners also play. Turn on Audio only to rank by sound alone.",
      audioBlurb:
        "Likely covers, samples and shared melodies first, then the songs that sound most alike.",
      // Servers without the main list send Best matches with listener picks in it.
      mixedBlurb:
        "Likely covers, samples and shared melodies first, then similar-sounding songs by artists this song's listeners also play, marked Listeners also play. Turn on Audio only to leave those out.",
    },
    {
      key: "listeners",
      label: "Listeners also play",
      blurb:
        "Songs by other artists that this song's listeners also play, one per artist. The likelier its listeners are to play a song, and the closer it sounds, the higher it ranks.",
    },
    {
      key: "melody",
      label: "Melody match",
      blurb:
        "Likely covers, samples and shared melodies first, from the melody and sample matches fused into one ranking, then similar-sounding songs by artists this song's listeners also play, marked Listeners also play. Turn on Audio only to leave those out.",
      audioBlurb:
        "Likely covers, samples and shared melodies: the melody and sample matches fused into one ranking, by audio alone.",
    },
  ];
  // Audio only and artists without listener data leave Listeners also play empty.
  const audioOnlyListenersNote =
    "Audio only is on, and this list comes from what this song's listeners also play. Turn Audio only off to see it.";
  const noListenersNote =
    "We don't know what this artist's listeners play yet, so there's no list here. Best matches still has this song's closest matches.";
  // Match a part: a server that can rank a part of the preview says so in
  // every answer (query.parts), so the control costs no request of its own and
  // stays hidden on servers without it.
  const partStepSeconds = 5;
  const partLongestSeconds = 20;
  const partDefault = [10, 20];
  // Where previews usually start in the song, for songs the server gives no place for.
  const usualPreviewOffset = 48;
  const partIntro =
    "Soundalike hears a 30-second preview. Pick the part of it you care about.";
  const partTabsNote =
    "All three tabs follow the part. Listeners also play still starts from the whole song's listeners; only how close each song sounds follows the part.";
  const partPlaceNote =
    "Song times assume the preview sits where previews usually do; in a few songs it is elsewhere.";
  const partKeysNote =
    "Drag the part or its edges. Keyboard: arrow keys move it, Shift with an arrow key changes its length.";
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
  const durations = new Map(); // Deezer id -> seconds, from search suggestions
  let timer = null;
  let controller = null;
  let requestSequence = 0;
  let tab = savedTab();
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

  // The choice outlasts the page; browsers without storage just don't keep it.
  function savedAudioOnly() {
    try {
      return window.localStorage.getItem(audioOnlyKey) === "true";
    } catch {
      return false;
    }
  }

  function saveAudioOnly(audioOnly) {
    try {
      if (audioOnly) window.localStorage.setItem(audioOnlyKey, "true");
      else window.localStorage.removeItem(audioOnlyKey);
    } catch {}
  }

  function savedTab() {
    try {
      const key = window.localStorage.getItem(tabStoreKey);
      return resultTabs.some(entry => entry.key === key) ? key : "best";
    } catch {
      return "best";
    }
  }

  function saveTab(key) {
    try {
      window.localStorage.setItem(tabStoreKey, key);
    } catch {}
  }

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

  // Links from before these tabs name Same vibe, Samples & loops or Sounds
  // like ("sounds", or "sonic" on older any-song links); they open Best
  // matches, as do unknown names. Old Same melody links open Melody match. A
  // link without a tab keeps the current one.
  const tabKey = key =>
    key === null ? null : resultTabs.some(entry => entry.key === key) ? key : "best";

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
    if (Number(song.duration) > 0) durations.set(id, Number(song.duration));
    openId(id);
  }

  // ---------- links ----------
  // #id=<Deezer id>&tab=<tab> is the share link, with &part=<start>-<end>
  // (seconds of the preview) when a part is matched. #s=<title — artist> is
  // the older home-page link, still used by the example chips.
  const partText = part => `${part[0]}-${part[1]}`;

  function parsePart(text) {
    const found = /^(\d{1,2}(?:\.\d{1,3})?)-(\d{1,2}(?:\.\d{1,3})?)$/.exec(text || "");
    if (!found) return null;
    const start = Number(found[1]);
    const end = Number(found[2]);
    return end <= 30 && end - start >= 10 ? [start, end] : null;
  }

  function writeHash(id, push, part = null) {
    const hash = `#id=${id}&tab=${tab}${part ? `&part=${partText(part)}` : ""}`;
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
    const part = parsePart(params.get("part"));
    return {
      target: ID_RE.test(id) ? { id: Number(id), ...(part ? { part } : {}) } : null,
      tab: tabKey(params.get("tab")),
    };
  }

  // A part of a song is its own page: its lists differ from the whole preview's.
  const targetKey = target =>
    target.id
      ? `id:${target.id}${target.part ? `:${partText(target.part)}` : ""}`
      : `s:${songKey(target)}`;

  // /anysong forwards its query string, so /?id=<id>&tab=<tab> opens that song
  // (and &part=<start>-<end> that part of it).
  function adoptQueryLink() {
    const params = new URLSearchParams(location.search);
    const id = params.get("id");
    if (id === null) return;
    const wanted = tabKey(params.get("tab"));
    const part = parsePart(params.get("part"));
    params.delete("id");
    params.delete("tab");
    params.delete("part");
    const rest = params.toString();
    const keepHash = !ID_RE.test(id) || (location.hash && location.hash !== "#");
    const hash = keepHash
      ? location.hash
      : `#id=${id}&tab=${wanted || tab}${part ? `&part=${partText(part)}` : ""}`;
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
  // part: [start, end] seconds of the preview to match instead of all of it.
  async function fetchBestMatches(target, isCurrent, lang, audioOnly = false, part = null) {
    // The policy keeps HTTP-cached responses from before a format change out.
    const params = new URLSearchParams(
      target.id
        ? { id: String(target.id) }
        : { title: target.title, artist: target.artist }
    );
    params.set("k", "20");
    params.set("lang", lang);
    params.set("policy", bestMatchesPolicy);
    if (audioOnly) params.set("scene", "0");
    if (part) params.set("part", partText(part));
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
      // A server without Match a part refuses the span: ask for the whole preview.
      if (params.has("part") && (response.status === 400 || response.status === 404)) {
        params.delete("part");
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
        audioOnly: savedAudioOnly(),
        // Only a link carries a part; a newly picked song starts whole.
        part: target.part || null,
        bestSequence: 0,
        best: { status: "pending" },
        body: null,
      };
      let body;
      try {
        body = await fetchBestMatches(
          picked.target, isCurrent, picked.lang, picked.audioOnly, picked.part
        );
      } catch (error) {
        if (isCurrent()) setState(esc(errorText(error)), "err");
        return;
      }
      if (!isCurrent()) return;
      picked.best = { status: "ready", body };
      picked.body = body;
      // A server without Match a part answers for the whole preview.
      if (!(body.query || {}).part) picked.part = null;
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
        currentKey = targetKey({ id, part: picked.part });
      }
      if (document.activeElement !== input) {
        input.value = seed.title ? `${seed.title} — ${seed.artist}` : "";
      }
      showPick(picked);
      if (id) writeHash(id, false, picked.part);
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

  // The Best matches tab reads the main list, or Best matches from a server
  // without it. Melody match is the server's "best" list.
  const listName = (results, key) => {
    if (key === "melody") return "best";
    if (key !== "best") return key;
    return Array.isArray(results.main) && results.main.length ? "main" : "best";
  };

  // The server's percentages: whole numbers from 0 to 100.
  const percent = value =>
    Number.isInteger(value) && value >= 0 && value <= 100 ? value : null;

  function listRows(body, key) {
    const results = body.results || {};
    const name = listName(results, key);
    const rows = (Array.isArray(results[name]) ? results[name] : []).filter(
      x => Number.isInteger(x.id) && x.id > 0
    );
    const scores = rows.map(x => Number(x.score) || 0);
    const high = Math.max(...scores);
    const low = Math.min(...scores);
    return rows.map((x, index) => {
      const title = x.title || `Deezer track ${x.id}`;
      const artist = x.artist || "";
      // Of the main list, only the audio head's rows keep the channel ranks that
      // put them there; Listeners also play rows show none.
      const ranked = name === "best" || (name === "main" && x.basis === "audio");
      const why =
        ranked && x.ranks && typeof x.ranks === "object"
          ? Object.entries(x.ranks)
              .filter(([, rank]) => Number.isInteger(rank))
              .map(([list, rank]) => `${list} #${rank}`)
              .join(" · ")
          : "";
      // A consensus row's match scores the model's ranking, not its sound.
      const overall = percent(x.overall);
      const sound = name === "main" && x.basis === "consensus" ? null : percent(x.match);
      return {
        id: x.id,
        title,
        artist,
        why,
        listenerPick: name === "best" && x.basis === "listeners",
        // The server's overall match (main list and Listeners also play) never
        // rises down the list. Without it, scores are only comparable within
        // one list, so bars are relative to it.
        width:
          overall !== null
            ? overall
            : high > low
              ? Math.round(25 + (75 * (scores[index] - low)) / (high - low))
              : 100,
        label:
          overall === null
            ? ""
            : `Match ${overall}%: how strongly Soundalike ranks it for this song` +
              (sound === null ? "" : `. Sound match ${sound}%: sounds more alike than ${sound}% of typical matches`),
        url: spotifySearchUrl(title, artist),
      };
    });
  }


  // Best matches from a server without the main list mark the rows the listener
  // ranker placed before their ranks.
  const whyHtml = x => {
    const pick = x.listenerPick
      ? ` · <span class="basis" title="${esc(listenerPickTitle)}">${esc(listenerPickLabel)}</span>`
      : "";
    return pick || x.why
      ? `<span class="why">${pick}${x.why ? ` · ${esc(x.why)}` : ""}</span>`
      : "";
  };

  const rowHtml = (x, index) =>
    `<li class="row" data-id="${x.id}">` +
    `<button class="pv" type="button" aria-label="Preview ${esc(x.title)} by ${esc(x.artist)}" title="Preview" data-id="${x.id}">▶</button>` +
    `<div class="rank">${index + 1}</div>` +
    `<div class="meta"><button class="t" type="button" data-id="${x.id}" title="Find songs like this">${esc(x.title)}</button><div class="a">${esc(x.artist)}${whyHtml(x)}</div></div>` +
    (x.label
      ? `<div class="bar" role="img" aria-label="${esc(x.label)}" title="${esc(x.label)}"><i style="width:${x.width}%"></i></div>`
      : `<div class="bar" aria-hidden="true"><i style="width:${x.width}%"></i></div>`) +
    `<a class="play" href="${esc(x.url)}" target="_blank" rel="noopener" aria-label="Open ${esc(x.title)} by ${esc(x.artist)} in Spotify">Open in Spotify ▸</a></li>`;

  function seedTags(picked) {
    const body = picked.body;
    if (!body) return [];
    const query = body.query || {};
    const total = (body.timings_ms || {}).total;
    const live = query.source === "live";
    return [
      [live ? (body.cached ? "heard live earlier" : "heard live just now") : "in our library", live],
      [!body.cached && total ? `matched in ${(total / 1000).toFixed(1)} s` : "", false],
    ].filter(([text]) => text);
  }

  function blurbText(picked) {
    const entry = resultTabs.find(candidate => candidate.key === tab);
    const body = picked.body || {};
    if (tab === "melody") {
      const soundOnly = picked.audioOnly || (body.query || {}).scene === "unknown_artist";
      return soundOnly ? entry.audioBlurb : entry.blurb;
    }
    if (tab !== "best") return entry.blurb;
    const results = body.results || {};
    const main = listName(results, "best") === "main";
    // The five-signal model placed rows; it needs no listener data.
    const consensus = main && results.main.some(x => x && x.basis === "consensus");
    // Audio only, and artists without listener data, leave the listeners out.
    if (picked.audioOnly || (!consensus && (body.query || {}).scene === "unknown_artist")) {
      return entry.audioBlurb;
    }
    return !main ? entry.mixedBlurb : consensus ? entry.blurb : entry.nudgedBlurb;
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

  // Every tab hides confidently other-language songs unless the listener asks
  // for all languages. The server's query.hidden counts them in its fused order
  // of every channel, not in the main list or Listeners also play, so the tabs
  // only say whether other languages are left out. A main-only guess
  // (query.main_language_guess) gates Best matches alone.
  function renderLanguage(picked, noListeners) {
    const button = document.querySelector("#language-toggle");
    const audio = document.querySelector("#audio-only");
    if (!button || !audio) return;
    audio.setAttribute("aria-pressed", String(picked.audioOnly));
    // After a failed reload, and on an empty Listeners also play, only Audio
    // only is left, so the switch can be undone.
    if (picked.best.status === "failed" || noListeners) {
      document.querySelector("#language-status").textContent = "";
      button.hidden = true;
      return;
    }
    if (picked.best.status !== "ready") {
      button.setAttribute("aria-pressed", String(picked.lang === "all"));
      button.textContent = picked.lang === "all" ? "Show same language" : "Show all languages";
      return;
    }
    const query = picked.best.body.query || {};
    const language = query.language || "unknown";
    const mainGuess = tab === "best" && language === "unknown" &&
      /^[a-z]{2,3}$/.test(String(query.main_language_guess || ""))
      ? String(query.main_language_guess) : "";
    const code = mainGuess || language;
    const all = query.language_mode === "all";
    const guessed = query.language_guessed === true || Boolean(mainGuess);
    document.querySelector("#language-status").textContent =
      code === "unknown"
        ? "Language uncertain; showing all languages."
        : `${languageName(code)}${guessed ? " (best guess)" : ""} · ${all ? "all languages" : "other languages left out"}`;
    button.hidden = code === "unknown";
    button.setAttribute("aria-pressed", String(all));
    button.textContent = all ? "Show same language" : "Show all languages";
  }

  function toggleLanguage(picked) {
    if (picked.best.status !== "ready") return;
    picked.lang = (picked.best.body.query || {}).language_mode === "all" ? "same" : "all";
    reloadBest(picked);
  }

  // Audio only drops the listener ranker (scene=0) and is kept for later songs.
  function toggleAudioOnly(picked) {
    picked.audioOnly = !picked.audioOnly;
    saveAudioOnly(picked.audioOnly);
    reloadBest(picked);
  }

  function reloadBest(picked) {
    const sequence = ++picked.bestSequence;
    picked.best = { status: "pending" };
    renderTab(picked);
    const isCurrent = () =>
      shownPick === picked && picked.bestSequence === sequence;
    fetchBestMatches(picked.target, isCurrent, picked.lang, picked.audioOnly, picked.part)
      .then(
        body => ({ status: "ready", body }),
        error => ({ status: "failed", error })
      )
      .then(entry => {
        if (picked.bestSequence !== sequence) return;
        picked.best = entry;
        if (entry.status === "ready") {
          picked.body = entry.body;
          if (picked.part && !(entry.body.query || {}).part) setPart(picked, null);
        }
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

  function renderTab(picked) {
    const list = document.querySelector("#matches");
    if (shownPick !== picked || !list) return;
    const entry = picked.best;
    const rows = entry.status === "ready" ? listRows(entry.body, tab) : [];
    // Audio only and artists without listener data leave Listeners also play
    // empty, so it says why instead.
    const noListeners = entry.status === "ready" && tab === "listeners" && !rows.length;
    lastResults = rows;
    list.dataset.tab = tab;
    list.setAttribute("aria-labelledby", `tab-${tab}`);
    document.querySelector("#seedtags").innerHTML = seedTags(picked)
      .map(([text, live]) => `<span class="tag${live ? " live" : ""}">${esc(text)}</span>`)
      .join("");
    // An empty Listeners also play has no ranking to describe; its note says why.
    const blurb = document.querySelector("#matchblurb");
    blurb.textContent = noListeners ? "" : blurbText(picked);
    blurb.hidden = noListeners;
    renderLanguage(picked, noListeners);
    renderPart(picked);
    if (entry.status === "pending") {
      list.innerHTML = '<li class="state"><span class="spin"></span> matching…</li>';
    } else if (entry.status === "failed") {
      list.innerHTML = `<li class="state err">${esc(errorText(entry.error))}</li>`;
    } else if (noListeners) {
      const note = picked.audioOnly ? audioOnlyListenersNote : noListenersNote;
      list.innerHTML = `<li class="state">${esc(note)}</li>`;
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

  // updateHash: the listener chose the tab, so it is also kept for next time.
  function selectTab(key, updateHash) {
    if (key === tab && document.querySelector("#matches")?.dataset.tab === key) {
      return;
    }
    tab = key;
    if (updateHash) saveTab(key);
    if (updateHash && shownPick && shownPick.seed.id) {
      writeHash(shownPick.seed.id, false, shownPick.part);
    }
    markTabs();
    if (shownPick) renderTab(shownPick);
  }

  function showPick(picked) {
    const seed = picked.seed;
    shownPick = picked;
    lastSeed = seed;
    out.innerHTML =
      '<section class="seed" aria-label="Selected song">' +
      `<div class="cover">${seed.cover ? `<img alt="" src="${esc(seed.cover)}">` : ""}</div>` +
      `<div class="info"><h2>${esc(seed.title)}</h2><div class="art">${esc(seed.artist)}</div><div class="tags" id="seedtags"></div></div>` +
      '<div class="actions">' +
      (seed.id
        ? `<button class="pv" type="button" data-id="${seed.id}" aria-label="Preview ${esc(seed.title)}" title="Preview">▶</button>`
        : "") +
      '<button class="copy" id="part-open" type="button" aria-expanded="false" aria-controls="part-panel" hidden>Match a part <span class="beta">beta</span></button>' +
      '<button class="copy" id="share" type="button">🔗 Share</button><button class="copy" id="copy" type="button">⧉ Copy list</button><button class="save" id="save" type="button">＋ Save to Spotify</button></div></section>' +
      '<section class="part" id="part-panel" aria-label="Match a part of the preview" hidden></section>' +
      '<div class="part-chip" id="part-chip" hidden></div>' +
      `<div class="tabs" role="tablist" aria-label="Kind of match">${resultTabs
        .map(
          entry =>
            `<button class="tab" type="button" role="tab" id="tab-${entry.key}" data-tab="${entry.key}" aria-controls="matches" aria-selected="false" tabindex="-1">${esc(entry.label)}</button>`
        )
        .join("")}</div>` +
      '<div class="language-filter" id="language" role="status"><span id="language-status"></span><span class="filter-actions"><button id="language-toggle" type="button" aria-pressed="false" hidden></button>' +
      `<button id="audio-only" type="button" aria-pressed="false" title="${esc(audioOnlyTitle)}">Audio only</button></span></div>` +
      '<p class="blurb" id="matchblurb"></p><ol id="matches" role="tabpanel"></ol>';
    document.querySelector("#copy").onclick = copyList;
    document.querySelector("#share").onclick = share;
    document.querySelector("#language-toggle").onclick = () => toggleLanguage(picked);
    document.querySelector("#audio-only").onclick = () => toggleAudioOnly(picked);
    document.querySelector("#part-open").onclick = () => togglePartPanel(picked);
    const seedPreview = out.querySelector(".seed .pv");
    if (seedPreview) seedPreview.onclick = () => togglePreview(seed.id, seedPreview);
    const tabs = [...out.querySelectorAll(".tab")];
    tabs.forEach(button => {
      button.onclick = () => selectTab(button.dataset.tab, true);
      button.onkeydown = event => {
        if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
        event.preventDefault?.();
        const step = event.key === "ArrowRight" ? 1 : tabs.length - 1;
        const next = tabs[(tabs.indexOf(button) + step) % tabs.length];
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
  // While a part of the seed's preview plays: the second it stops at.
  let partEnd = null;
  // The open picker: {picked, limits, start, end, peaks}; the drag in progress.
  let partDraft = null;
  let partDrag = null;
  const peaksCache = new Map(); // Deezer id -> Promise of the preview's peaks
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
    const partPlay = document.querySelector("#part-play");
    if (partPlay) partPlay.textContent = partEnd !== null ? "⏸ Stop" : "▶ Play this part";
    const head = document.querySelector("#part-head");
    if (head && partEnd === null) head.hidden = true;
  }

  function stopPreview() {
    audio.pause();
    playingId = null;
    partEnd = null;
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
      partEnd = null;
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
  audio.addEventListener("pause", () => {
    if (audio.paused) partEnd = null;
    syncPlaying();
  });
  audio.addEventListener("timeupdate", () => {
    if (partEnd === null) return;
    if (audio.currentTime >= partEnd) {
      stopPreview();
      return;
    }
    const head = document.querySelector("#part-head");
    if (!head || !partDraft) return;
    head.hidden = false;
    head.style.left = `${(100 * audio.currentTime) / partDraft.limits.total}%`;
  });

  // ---------- match a part ----------
  // The seed card's "Match a part" opens a 30-second timeline of the preview
  // with a span to drag; Apply reloads the lists for that span.
  const clock = seconds => {
    const whole = Math.max(0, Math.round(seconds));
    return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
  };

  // What the server can match: the preview's length and the shortest and
  // longest part, or null on a server without Match a part.
  function partLimits(picked) {
    const parts = ((picked.body || {}).query || {}).parts;
    if (!parts || typeof parts !== "object" || !picked.seed || !picked.seed.id) return null;
    const total = Number(parts.preview_s) || 30;
    const min = Math.max(partStepSeconds, Number(parts.min_s) || 10);
    const max = Math.min(partLongestSeconds, Number(parts.max_s) || partLongestSeconds);
    return min <= max && max <= total ? { total, min, max } : null;
  }

  // Where the preview starts in the song, in seconds, or null when unknown.
  // The server says it when it knows the song's length; a song the search
  // listed as 90 s or longer is taken to sit where previews usually do.
  function songOffset(picked) {
    const parts = ((picked.body || {}).query || {}).parts || {};
    if (Number.isFinite(parts.song_offset_s) && parts.song_offset_s >= 0) {
      return parts.song_offset_s;
    }
    return (durations.get(picked.seed.id) || 0) >= 90 ? usualPreviewOffset : null;
  }

  // "0:58–1:13 of the song", or of the preview when its place is unknown.
  function spanLabel(picked, part) {
    const offset = songOffset(picked);
    const base = offset === null ? 0 : offset;
    return `${clock(base + part[0])}–${clock(base + part[1])} of the ${
      offset === null ? "preview" : "song"
    }`;
  }

  // The seconds each model heard for the part, as the server reports them.
  function heardText(picked) {
    const heard = (((picked.body || {}).query || {}).part || {}).heard;
    if (!heard || typeof heard !== "object") return "";
    const offset = songOffset(picked) || 0;
    const spans = ["sonic", "melody", "samples"]
      .filter(name => Array.isArray(heard[name]) && heard[name].length === 2)
      .map(name =>
        `${name === "sonic" ? "sound" : name} ${clock(offset + heard[name][0])}–${clock(offset + heard[name][1])}`
      );
    return spans.length ? `Heard: ${spans.join(", ")}` : "";
  }

  // Keeps a span on the step grid, inside the preview and within the lengths.
  function fitSpan(start, end, limits) {
    const snap = value => Math.round(value / partStepSeconds) * partStepSeconds;
    const length = Math.min(limits.max, Math.max(limits.min, snap(end - start)));
    const from = Math.min(limits.total - length, Math.max(0, snap(start)));
    return [from, from + length];
  }

  function setPart(picked, part) {
    picked.part = part;
    if (shownPick !== picked || !picked.seed.id) return;
    currentKey = targetKey({ id: picked.seed.id, part });
    writeHash(picked.seed.id, false, part);
  }

  function applyPart(picked, part) {
    closePartPanel();
    setPart(picked, part);
    reloadBest(picked);
  }

  // The chip under the seed card while a part is matched, and the button that
  // opens the picker: shown only when the server's answer offers parts.
  function renderPart(picked) {
    const open = document.querySelector("#part-open");
    const chip = document.querySelector("#part-chip");
    if (!open || !chip) return;
    const limits = partLimits(picked);
    open.hidden = !limits;
    if (!limits) closePartPanel();
    chip.hidden = !picked.part;
    if (!picked.part) {
      chip.innerHTML = "";
      return;
    }
    const heard = heardText(picked);
    chip.innerHTML =
      `<span id="part-chip-text"${heard ? ` title="${esc(heard)}"` : ""}>Matching ${esc(spanLabel(picked, picked.part))}</span>` +
      ' · <button type="button" id="part-clear" title="Match the whole preview again">whole preview</button>';
    document.querySelector("#part-clear").onclick = () => applyPart(picked, null);
  }

  function closePartPanel() {
    const panel = document.querySelector("#part-panel");
    if (partEnd !== null) stopPreview();
    partDraft = null;
    if (!panel || panel.hidden) return;
    panel.hidden = true;
    panel.innerHTML = "";
    document.querySelector("#part-open")?.setAttribute("aria-expanded", "false");
  }

  function togglePartPanel(picked) {
    const panel = document.querySelector("#part-panel");
    const limits = partLimits(picked);
    if (!panel || !limits) return;
    if (!panel.hidden) {
      closePartPanel();
      return;
    }
    const [start, end] = fitSpan(...(picked.part || partDefault), limits);
    partDraft = { picked, limits, start, end, peaks: null };
    const offset = songOffset(picked);
    const base = offset === null ? 0 : offset;
    const lengths = [];
    for (let length = limits.min; length <= limits.max; length += partStepSeconds) {
      lengths.push(length);
    }
    const ticks = [];
    for (let at = 0; at <= limits.total; at += 10) ticks.push(at);
    panel.innerHTML =
      `<p class="part-intro"><span class="beta">New · beta</span> ${esc(partIntro)}</p>` +
      '<div class="part-track" id="part-track"><canvas id="part-wave" aria-hidden="true"></canvas>' +
      '<div class="part-span" id="part-span" role="slider" tabindex="0" aria-label="Part of the preview to match" aria-valuemin="0">' +
      '<i class="part-edge" data-edge="start"></i><i class="part-edge" data-edge="end"></i></div>' +
      '<i class="part-head" id="part-head" hidden></i></div>' +
      `<div class="part-ticks" aria-hidden="true">${ticks
        .map(at => `<span>${clock(base + at)}</span>`)
        .join("")}</div>` +
      '<p class="part-label" id="part-label"></p>' +
      '<div class="part-controls"><span class="part-lengths" role="group" aria-label="Length of the part">' +
      lengths
        .map(length => `<button type="button" data-length="${length}" aria-pressed="false">${length} s</button>`)
        .join("") +
      '</span><button type="button" id="part-play">▶ Play this part</button>' +
      '<button type="button" class="save" id="part-apply">Apply</button>' +
      '<button type="button" id="part-cancel">Cancel</button></div>' +
      `<p class="part-note">${esc(partTabsNote)}${offset === null ? "" : ` ${esc(partPlaceNote)}`} ${esc(partKeysNote)}</p>`;
    panel.hidden = false;
    document.querySelector("#part-open").setAttribute("aria-expanded", "true");
    document.querySelector("#part-apply").onclick = () =>
      applyPart(picked, [partDraft.start, partDraft.end]);
    document.querySelector("#part-cancel").onclick = closePartPanel;
    document.querySelector("#part-play").onclick = () => playPart(picked);
    panel.querySelectorAll(".part-lengths button").forEach(button => {
      button.onclick = () =>
        moveDraft(partDraft.start, partDraft.start + Number(button.dataset.length));
    });
    const span = document.querySelector("#part-span");
    span.onkeydown = onPartKey;
    const track = document.querySelector("#part-track");
    track.onpointerdown = onPartPointerDown;
    track.onpointermove = onPartPointerMove;
    track.onpointerup = track.onpointercancel = () => {
      partDrag = null;
    };
    // Without this a flicked touch drag ends in a fling, and the browser
    // takes the next tap (on Apply, say) as the tap that stops it.
    track.addEventListener("touchmove", event => event.preventDefault(), { passive: false });
    showDraft();
    const draft = partDraft;
    previewPeaks(picked.seed.id).then(peaks => {
      if (partDraft !== draft || !peaks) return;
      draft.peaks = peaks;
      drawWave();
    });
  }

  function moveDraft(start, end) {
    if (!partDraft) return;
    const [from, to] = fitSpan(start, end, partDraft.limits);
    if (from === partDraft.start && to === partDraft.end) return;
    if (partEnd !== null) stopPreview();
    partDraft.start = from;
    partDraft.end = to;
    showDraft();
  }

  function showDraft() {
    const span = document.querySelector("#part-span");
    if (!partDraft || !span) return;
    const { picked, limits, start, end } = partDraft;
    const label = `${spanLabel(picked, [start, end])} · ${end - start} s`;
    span.style.left = `${(100 * start) / limits.total}%`;
    span.style.width = `${(100 * (end - start)) / limits.total}%`;
    span.setAttribute("aria-valuemax", String(limits.total - limits.min));
    span.setAttribute("aria-valuenow", String(start));
    span.setAttribute("aria-valuetext", label);
    document.querySelector("#part-label").textContent = label;
    document.querySelectorAll("#part-panel .part-lengths button").forEach(button => {
      button.setAttribute("aria-pressed", String(Number(button.dataset.length) === end - start));
    });
    const same = picked.part && picked.part[0] === start && picked.part[1] === end;
    document.querySelector("#part-apply").disabled = Boolean(same);
    drawWave();
  }

  // Arrow keys move the part a step; with Shift they change its length.
  function onPartKey(event) {
    if (!partDraft) return;
    const { limits, start, end } = partDraft;
    const step =
      event.key === "ArrowRight" || event.key === "ArrowUp"
        ? partStepSeconds
        : event.key === "ArrowLeft" || event.key === "ArrowDown"
          ? -partStepSeconds
          : 0;
    if (step) {
      if (!event.shiftKey) moveDraft(start + step, end + step);
      else {
        // A longer part that would pass the end of the preview grows backwards.
        const length = Math.min(limits.max, Math.max(limits.min, end - start + step));
        const from = Math.min(start, limits.total - length);
        moveDraft(from, from + length);
      }
    } else if (event.key === "Home") moveDraft(0, end - start);
    else if (event.key === "End") moveDraft(limits.total - (end - start), limits.total);
    else return;
    event.preventDefault?.();
  }

  function pointerSeconds(event) {
    const box = document.querySelector("#part-track").getBoundingClientRect();
    return box.width > 0 ? ((event.clientX - box.left) / box.width) * partDraft.limits.total : 0;
  }

  // Dragging the span moves it and dragging an edge resizes it; a press
  // outside the span brings the span's middle there.
  function onPartPointerDown(event) {
    if (!partDraft) return;
    const at = pointerSeconds(event);
    const { start, end } = partDraft;
    const edge = event.target && event.target.dataset ? event.target.dataset.edge : "";
    const inside = at >= start && at <= end;
    partDrag = {
      mode: edge === "start" || edge === "end" ? edge : "move",
      grab: inside ? at - start : (end - start) / 2,
    };
    event.currentTarget.setPointerCapture?.(event.pointerId);
    event.preventDefault?.();
    if (!inside && partDrag.mode === "move") onPartPointerMove(event);
  }

  function onPartPointerMove(event) {
    if (!partDraft || !partDrag) return;
    const at = pointerSeconds(event);
    const { limits, start, end } = partDraft;
    const within = length => Math.min(limits.max, Math.max(limits.min, length));
    if (partDrag.mode === "move") {
      const from = at - partDrag.grab;
      moveDraft(from, from + (end - start));
    } else if (partDrag.mode === "start") {
      const from = Math.max(0, end - within(end - at));
      moveDraft(from, end);
    } else {
      const to = Math.min(limits.total, start + within(at - start));
      moveDraft(start, to);
    }
  }

  async function playPart(picked) {
    if (!partDraft) return;
    if (partEnd !== null) {
      stopPreview();
      return;
    }
    const id = picked.seed.id;
    const { start, end } = partDraft;
    const button = document.querySelector("#part-play");
    stopPreview();
    playingId = id;
    const preview = await fetchPreview(id);
    if (playingId !== id) return;
    try {
      if (!preview || !preview.preview) throw new Error("no preview");
      audio.src = preview.preview;
      audio.currentTime = start;
      partEnd = end;
      await audio.play();
      syncPlaying();
    } catch {
      if (playingId !== id) return;
      stopPreview();
      if (button) button.textContent = "✕ No preview";
    }
  }

  // The preview's loudness in 240 columns, for the timeline: Deezer's CDN lets
  // pages read the audio. Browsers without WebAudio, and failures, give null
  // and the timeline stays a plain bar.
  function previewPeaks(id) {
    if (!peaksCache.has(id)) peaksCache.set(id, loadPeaks(id).catch(() => null));
    return peaksCache.get(id);
  }

  async function loadPeaks(id) {
    const Offline = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    if (!Offline) return null;
    const preview = await fetchPreview(id);
    // The same host check as for covers: only Deezer's CDN is read.
    const url = safeImg((preview && preview.preview) || "");
    if (!url) return null;
    const response = await fetch(url);
    if (!response.ok) return null;
    const buffer = await new Offline(1, 1, 44100).decodeAudioData(await response.arrayBuffer());
    const data = buffer.getChannelData(0);
    const columns = 240;
    const size = Math.floor(data.length / columns);
    if (size < 1) return null;
    const peaks = new Float32Array(columns);
    const stride = Math.max(1, Math.floor(size / 200));
    let top = 0;
    for (let column = 0; column < columns; column += 1) {
      let peak = 0;
      for (let index = column * size; index < (column + 1) * size; index += stride) {
        const level = Math.abs(data[index]);
        if (level > peak) peak = level;
      }
      peaks[column] = peak;
      if (peak > top) top = peak;
    }
    return top > 0 ? peaks.map(peak => peak / top) : null;
  }

  function drawWave() {
    const canvas = document.querySelector("#part-wave");
    if (!partDraft || !partDraft.peaks || !canvas || !canvas.getContext) return;
    const { peaks, limits, start, end } = partDraft;
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
    const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, width, height);
    const bar = width / peaks.length;
    peaks.forEach((peak, column) => {
      const at = ((column + 0.5) / peaks.length) * limits.total;
      context.fillStyle = at >= start && at < end ? "#1db954" : "#3a4658";
      const tall = Math.max(ratio, peak * (height - 2 * ratio));
      context.fillRect(column * bar, (height - tall) / 2, Math.max(1, bar - ratio), tall);
    });
  }

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
