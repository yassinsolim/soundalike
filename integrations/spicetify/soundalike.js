// soundalike — Spicetify extension
// Adds a right-click "Find soundalikes" item to any track in the Spotify
// desktop client. It prefers the optional local soundalike server and otherwise
// uses the hosted library, then shows the results on a navigable Spotify page.
//
// Install (requires a patchable desktop Spotify app): follow this directory's
// README for Marketplace or manual setup.

(function soundalike() {
  const LOCAL_SERVER = "http://127.0.0.1:8787";
  const PRIMARY_HOSTED_SERVER = "https://soundalike-api.yassin.app";
  const FALLBACK_HOSTED_SERVER = "https://soundalike.yassin.app";
  const LOCAL_PROBE_TIMEOUT_MS = 250;
  const PRIMARY_HOSTED_TIMEOUT_MS = 5000;
  const FALLBACK_HOSTED_TIMEOUT_MS = 65000;
  const LOCAL_STATUS_TTL_MS = 30000;
  const CACHE_KEY = "soundalike:spicetify-cache:v6";
  const LEGACY_CACHE_KEYS = [
    "soundalike:spicetify-cache:v2",
    "soundalike:spicetify-cache:v3",
    "soundalike:spicetify-cache:v4",
    "soundalike:spicetify-cache:v5",
  ];
  const CACHE_TTL_MS = 7 * 24 * 60 * 60 * 1000;
  const MAX_RECOMMENDATION_CACHE_SIZE = 50;
  const MAX_SPOTIFY_CACHE_SIZE = 500;
  const HOSTED_API_VERSION = "4";
  const LANGUAGE_POLICY = "spotify-lyrics-strict-v2";
  const RECOMMENDATION_POOL_SIZE = 50;
  const DISPLAY_RESULT_COUNT = 20;
  const SPOTIFY_TRACK_SEARCH_LIMIT = 20;
  const SPOTIFY_TRACK_SEARCH_HASH =
    "59ee4a659c32e9ad894a71308207594a65ba67bb6b632b183abe97303a51fa55";
  const RESULTS_PATH = "/soundalike";
  let localStatus = { available: false, checkedAt: 0 };
  let nativeContextChain;
  let warnedAboutNativeMenus = false;
  let resultsState = null;
  let activePage = null;
  let routeMountTimer;
  let cacheSaveTimer;
  const pendingRecommendations = new Map();
  const pendingSpotifyTracks = new Map();
  let persistentCache;
  let fallbackTrackSearchDefinition;

  // Wait until the Spicetify APIs we need are ready.
  if (!(
    window.Spicetify &&
    Spicetify.ContextMenu &&
    Spicetify.GraphQL?.Definitions?.getTrack &&
    Spicetify.GraphQL?.Request &&
    Spicetify.Platform?.History?.listen &&
    Spicetify.React?.createElement &&
    Spicetify.ReactDOM?.createRoot &&
    Spicetify.ReactJSX?.jsx &&
    Spicetify.URI
  )) {
    setTimeout(soundalike, 400);
    return;
  }
  if (window.__soundalikeContextMenuItem) return;
  removeLegacyCaches();
  persistentCache = loadPersistentCache();

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

  async function getCacheableHostedRecommendations(
    server,
    payload,
    timeoutMs,
    apiVersion = HOSTED_API_VERSION,
    languagePolicy = LANGUAGE_POLICY
  ) {
    const values = {
      query: payload.query,
      n: String(payload.n),
      diversity: String(payload.diversity),
      v: apiVersion,
    };
    if (languagePolicy) values.language_policy = languagePolicy;
    const params = new URLSearchParams(values);
    const response = await fetchWithTimeout(
      `${server}/api/spicetify_recommend?${params}`,
      { cache: "default" },
      timeoutMs
    );
    let result;
    try {
      result = await response.json();
    } catch {
      throw new Error(`Recommendation service returned HTTP ${response.status}.`);
    }
    if (!response.ok && response.status !== 422) {
      const error = new Error(
        result?.error || `Recommendation service returned HTTP ${response.status}.`
      );
      error.status = response.status;
      throw error;
    }
    if (!response.ok && !result?.error) {
      throw new Error(`Recommendation service returned HTTP ${response.status}.`);
    }
    return { data: result, cacheable: true };
  }

  async function getHostedRecommendations(payload) {
    try {
      return await getCacheableHostedRecommendations(
        PRIMARY_HOSTED_SERVER,
        payload,
        PRIMARY_HOSTED_TIMEOUT_MS
      );
    } catch (error) {
      if (error?.status === 400) {
        try {
          console.warn(
            "[soundalike] Primary hosted endpoint uses the compatibility contract.",
            error
          );
          return await getCacheableHostedRecommendations(
            PRIMARY_HOSTED_SERVER,
            payload,
            PRIMARY_HOSTED_TIMEOUT_MS,
            "3",
            "spotify-lyrics-v1"
          );
        } catch (compatibilityError) {
          console.warn(
            "[soundalike] Primary compatibility request failed; using Vercel fallback.",
            compatibilityError
          );
        }
      } else {
        console.warn(
          "[soundalike] Primary hosted endpoint failed; using Vercel fallback.",
          error
        );
      }
    }
    try {
      return await getCacheableHostedRecommendations(
        FALLBACK_HOSTED_SERVER,
        payload,
        FALLBACK_HOSTED_TIMEOUT_MS
      );
    } catch (error) {
      console.warn(
        "[soundalike] Cacheable Vercel endpoint failed; using legacy endpoint.",
        error
      );
      return getLegacyHostedRecommendations(payload);
    }
  }

  async function getLegacyHostedRecommendations(payload) {
    return {
      data: await postRecommendations(
        FALLBACK_HOSTED_SERVER,
        payload,
        FALLBACK_HOSTED_TIMEOUT_MS,
        true
      ),
      cacheable: false,
    };
  }

  async function requestRecommendations(payload, cacheId) {
    const cached = readCacheEntry("recommendations", cacheId);
    if (cached) {
      return { data: cached.data, source: cached.source, cached: true };
    }
    if (pendingRecommendations.has(cacheId)) {
      return pendingRecommendations.get(cacheId);
    }
    const request = requestRecommendationsUncached(payload)
      .then((recommendation) => {
        if (recommendation.data?.ok && recommendation.cacheable !== false) {
          writeCacheEntry("recommendations", cacheId, recommendation);
        }
        return recommendation;
      })
      .finally(() => pendingRecommendations.delete(cacheId));
    pendingRecommendations.set(cacheId, request);
    return request;
  }

  async function requestRecommendationsUncached(payload) {
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
      "Using hosted Soundalike.",
      false,
      5000
    );
    const hosted = await getHostedRecommendations(payload);
    return { data: hosted.data, source: "hosted", cacheable: hosted.cacheable };
  }

  async function findSoundalikes(uris) {
    const id = uris[0].split(":track:")[1];
    Spicetify.showNotification("Finding soundalikes…");
    let data;
    let seedTrack;
    try {
      const seedUri = `spotify:track:${id}`;
      seedTrack = readCacheEntry("spotifyTracks", seedUri);
      if (!seedTrack) {
        const metadata = await Spicetify.GraphQL.Request(
          Spicetify.GraphQL.Definitions.getTrack,
          { uri: seedUri }
        );
        seedTrack = metadata?.data?.trackUnion;
        if (seedTrack) writeCacheEntry("spotifyTracks", seedUri, compactSpotifyTrack(seedTrack));
      }
      const artist = seedTrack?.firstArtist?.items?.[0]?.profile?.name;
      if (!seedTrack?.name || !artist) {
        throw new Error("Spotify did not return track metadata.");
      }
      const languagePromise = Object.hasOwn(seedTrack, "soundalikeLanguage")
        ? Promise.resolve(seedTrack.soundalikeLanguage)
        : getSpotifyTrackLanguage(seedUri);
      const recommendationPromise = requestRecommendations({
        query: `${seedTrack.name} — ${artist}`,
        n: RECOMMENDATION_POOL_SIZE,
        diversity: 0.15,
      }, `${LANGUAGE_POLICY}:${seedUri}`);
      const [recommendation, seedLanguage] = await Promise.all([
        recommendationPromise,
        languagePromise,
      ]);
      seedTrack = {
        ...seedTrack,
        uri: seedTrack.uri || seedUri,
        soundalikeLanguage: seedLanguage,
      };
      writeCacheEntry("spotifyTracks", seedUri, compactSpotifyTrack(seedTrack));
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
    showResultsPage(data, seedTrack);
  }

  async function findSpotifyTrack(result) {
    const cacheId = `result:${normalizeLabel(result.title)}::${normalizeLabel(result.artist)}`;
    const cached = readCacheEntry("spotifyTracks", cacheId);
    if (cached) return cached;
    if (pendingSpotifyTracks.has(cacheId)) return pendingSpotifyTracks.get(cacheId);
    const request = findSpotifyTrackUncached(result)
      .then((track) => {
        if (track) {
          writeCacheEntry("spotifyTracks", cacheId, compactSpotifyTrack(track));
        }
        return track;
      })
      .finally(() => pendingSpotifyTracks.delete(cacheId));
    pendingSpotifyTracks.set(cacheId, request);
    return request;
  }

  async function findSpotifyTrackUncached(result) {
    const searchTerm = `${result.title} ${result.artist}`;
    const response = await Spicetify.GraphQL.Request(
      Spicetify.GraphQL.Definitions.searchModalResults,
      {
        searchTerm,
        offset: 0,
        limit: 5,
        numberOfTopResults: 5,
        includeAudiobooks: false,
        includeAuthors: false,
      }
    );
    let match = bestSpotifyTrackMatch(
      response?.data?.searchV2?.topResultsV2?.itemsV2,
      result
    );
    const trackSearchDefinition = getSpotifyTrackSearchDefinition();
    if (!match && trackSearchDefinition) {
      try {
        const trackResponse = await Spicetify.GraphQL.Request(
          trackSearchDefinition,
          {
            searchTerm,
            offset: 0,
            limit: SPOTIFY_TRACK_SEARCH_LIMIT,
            numberOfTopResults: 5,
            includeAudiobooks: false,
            includeAuthors: false,
          }
        );
        match = bestSpotifyTrackMatch(
          trackResponse?.data?.searchV2?.tracksV2?.items,
          result
        );
      } catch (error) {
        console.warn(
          `[soundalike] Spotify song search failed for ${searchTerm}.`,
          error
        );
      }
    }
    if (!match) return null;
    const languagePromise = getSpotifyTrackLanguage(match.uri);
    let resolvedTrack = match;
    try {
      const details = await Spicetify.GraphQL.Request(
        Spicetify.GraphQL.Definitions.getTrack,
        { uri: match.uri }
      );
      resolvedTrack = mergeSpotifyTrackDetails(match, details?.data?.trackUnion);
    } catch (error) {
      console.warn(
        `[soundalike] Spotify action metadata lookup failed for ${match.uri}`,
        error
      );
    }
    return {
      ...resolvedTrack,
      soundalikeLanguage: await languagePromise,
    };
  }

  function bestSpotifyTrackMatch(hits, result) {
    const ranked = (hits || [])
      .map((hit) => hit?.item?.data || hit?.data || hit)
      .filter((track) => track?.__typename === "Track")
      .filter((track) => spotifyArtistMatchesExactly(track, result))
      .map((track) => ({ track, score: spotifyMatchScore(track, result) }))
      .sort((a, b) => b.score - a.score);
    return ranked[0]?.score >= 6 ? ranked[0].track : null;
  }

  function getSpotifyTrackSearchDefinition() {
    if (Spicetify.GraphQL.Definitions.searchTracks) {
      return Spicetify.GraphQL.Definitions.searchTracks;
    }
    if (fallbackTrackSearchDefinition) return fallbackTrackSearchDefinition;
    const Definition =
      Spicetify.GraphQL.Definitions.searchModalResults?.constructor;
    if (typeof Definition !== "function" || Definition === Object) return null;
    try {
      fallbackTrackSearchDefinition = new Definition(
        "searchTracks",
        "query",
        SPOTIFY_TRACK_SEARCH_HASH,
        null
      );
      return fallbackTrackSearchDefinition;
    } catch (error) {
      console.warn(
        "[soundalike] Spotify song search is unavailable in this client.",
        error
      );
      return null;
    }
  }

  async function getSpotifyTrackLanguage(uri) {
    const trackId = uri?.split(":track:")[1];
    if (!trackId || !Spicetify.CosmosAsync?.get) return null;
    try {
      const response = await Spicetify.CosmosAsync.get(
        `https://spclient.wg.spotify.com/color-lyrics/v2/track/${trackId}` +
        "?format=json&vocalRemoval=false&market=from_token"
      );
      const language = String(response?.lyrics?.language || "")
        .trim()
        .toLowerCase()
        .split("-")[0];
      return /^[a-z]{2,3}$/.test(language) ? language : null;
    } catch (error) {
      console.debug?.(
        `[soundalike] Lyrics language is unavailable for ${uri}.`,
        error
      );
      return null;
    }
  }

  function loadPersistentCache() {
    try {
      const raw = Spicetify.LocalStorage?.get?.(CACHE_KEY);
      const parsed = raw ? JSON.parse(raw) : null;
      if (parsed?.recommendations && parsed?.spotifyTracks) return parsed;
    } catch (error) {
      console.warn("[soundalike] Could not load the local result cache.", error);
    }
    return { recommendations: {}, spotifyTracks: {} };
  }

  function removeLegacyCaches() {
    for (const key of LEGACY_CACHE_KEYS) {
      try {
        Spicetify.LocalStorage?.remove?.(key);
      } catch (error) {
        console.warn(`[soundalike] Could not remove legacy cache ${key}.`, error);
      }
    }
  }

  function readCacheEntry(bucket, key) {
    const entry = persistentCache[bucket]?.[key];
    if (!entry) return null;
    if (Date.now() - entry.cachedAt > CACHE_TTL_MS) {
      delete persistentCache[bucket][key];
      scheduleCacheSave();
      return null;
    }
    entry.lastUsedAt = Date.now();
    return entry.value;
  }

  function writeCacheEntry(bucket, key, value) {
    persistentCache[bucket][key] = {
      value,
      cachedAt: Date.now(),
      lastUsedAt: Date.now(),
    };
    const limit = bucket === "recommendations"
      ? MAX_RECOMMENDATION_CACHE_SIZE
      : MAX_SPOTIFY_CACHE_SIZE;
    const entries = Object.entries(persistentCache[bucket]);
    if (entries.length > limit) {
      entries
        .sort(([, left], [, right]) => left.lastUsedAt - right.lastUsedAt)
        .slice(0, entries.length - limit)
        .forEach(([staleKey]) => delete persistentCache[bucket][staleKey]);
    }
    scheduleCacheSave();
  }

  function scheduleCacheSave() {
    if (!Spicetify.LocalStorage?.set) return;
    clearTimeout(cacheSaveTimer);
    cacheSaveTimer = setTimeout(() => {
      try {
        Spicetify.LocalStorage.set(CACHE_KEY, JSON.stringify(persistentCache));
      } catch (error) {
        console.warn("[soundalike] Could not save the local result cache.", error);
      }
    }, 100);
  }

  function compactSpotifyTrack(track) {
    const compactArtists = (items) => (items || [])
      .map((artist) => ({
        uri: artist?.uri,
        profile: { name: artist?.profile?.name },
      }))
      .filter((artist) => artist.uri || artist.profile.name);
    const firstArtist = compactArtists(track.firstArtist?.items);
    const otherArtists = compactArtists(track.otherArtists?.items);
    const artists = compactArtists(track.artists?.items);
    const coverSources = (track.albumOfTrack?.coverArt?.sources || [])
      .map((source) => ({
        url: source?.url,
        width: source?.width,
        height: source?.height,
      }))
      .filter((source) => source.url);
    return {
      __typename: track.__typename,
      name: track.name,
      uri: track.uri,
      albumOfTrack: {
        name: track.albumOfTrack?.name,
        uri: track.albumOfTrack?.uri,
        coverArt: {
          sources: coverSources,
        },
      },
      firstArtist: { items: firstArtist },
      otherArtists: { items: otherArtists },
      artists: { items: artists.length ? artists : [...firstArtist, ...otherArtists] },
      soundalikeLanguage: track.soundalikeLanguage ?? null,
    };
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
    if (!expectedTitle || !expectedArtist) return 0;
    let score = 0;
    if (actualTitle === expectedTitle) {
      score += 4;
    } else if (actualTitle.includes(expectedTitle) || expectedTitle.includes(actualTitle)) {
      score += 2;
    }
    if (spotifyArtistMatchesExactly(track, result)) score += 4;
    return score;
  }

  function spotifyArtistMatchesExactly(track, result) {
    const expectedArtist = normalizeLabel(result.artist);
    const actualArtists = new Set(
      (track.artists?.items || [])
        .map((item) => normalizeLabel(item?.profile?.name))
        .filter(Boolean)
    );
    if (!expectedArtist || !actualArtists.size) return false;
    if (actualArtists.has(expectedArtist)) return true;
    const hasCollaborationMarker =
      /&|\b(?:and|feat\.?|featuring|versus|vs\.?|with|x)\b/i.test(
        String(result.artist || "")
      );
    if (!hasCollaborationMarker) return false;
    let remainder = ` ${expectedArtist} `;
    let matchedArtists = 0;
    const longestFirst = [...actualArtists].sort((left, right) =>
      right.length - left.length
    );
    for (const artist of longestFirst) {
      const exactName = ` ${artist} `;
      if (!remainder.includes(exactName)) continue;
      remainder = remainder.replace(exactName, " ");
      matchedArtists++;
    }
    const collaborationWords = new Set([
      "and", "feat", "featuring", "versus", "vs", "with", "x",
    ]);
    const unmatched = normalizeLabel(remainder)
      .split(" ")
      .filter((word) => word && !collaborationWords.has(word));
    return matchedArtists > 1 && unmatched.length === 0;
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

  function findNativeContextChain(container = activePage?.container) {
    if (nativeContextChain) return nativeContextChain;
    const candidates = [];
    for (let element = container; element; element = element.parentElement) {
      candidates.push(element);
    }
    if (typeof document.querySelector === "function") {
      for (const selector of [
        '[data-testid="now-playing-bar"]',
        '[data-testid="topbar-content"]',
        "main",
        "[data-testid]",
      ]) {
        const element = document.querySelector(selector);
        if (element) candidates.push(element);
      }
    }

    for (const element of candidates) {
      const fiberKey = element && Object.keys(element).find(
        (key) => key.startsWith("__reactFiber$") || key.startsWith("__reactInternalInstance$")
      );
      let fiber = fiberKey ? element[fiberKey] : null;
      const contextChain = [];
      const providers = new Set();
      while (fiber) {
        if (fiber.tag === 10 && fiber.elementType && !providers.has(fiber.elementType)) {
          providers.add(fiber.elementType);
          contextChain.push({
            Provider: fiber.elementType,
            value: fiber.memoizedProps?.value,
          });
        }
        fiber = fiber.return;
      }
      if (contextChain.length) {
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
    if (!(
      React?.createElement &&
      React?.Suspense &&
      components?.RightClickMenu &&
      components?.TrackMenu
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

    return React.createElement(
      React.Suspense,
      {
        // RightClickMenu uses the menu element's URI props to inject registered items.
        uri: track.uri,
        fallback: React.createElement(
          "div",
          { className: "sa-menu-loading" },
          "Loading track actions..."
        ),
      },
      React.createElement(components.TrackMenu, menuProps)
    );
  }

  function withNativeContext(element) {
    const contextChain = findNativeContextChain();
    if (!contextChain) {
      if (!warnedAboutNativeMenus) {
        warnedAboutNativeMenus = true;
        console.warn(
          "[soundalike] Native Spotify track menus are unavailable; direct playback remains enabled."
        );
      }
      return null;
    }
    for (const { Provider, value } of contextChain) {
      element = Spicetify.React.createElement(Provider, { value }, element);
    }
    return element;
  }

  function spotifyArtists(track, fallback) {
    const artists = (track?.artists?.items || [])
      .map((item) => item?.profile?.name)
      .filter(Boolean);
    return artists.length ? artists.join(", ") : fallback;
  }

  function spotifyAlbum(track) {
    return track?.albumOfTrack?.name || "\u2014";
  }

  function formatBpm(value) {
    const bpm = Number.parseFloat(String(value ?? ""));
    return Number.isFinite(bpm) && bpm > 0 ? String(Math.round(bpm)) : "\u2014";
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
  }

  function renderResultRow(row, result, index, track = null, rank = index + 1) {
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
      row.ondblclick = () => activateResult(result, track);
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
          ? `Double-click to play ${result.title} by ${artist}`
          : `Double-click to search Spotify for ${result.title} by ${result.artist}`,
        "aria-label": track?.uri
          ? `Play ${result.title} by ${artist}`
          : `Search Spotify for ${result.title} by ${result.artist}`,
        onDoubleClick: activate,
        onKeyDown: (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            activate();
          }
        },
      },
      React.createElement(
        "div",
        { className: "sa-leading" },
        React.createElement("span", { className: "sa-rank" }, rank),
        React.createElement(
          "button",
          {
            className: "sa-play",
            type: "button",
            title: track?.uri
              ? `Play ${result.title}`
              : `Search Spotify for ${result.title}`,
            "aria-label": track?.uri
              ? `Play ${result.title}`
              : `Search Spotify for ${result.title}`,
            onClick: (event) => {
              event.stopPropagation();
              activate();
            },
          },
          track?.uri ? "\u25B6" : "\u203A"
        )
      ),
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
      React.createElement("div", { className: "sa-album" }, spotifyAlbum(track)),
      React.createElement("div", { className: "sa-bpm" }, formatBpm(result.bpm))
    );
    const menu = nativeTrackMenu(track);
    const interactiveRow = menu
      ? React.createElement(Spicetify.ReactComponent.RightClickMenu, { menu }, child)
      : child;
    row.__soundalikeRoot.render(menu ? withNativeContext(interactiveRow) || child : child);
  }

  function languageName(code) {
    const names = {
      ar: "Arabic", de: "German", en: "English", es: "Spanish", fr: "French",
      hi: "Hindi", it: "Italian", ja: "Japanese", ko: "Korean", nl: "Dutch",
      pl: "Polish", pt: "Portuguese", ru: "Russian", sv: "Swedish",
      tr: "Turkish", uk: "Ukrainian", zh: "Chinese",
    };
    return names[code] || code?.toUpperCase() || "Unknown";
  }

  function applyLanguageGate(page, results, tracks, seedLanguage) {
    const eligible = [];
    let sameLanguage = 0;
    let unavailable = 0;
    let differentLanguage = 0;
    for (let index = 0; index < results.length; index++) {
      const language = tracks[index]?.soundalikeLanguage || null;
      if (!seedLanguage || language === seedLanguage) {
        if (eligible.length < DISPLAY_RESULT_COUNT) eligible.push(index);
        if (language === seedLanguage) sameLanguage++;
      } else if (!language) {
        unavailable++;
      } else {
        differentLanguage++;
      }
    }
    const visible = new Set(eligible);
    page.querySelectorAll(".sa-row").forEach((row) => {
      row.hidden = !visible.has(Number(row.dataset.index));
    });
    eligible.forEach((index, displayIndex) => {
      const row = page.querySelector(`.sa-row[data-index="${index}"]`);
      if (row) {
        renderResultRow(
          row,
          results[index],
          index,
          tracks[index] || null,
          displayIndex + 1
        );
      }
    });
    const status = page.querySelector(".sa-language-status");
    if (!status) return;
    if (!seedLanguage) {
      status.textContent = "Language gate unavailable - preserving normal ranking";
      return;
    }
    status.textContent = `${languageName(seedLanguage)} lyrics · ` +
      `${sameLanguage} exact ${sameLanguage === 1 ? "match" : "matches"}` +
      `${differentLanguage ? ` · ${differentLanguage} different hidden` : ""}` +
      `${unavailable ? ` · ${unavailable} unavailable hidden` : ""}`;
    if (!eligible.length) {
      Spicetify.showNotification(
        `No exact ${languageName(seedLanguage)}-language matches were found.`,
        true
      );
    }
  }

  async function hydrateSpotifyRows(results, page, tracks, seedLanguage) {
    if (!Spicetify.GraphQL.Definitions.searchModalResults) {
      console.warn("[soundalike] Spotify artwork lookup is unavailable in this client.");
      applyLanguageGate(page, results, tracks, seedLanguage);
      return;
    }
    let cursor = 0;
    let enriched = 0;
    async function worker() {
      while (cursor < results.length) {
        const index = cursor++;
        if (index in tracks) {
          const cachedRow = page.querySelector(`.sa-row[data-index="${index}"]`);
          if (cachedRow) renderResultRow(cachedRow, results[index], index, tracks[index]);
          continue;
        }
        try {
          const track = await findSpotifyTrack(results[index]);
          tracks[index] = track;
          const row = page.querySelector(`.sa-row[data-index="${index}"]`);
          if (!track || !row || page !== activePage?.view) continue;
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
    const workerCount = Math.min(8, results.length);
    await Promise.all(Array.from({ length: workerCount }, worker));
    if (page === activePage?.view) {
      applyLanguageGate(page, results, tracks, seedLanguage);
    }
    console.log(
      `[soundalike] added Spotify metadata to ${enriched}/${results.length} rows.`
    );
  }

  function buildResultsPage(data, seedTrack, tracks, seedLanguage) {
    const s = data.seed, v = data.vibe;
    const wrap = document.createElement("div");
    const seedImage = artworkUrl(seedTrack);
    const source = data.__soundalikeSource === "local"
      ? "LOCAL ENGINE"
      : "HOSTED LIBRARY";
    wrap.className = "sa-results";

    const languageStatus = seedLanguage
      ? `${languageName(seedLanguage)} lyrics · checking recommendations`
      : "Language gate unavailable - preserving normal ranking";
    const modelLabel = data.method === "dual_sonic64_guardrail"
      ? "Dual-Sonic64"
      : "Production model";
    const tags = [modelLabel, "V5 strict language", v.tempo, v.dynamics, v.low_end, v.tone]
      .map((t) => `<span class="sa-tag">${esc(t)}</span>`)
      .join("") +
      `<span class="sa-tag sa-language-status">${esc(languageStatus)}</span>`;
    const rows = data.results
      .map(
        (x, i) => `
      <div class="sa-row" data-index="${i}" data-q="${esc(x.title + " " + x.artist)}"${i >= DISPLAY_RESULT_COUNT ? " hidden" : ""}>
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
        .sa-results{box-sizing:border-box;min-height:100%;padding:32px clamp(24px,5vw,64px) 56px;font:14px var(--encore-body-font-stack,system-ui,sans-serif);color:var(--spice-text,#fff)}
        .sa-seed{display:flex;align-items:center;gap:18px;margin:0 0 18px}
        .sa-seed img,.sa-seed-fallback{width:96px;height:96px;border-radius:6px;object-fit:cover;background:#282828;box-shadow:0 8px 24px rgba(0,0,0,.35)}
        .sa-seed-fallback{display:grid;place-items:center;color:#b3b3b3;font-size:22px}
        .sa-kicker{color:var(--spice-subtext,#b3b3b3);font-size:12px;margin-bottom:2px}
        .sa-seed-title{font-size:clamp(24px,4vw,40px);font-weight:800;line-height:1.1}
        .sa-seed-artist{color:var(--spice-subtext,#b3b3b3);margin-top:2px}
        .sa-tags{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px}
        .sa-tag{background:rgba(255,255,255,.08);border-radius:999px;padding:4px 9px;color:var(--spice-subtext,#b3b3b3);font-size:12px}
        .sa-list{padding-right:4px}
        .sa-list-head,.sa-row-content{display:grid;grid-template-columns:32px 44px minmax(180px,1.6fr) minmax(120px,1fr) 64px;align-items:center;column-gap:12px}
        .sa-list-head{height:28px;color:var(--spice-subtext,#b3b3b3);font-size:12px;border-bottom:1px solid rgba(255,255,255,.1);margin-bottom:4px}
        .sa-row{border-radius:6px}
        .sa-row-content{min-height:56px;padding:4px 6px;border-radius:6px;cursor:pointer;outline:none}
        .sa-row-content:hover,.sa-row-content:focus-visible{background:rgba(255,255,255,.1)}
        .sa-leading{width:32px;height:32px;display:grid;place-items:center}
        .sa-rank{grid-area:1/1;text-align:center;color:var(--spice-subtext,#b3b3b3);font-variant-numeric:tabular-nums}
        .sa-cover{width:44px;height:44px;border-radius:4px;background:#282828;display:grid;place-items:center;color:#727272;overflow:hidden}
        .sa-cover img{width:100%;height:100%;object-fit:cover}
        .sa-meta{min-width:0}
        .sa-title{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
        .sa-artist{color:var(--spice-subtext,#b3b3b3);font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:2px}
        .sa-album{min-width:0;color:var(--spice-subtext,#b3b3b3);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
        .sa-bpm{color:var(--spice-subtext,#b3b3b3);font-variant-numeric:tabular-nums;text-align:right}
        .sa-play{grid-area:1/1;display:none;width:30px;height:30px;border:0;border-radius:50%;background:transparent;color:var(--spice-text,#fff);font-size:15px;cursor:pointer}
        .sa-row-content:hover .sa-rank,.sa-row-content:focus-within .sa-rank{display:none}
        .sa-row-content:hover .sa-play,.sa-play:focus-visible{display:grid;place-items:center}
        .sa-play:hover,.sa-play:focus-visible{background:var(--spice-button,#1ed760);color:#000;outline:none}
        .sa-row-content:hover .sa-title,.sa-row-content:focus-visible .sa-title{color:var(--spice-text,#fff)}
        .sa-menu-loading{padding:8px 12px;color:var(--spice-subtext,#b3b3b3);white-space:nowrap}
        @media(max-width:760px){
          .sa-list-head,.sa-row-content{grid-template-columns:32px 44px minmax(0,1fr) 56px}
          .sa-album,.sa-list-head .sa-album-head{display:none}
        }
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
      <div class="sa-list-head"><div>#</div><div></div><div>Title</div><div class="sa-album-head">Album</div><div class="sa-bpm">BPM</div></div>
      <div class="sa-list">${rows}</div>`;

    wrap.querySelectorAll(".sa-row").forEach((row, index) => {
      renderResultRow(row, data.results[index], index, tracks[index] || null);
    });
    return wrap;
  }

  function teardownResultsPage() {
    clearTimeout(routeMountTimer);
    routeMountTimer = undefined;
    if (!activePage) return;
    activePage.view?.querySelectorAll(".sa-row").forEach((row) => {
      row.__soundalikeRoot?.unmount?.();
    });
    activePage.resizeObserver?.disconnect?.();
    if (activePage.syncHost) {
      window.removeEventListener?.("resize", activePage.syncHost);
    }
    activePage.host?.remove?.();
    activePage = null;
    nativeContextChain = undefined;
  }

  function createResultsHost(container, view) {
    const host = document.createElement("div");
    host.className = "sa-results-host";
    Object.assign(host.style, {
      position: "fixed",
      zIndex: "10",
      overflow: "auto",
      background: "var(--spice-main, #121212)",
    });
    const syncHost = () => {
      const bounds = container.getBoundingClientRect?.();
      if (!bounds) return;
      host.style.left = `${bounds.left}px`;
      host.style.top = `${bounds.top}px`;
      host.style.width = `${bounds.width}px`;
      host.style.height = `${bounds.height}px`;
    };
    syncHost();
    host.appendChild(view);
    document.body.appendChild(host);
    window.addEventListener?.("resize", syncHost);
    const resizeObserver = typeof ResizeObserver === "function"
      ? new ResizeObserver(syncHost)
      : null;
    resizeObserver?.observe?.(container);
    return { host, syncHost, resizeObserver };
  }

  function renderResultsPage(container) {
    if (!resultsState || !container || !document.body) return;
    teardownResultsPage();
    const view = buildResultsPage(
      resultsState.data,
      resultsState.seedTrack,
      resultsState.tracks,
      resultsState.seedLanguage
    );
    const mounted = createResultsHost(container, view);
    activePage = { container, view, ...mounted };
    nativeContextChain = undefined;
    hydrateSpotifyRows(
      resultsState.data.results,
      view,
      resultsState.tracks,
      resultsState.seedLanguage
    ).catch((error) => {
      console.error("[soundalike] Spotify result enrichment failed", error);
    });
  }

  function mountResultsRoute() {
    if (Spicetify.Platform.History.location?.pathname !== RESULTS_PATH) return;
    const container = document.querySelector(
      '[data-testid="main-view-container"]'
    ) || document.querySelector("main");
    if (!container) {
      clearTimeout(routeMountTimer);
      routeMountTimer = setTimeout(mountResultsRoute, 100);
      return;
    }
    renderResultsPage(container);
  }

  function scheduleResultsRoute() {
    clearTimeout(routeMountTimer);
    routeMountTimer = setTimeout(mountResultsRoute, 0);
  }

  function onHistoryChange(location) {
    const path = location?.pathname || Spicetify.Platform.History.location?.pathname;
    if (path !== RESULTS_PATH) {
      teardownResultsPage();
      return;
    }
    scheduleResultsRoute();
  }

  function showResultsPage(data, seedTrack) {
    resultsState = {
      data,
      seedTrack,
      seedLanguage: seedTrack?.soundalikeLanguage || null,
      tracks: [],
    };
    Spicetify.Platform.History.push({
      pathname: RESULTS_PATH,
      state: { soundalike: true },
    });
    scheduleResultsRoute();
  }

  function esc(str) {
    return String(str || "").replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  Spicetify.Platform.History.listen(onHistoryChange);

  window.__soundalikeContextMenuItem = new Spicetify.ContextMenu.Item(
    "Find soundalikes",
    findSoundalikes,
    onlyTracks,
    "enhance" // Spicetify built-in icon
  );
  window.__soundalikeContextMenuItem.register();

  function prewarmFallbackRecommender() {
    if (!window.requestIdleCallback) return;
    window.requestIdleCallback(async () => {
      const params = new URLSearchParams({
        query: "Blinding Lights — The Weeknd",
        n: "1",
        diversity: "0",
        v: HOSTED_API_VERSION,
        language_policy: LANGUAGE_POLICY,
        warm: "1",
      });
      try {
        await fetchWithTimeout(
          `${FALLBACK_HOSTED_SERVER}/api/spicetify_recommend?${params}`,
          { cache: "no-store" },
          FALLBACK_HOSTED_TIMEOUT_MS
        );
      } catch (error) {
        console.debug?.(
          "[soundalike] Vercel fallback prewarm was unavailable.",
          error
        );
      }
    }, { timeout: 5000 });
  }

  prewarmFallbackRecommender();

  console.log("[soundalike] extension loaded — right-click a track to try it.");
})();
