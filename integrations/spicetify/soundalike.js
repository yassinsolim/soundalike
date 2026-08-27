// soundalike — Spicetify extension
// Adds a right-click "Find soundalikes" item to any track in the Spotify
// desktop client. It prefers the optional local soundalike server and otherwise
// uses the hosted library, then shows the results on a navigable Spotify page.
//
// Install (requires a patchable desktop Spotify app): follow this directory's
// README for Marketplace or manual setup.

(function soundalike() {
  const RUNTIME_SEMANTIC_VERSION = "2.1.1";
  const LOCAL_SERVER = "http://127.0.0.1:8787";
  const PRIMARY_HOSTED_SERVER = "https://soundalike-api.yassin.app";
  const FALLBACK_HOSTED_SERVER = "https://soundalike.yassin.app";
  const LOCAL_PROBE_TIMEOUT_MS = 250;
  const PRIMARY_HOSTED_TIMEOUT_MS = 5000;
  const FALLBACK_HOSTED_TIMEOUT_MS = 65000;
  const FEEDBACK_TIMEOUT_MS = 10000;
  const LOCAL_STATUS_TTL_MS = 30000;
  const CACHE_KEY = "soundalike:spicetify-cache:v9";
  const LEGACY_CACHE_KEYS = [
    "soundalike:spicetify-cache:v8",
    "soundalike:spicetify-cache:v2",
    "soundalike:spicetify-cache:v3",
    "soundalike:spicetify-cache:v4",
    "soundalike:spicetify-cache:v5",
    "soundalike:spicetify-cache:v6",
    "soundalike:spicetify-cache:v7",
  ];
  const CACHE_TTL_MS = 7 * 24 * 60 * 60 * 1000;
  const MAX_RECOMMENDATION_CACHE_SIZE = 50;
  const MAX_SPOTIFY_CACHE_SIZE = 500;
  const HOSTED_API_VERSION = "4";
  const LANGUAGE_POLICY = "spotify-lyrics-strict-v2";
  const RANKING_POLICY = "model-quality-v1";
  const FEEDBACK_SELECTION_POLICY =
    "top-20-strict-language-related-artist-model-quality-v1";
  const DISPLAY_RESULT_COUNT = 20;
  const RECOMMENDATION_POOL_SIZE = DISPLAY_RESULT_COUNT;
  const LANGUAGE_LOOKUP_ATTEMPTS = 2;
  const LANGUAGE_RETRY_DELAY_MS = 350;
  const RELATED_ARTIST_TIMEOUT_MS = 1500;
  const SPOTIFY_ENRICH_WORKERS = 4;
  const SPOTIFY_TRACK_SEARCH_LIMIT = 20;
  const SPOTIFY_TRACK_SEARCH_HASH =
    "59ee4a659c32e9ad894a71308207594a65ba67bb6b632b183abe97303a51fa55";
  const RESULTS_PATH = "/soundalike";
  const FEEDBACK_ENDPOINT =
    `${FALLBACK_HOSTED_SERVER}/api/spicetify-feedback`;
  const FEEDBACK_INSTALL_KEY = "soundalike:feedback-install:v1";
  const FEEDBACK_PREFERENCE_KEY = "soundalike:feedback-preference:v2";
  const LEGACY_FEEDBACK_SUPPRESSION_KEY = "soundalike:feedback-suppression:v1";
  const FEEDBACK_METHODS = new Set([
    "dual_sonic64_guardrail",
    "sonic64_stable_head",
    "legacy_no_sonic_seed",
  ]);
  let localStatus = { available: false, checkedAt: 0 };
  let pendingLocalProbe;
  let nativeContextChain;
  let warnedAboutNativeMenus = false;
  let resultsState = null;
  let activePage = null;
  let routeMountTimer;
  let cacheSaveTimer;
  const pendingRecommendations = new Map();
  const pendingSpotifyTracks = new Map();
  const pendingRelatedArtists = new Map();
  const relatedArtistCache = new Map();
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
  // The signed Marketplace bootstrap may be executed more than once during
  // client navigation. Never register a second context-menu item.
  if (window.__soundalikeRuntimeVersion || window.__soundalikeContextMenuItem) return;
  window.__soundalikeRuntimeVersion = RUNTIME_SEMANTIC_VERSION;
  removeLegacyCaches();
  removeLegacyFeedbackSuppression();
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

  function anonymousNonce() {
    if (typeof globalThis.crypto?.getRandomValues !== "function") return null;
    const bytes = new Uint8Array(16);
    globalThis.crypto.getRandomValues(bytes);
    return Array.from(bytes, (value) => value.toString(16).padStart(2, "0"))
      .join("");
  }

  function nowMs() {
    return globalThis.performance?.now?.() ?? Date.now();
  }

  function elapsedMs(startedAt) {
    return Math.max(0, Math.round(nowMs() - startedAt));
  }

  function feedbackInstallNonce() {
    const existing = Spicetify.LocalStorage.get(FEEDBACK_INSTALL_KEY);
    if (/^[a-f0-9]{32}$/.test(existing || "")) return existing;
    const created = anonymousNonce();
    if (created) Spicetify.LocalStorage.set(FEEDBACK_INSTALL_KEY, created);
    return created;
  }

  function removeLegacyFeedbackSuppression() {
    Spicetify.LocalStorage.remove(LEGACY_FEEDBACK_SUPPRESSION_KEY);
  }

  function feedbackPreference() {
    const saved = Spicetify.LocalStorage.get(FEEDBACK_PREFERENCE_KEY);
    if (!saved) return { dismissals: 0, showAgain: null };
    try {
      const value = JSON.parse(saved);
      return {
        dismissals: Number.isInteger(value?.dismissals) && value.dismissals > 0
          ? Math.min(value.dismissals, 2)
          : 0,
        showAgain: typeof value?.showAgain === "boolean" ? value.showAgain : null,
      };
    } catch (error) {
      console.warn("[soundalike] Ignoring invalid local feedback preference.", error);
      Spicetify.LocalStorage.remove(FEEDBACK_PREFERENCE_KEY);
      return { dismissals: 0, showAgain: null };
    }
  }

  function saveFeedbackPreference(preference) {
    Spicetify.LocalStorage.set(
      FEEDBACK_PREFERENCE_KEY,
      JSON.stringify(preference)
    );
  }

  function feedbackIsSuppressed() {
    return feedbackPreference().showAgain === false;
  }

  function recordFeedbackDismissal() {
    const current = feedbackPreference();
    const next = {
      ...current,
      dismissals: Math.min(current.dismissals + 1, 2),
    };
    saveFeedbackPreference(next);
    return next;
  }

  function setFeedbackShowAgain(showAgain) {
    const current = feedbackPreference();
    saveFeedbackPreference({ ...current, showAgain });
  }

  function feedbackMethod(value) {
    return FEEDBACK_METHODS.has(value) ? value : "unknown";
  }

  function feedbackIndexVersion(value) {
    return typeof value === "string" &&
        /^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$/.test(value)
      ? value
      : "unknown";
  }

  function feedbackPayload(data, displayed, installNonce, sessionNonce, form) {
    const selection = form.selection();
    const detailed = selection === "mixed" || selection === "off";
    return {
      schema_version: 1,
      survey_version: "spicetify-match-feedback-v1",
      install_nonce: installNonce,
      session_nonce: sessionNonce,
      seed: {
        title: String(data.seed?.title || ""),
        artist: String(data.seed?.artist || ""),
      },
      displayed_results: displayed.map((result, index) => ({
        position: index + 1,
        title: String(result.title || ""),
        artist: String(result.artist || ""),
      })),
      method: feedbackMethod(data.method),
      index_version: feedbackIndexVersion(data.index_version),
      api_version: ["4", "legacy", "local", "unknown"].includes(
        data.__soundalikeApiVersion
      )
        ? data.__soundalikeApiVersion
        : "unknown",
      language_policy: LANGUAGE_POLICY,
      selection_policy: FEEDBACK_SELECTION_POLICY,
      source: data.__soundalikeSource === "local" ? "local" : "hosted",
      selection,
      reasons: detailed ? form.reasons() : [],
      note: detailed ? form.note().slice(0, 280) : "",
    };
  }

  function setupFeedbackPrompt(page, data, displayed) {
    const panel = page.querySelector(".sa-feedback");
    if (
      !panel ||
      panel.dataset?.ready === "true" ||
      !displayed.length ||
      feedbackIsSuppressed()
    ) {
      return;
    }
    const installNonce = feedbackInstallNonce();
    const sessionNonce = anonymousNonce();
    if (!installNonce || !sessionNonce) {
      console.warn("[soundalike] Anonymous feedback nonce generation is unavailable.");
      return;
    }
    if (panel.dataset) panel.dataset.ready = "true";
    const ratings = Array.from(
      page.querySelectorAll('input[name="sa-feedback-rating"]')
    );
    const reasons = Array.from(
      page.querySelectorAll('input[name="sa-feedback-reason"]')
    );
    const details = page.querySelector(".sa-feedback-details");
    const note = page.querySelector(".sa-feedback-note");
    const count = page.querySelector(".sa-feedback-count");
    const question = page.querySelector(".sa-feedback-question");
    const actions = page.querySelector(".sa-feedback-actions");
    const preference = page.querySelector(".sa-feedback-preference");
    const keepShowing = page.querySelector(".sa-feedback-keep-showing");
    const stopShowing = page.querySelector(".sa-feedback-stop-showing");
    const send = page.querySelector(".sa-feedback-send");
    const dismiss = page.querySelector(".sa-feedback-dismiss");
    const status = page.querySelector(".sa-feedback-status");
    if (
      !details ||
      !note ||
      !count ||
      !question ||
      !actions ||
      !preference ||
      !keepShowing ||
      !stopShowing ||
      !send ||
      !dismiss ||
      !status
    ) {
      console.warn("[soundalike] Feedback controls are unavailable.");
      return;
    }
    const selectedRating = () => ratings.find((input) => input.checked)?.value || "";
    const selectedReasons = () =>
      reasons.filter((input) => input.checked).map((input) => input.value);
    const form = {
      selection: selectedRating,
      reasons: selectedReasons,
      note: () => String(note?.value || "").trim(),
    };
    let sending = false;

    const sync = () => {
      const selection = selectedRating();
      const detailed = selection === "mixed" || selection === "off";
      if (details) details.hidden = !detailed;
      if (!detailed) {
        reasons.forEach((input) => {
          input.checked = false;
          input.disabled = false;
        });
        if (note) note.value = "";
      }
      const selected = selectedReasons();
      reasons.forEach((input) => {
        input.disabled = selected.length >= 2 && !input.checked;
      });
      if (count) count.textContent = `${String(note?.value || "").length}/280`;
      if (send) send.disabled = sending || !selection;
    };
    ratings.forEach((input) => {
      input.onchange = () => {
        status.textContent = "";
        status.classList?.remove("sa-feedback-error");
        status.classList?.remove("sa-feedback-success");
        sync();
      };
    });
    reasons.forEach((input) => {
      input.onchange = () => {
        if (selectedReasons().length > 2) {
          input.checked = false;
          if (status) status.textContent = "Choose up to two reasons.";
        } else if (status) {
          status.textContent = "";
        }
        sync();
      };
    });
    if (note) {
      note.oninput = () => {
        if (note.value.length > 280) note.value = note.value.slice(0, 280);
        sync();
      };
    }
    dismiss.onclick = () => {
      if (sending) return;
      const saved = recordFeedbackDismissal();
      if (saved.dismissals > 1 && saved.showAgain === null) {
        question.hidden = true;
        details.hidden = true;
        actions.hidden = true;
        preference.hidden = false;
        return;
      }
      panel.hidden = true;
    };
    keepShowing.onclick = () => {
      setFeedbackShowAgain(true);
      panel.hidden = true;
    };
    stopShowing.onclick = () => {
      setFeedbackShowAgain(false);
      panel.hidden = true;
    };
    send.onclick = async () => {
      if (sending || !selectedRating()) return;
      sending = true;
      send.textContent = "Sending…";
      if (status) status.textContent = "Sending feedback…";
      sync();
      try {
        const payload = feedbackPayload(
          data,
          displayed,
          installNonce,
          sessionNonce,
          form
        );
        const response = await fetchWithTimeout(
          FEEDBACK_ENDPOINT,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          },
          FEEDBACK_TIMEOUT_MS
        );
        const receipt = await response.json();
        if (!response.ok || !/^[a-f0-9]{64}$/.test(receipt?.receipt_sha256 || "")) {
          throw new Error("feedback receipt was not accepted");
        }
        ratings.forEach((input) => {
          input.disabled = true;
        });
        reasons.forEach((input) => {
          input.disabled = true;
        });
        if (note) note.disabled = true;
        send.hidden = true;
        dismiss.hidden = true;
        if (details) details.hidden = true;
        if (status) {
          status.textContent =
            `Thanks — feedback received. Receipt ${receipt.receipt_sha256.slice(0, 12)}.`;
          status.classList?.remove("sa-feedback-error");
          status.classList?.add("sa-feedback-success");
        }
      } catch {
        console.warn("[soundalike] Feedback submission failed.");
        sending = false;
        send.textContent = "Retry feedback";
        if (status) {
          status.textContent = "Couldn’t send feedback. Check your connection and retry.";
          status.classList?.remove("sa-feedback-success");
          status.classList?.add("sa-feedback-error");
        }
        sync();
      }
    };
    panel.hidden = false;
    sync();
  }

  async function localServerReady() {
    if (Date.now() - localStatus.checkedAt < LOCAL_STATUS_TTL_MS) {
      return localStatus.available;
    }
    if (pendingLocalProbe) return pendingLocalProbe;
    pendingLocalProbe = (async () => {
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
    })().finally(() => {
      pendingLocalProbe = undefined;
    });
    return pendingLocalProbe;
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
    values.ranking_policy = RANKING_POLICY;
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
    if (result?.ok && result.ranking_policy !== RANKING_POLICY) {
      throw new Error("Recommendation service returned an outdated ranking policy.");
    }
    return { data: result, cacheable: true, apiVersion };
  }

  async function getHostedRecommendations(payload) {
    try {
      return await getCacheableHostedRecommendations(
        PRIMARY_HOSTED_SERVER,
        payload,
        PRIMARY_HOSTED_TIMEOUT_MS
      );
    } catch (error) {
      console.warn(
        "[soundalike] Primary hosted endpoint is unavailable or outdated; using Vercel fallback.",
        error
      );
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
      apiVersion: "legacy",
    };
  }

  async function requestRecommendations(payload, cacheId) {
    const cached = readCacheEntry("recommendations", cacheId);
    if (cached) {
      return {
        data: cached.data,
        source: cached.source,
        apiVersion: cached.apiVersion,
        cached: true,
      };
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
        const data = await postRecommendations(LOCAL_SERVER, payload);
        if (data?.ok && data.ranking_policy !== RANKING_POLICY) {
          throw new Error("Local engine uses an outdated ranking policy.");
        }
        return {
          data,
          source: "local",
          apiVersion: "local",
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
    return {
      data: hosted.data,
      source: "hosted",
      apiVersion: hosted.apiVersion,
      cacheable: hosted.cacheable,
    };
  }

  async function findSoundalikes(uris) {
    const startedAt = nowMs();
    const id = uris[0].split(":track:")[1];
    Spicetify.showNotification("Finding soundalikes…");
    let data;
    let seedTrack;
    let relatedArtistsPromise = Promise.resolve(new Set());
    let seedReadyAt = startedAt;
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
      seedReadyAt = nowMs();
      const cachedLanguageState = languageStateFromTrack(seedTrack);
      const languagePromise = cachedLanguageState
        ? Promise.resolve(cachedLanguageState)
        : getSpotifyTrackLanguageState(seedUri);
      relatedArtistsPromise = getRelatedArtistUris(seedTrack);
      const recommendationPromise = requestRecommendations({
        query: `${seedTrack.name} — ${artist}`,
        n: RECOMMENDATION_POOL_SIZE,
        diversity: 0.15,
      }, `${LANGUAGE_POLICY}:${RANKING_POLICY}:${seedUri}`);
      const [recommendation, seedLanguageState] = await Promise.all([
        recommendationPromise,
        languagePromise,
      ]);
      if (seedLanguageState.status === "error") {
        throw new Error("Spotify lyrics-language check failed. Please retry.");
      }
      seedTrack = {
        ...seedTrack,
        uri: seedTrack.uri || seedUri,
        soundalikeLanguage: seedLanguageState.language,
        soundalikeLanguageStatus: seedLanguageState.status,
      };
      writeCacheEntry("spotifyTracks", seedUri, compactSpotifyTrack(seedTrack));
      data = recommendation.data;
      if (data && typeof data === "object") {
        data.__soundalikeSource = recommendation.source;
        data.__soundalikeApiVersion = recommendation.apiVersion ||
          (recommendation.source === "local" ? "local" : "unknown");
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
    const initialReadyMs = elapsedMs(startedAt);
    console.log(
      `[soundalike] results page ready in ${initialReadyMs} ms ` +
      `(seed metadata ${Math.max(0, Math.round(seedReadyAt - startedAt))} ms).`
    );
    showResultsPage(data, seedTrack, relatedArtistsPromise, {
      startedAt,
      initialReadyMs,
    });
  }

  function languageStateFromTrack(track) {
    if (!track || !Object.hasOwn(track, "soundalikeLanguage")) return null;
    return {
      status: track.soundalikeLanguage ? "known" : "unavailable",
      language: track.soundalikeLanguage || null,
    };
  }

  function withLanguageState(track, state) {
    if (state.status === "error") return track;
    return {
      ...track,
      soundalikeLanguage: state.language,
      soundalikeLanguageStatus: state.status,
    };
  }

  function primaryArtistUri(track) {
    return (
      track?.firstArtist?.items?.[0]?.uri ||
      track?.artists?.items?.[0]?.uri ||
      null
    );
  }

  async function getRelatedArtistUris(seedTrack) {
    const artistUri = primaryArtistUri(seedTrack);
    const definition = Spicetify.GraphQL.Definitions.queryArtistRelated;
    if (!artistUri || !definition) return new Set();
    if (relatedArtistCache.has(artistUri)) {
      return relatedArtistCache.get(artistUri);
    }
    if (pendingRelatedArtists.has(artistUri)) {
      return pendingRelatedArtists.get(artistUri);
    }
    const lookup = Spicetify.GraphQL.Request(definition, { uri: artistUri })
      .then((response) => {
        const related = new Set();
        const items =
          response?.data?.artistUnion?.relatedContent?.relatedArtists?.items || [];
        for (const artist of items) {
          const uri = artist?.uri || (artist?.id ? `spotify:artist:${artist.id}` : "");
          if (uri) related.add(uri);
        }
        relatedArtistCache.set(artistUri, related);
        return related;
      });
    const request = Promise.race([
      lookup,
      new Promise((resolve) =>
        setTimeout(() => resolve(new Set()), RELATED_ARTIST_TIMEOUT_MS)
      ),
    ])
      .catch((error) => {
        console.warn("[soundalike] Spotify related-artist context is unavailable.", error);
        return new Set();
      })
      .finally(() => pendingRelatedArtists.delete(artistUri));
    pendingRelatedArtists.set(artistUri, request);
    return request;
  }

  async function findSpotifyTrack(result) {
    const cacheId = `result:${normalizeLabel(result.title)}::${normalizeLabel(result.artist)}`;
    if (pendingSpotifyTracks.has(cacheId)) return pendingSpotifyTracks.get(cacheId);
    const request = (async () => {
      const cached = readCacheEntry("spotifyTracks", cacheId);
      if (cached) {
        if (languageStateFromTrack(cached)) return cached;
        const languageState = await getSpotifyTrackLanguageState(cached.uri);
        const refreshed = withLanguageState(cached, languageState);
        if (languageState.status !== "error") {
          writeCacheEntry("spotifyTracks", cacheId, compactSpotifyTrack(refreshed));
        }
        return refreshed;
      }
      const track = await findSpotifyTrackUncached(result);
      if (track) {
        writeCacheEntry("spotifyTracks", cacheId, compactSpotifyTrack(track));
      }
      return track;
    })().finally(() => pendingSpotifyTracks.delete(cacheId));
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
    const languagePromise = getSpotifyTrackLanguageState(match.uri);
    let resolvedTrack = match;
    if (!hasCompleteSpotifyMetadata(match)) {
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
    }
    const languageState = await languagePromise;
    return withLanguageState({
      ...resolvedTrack,
    }, languageState);
  }

  function hasCompleteSpotifyMetadata(track) {
    const artists = track?.artists?.items || [];
    const artwork = track?.albumOfTrack?.coverArt?.sources || [];
    return Boolean(
      track?.albumOfTrack?.uri &&
      artists.some((artist) => artist?.uri && artist?.profile?.name) &&
      artwork.some((source) => source?.url)
    );
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

  async function getSpotifyTrackLanguageState(uri) {
    const trackId = uri?.split(":track:")[1];
    if (!trackId || !Spicetify.CosmosAsync?.get) {
      return { status: "unavailable", language: null };
    }
    let lastError = null;
    for (let attempt = 0; attempt < LANGUAGE_LOOKUP_ATTEMPTS; attempt++) {
      try {
        const response = await Spicetify.CosmosAsync.get(
          `https://spclient.wg.spotify.com/color-lyrics/v2/track/${trackId}` +
          "?format=json&vocalRemoval=false&market=from_token"
        );
        const language = String(response?.lyrics?.language || "")
          .trim()
          .toLowerCase()
          .split("-")[0];
        if (/^[a-z]{2,3}$/.test(language)) {
          return { status: "known", language };
        }
        lastError = null;
      } catch (error) {
        lastError = error;
      }
      if (attempt + 1 < LANGUAGE_LOOKUP_ATTEMPTS) {
        await new Promise((resolve) => setTimeout(resolve, LANGUAGE_RETRY_DELAY_MS));
      }
    }
    if (lastError) {
      console.debug?.(
        `[soundalike] Lyrics language lookup failed for ${uri}.`,
        lastError
      );
      return { status: "error", language: null };
    }
    return { status: "unavailable", language: null };
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
    const language = Object.hasOwn(track, "soundalikeLanguage")
      ? {
          soundalikeLanguage: track.soundalikeLanguage || null,
          soundalikeLanguageStatus:
            track.soundalikeLanguageStatus ||
            (track.soundalikeLanguage ? "known" : "unavailable"),
        }
      : {};
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
      ...language,
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
      React.createElement(
        "div",
        { className: "sa-album" },
        (() => {
          const albumName = track?.albumOfTrack?.name;
          if (!albumName) return "\u2014";
          const albumId = track?.albumOfTrack?.uri?.split(":album:")[1];
          if (!albumId) return albumName;
          return React.createElement(
            "a",
            {
              className: "sa-album-link",
              href: "#",
              title: "Open album in Spotify",
              onClick: (event) => {
                event.stopPropagation();
                event.preventDefault();
                Spicetify.Platform.History.push(`/album/${albumId}`);
              },
            },
            albumName
          );
        })()
      ),
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

  function trackMatchesRelatedArtist(track, relatedArtistUris) {
    if (!relatedArtistUris?.size) return false;
    return (track?.artists?.items || []).some(
      (artist) => artist?.uri && relatedArtistUris.has(artist.uri)
    );
  }

  function applyLanguageGate(
    page,
    results,
    tracks,
    seedLanguage,
    relatedArtistUris
  ) {
    const eligible = [];
    let sameLanguage = 0;
    let unavailable = 0;
    let differentLanguage = 0;
    let lookupFailures = 0;
    const qualityHeadSize = Math.min(results.length, DISPLAY_RESULT_COUNT);
    for (let index = 0; index < qualityHeadSize; index++) {
      const track = tracks[index] || null;
      const languageKnown = Object.hasOwn(track || {}, "soundalikeLanguage");
      const language = track?.soundalikeLanguage || null;
      if (!seedLanguage || language === seedLanguage) {
        eligible.push({
          index,
          related: trackMatchesRelatedArtist(track, relatedArtistUris),
        });
        if (language === seedLanguage) sameLanguage++;
      } else if (!languageKnown) {
        lookupFailures++;
      } else if (!language) {
        unavailable++;
      } else {
        differentLanguage++;
      }
    }
    eligible.sort(
      (left, right) =>
        Number(right.related) - Number(left.related) || left.index - right.index
    );
    const displayed = eligible.slice(0, DISPLAY_RESULT_COUNT);
    const relatedMatches = displayed.filter((item) => item.related).length;
    const visible = new Set(displayed.map((item) => item.index));
    page.querySelectorAll(".sa-row").forEach((row) => {
      row.hidden = !visible.has(Number(row.dataset.index));
    });
    displayed.forEach(({ index }, displayIndex) => {
      const row = page.querySelector(`.sa-row[data-index="${index}"]`);
      if (row) {
        row.style.order = String(displayIndex + 1);
        renderResultRow(
          row,
          results[index],
          index,
          tracks[index] || null,
          displayIndex + 1
        );
      }
    });
    const loading = page.querySelector(".sa-language-loading");
    if (loading) loading.hidden = true;
    if (page === activePage?.view && resultsState?.data) {
      setupFeedbackPrompt(
        page,
        resultsState.data,
        displayed.map(({ index }) => results[index])
      );
    }
    const status = page.querySelector(".sa-language-status");
    if (!status) return;
    if (!seedLanguage) {
      status.textContent = relatedMatches
        ? `Language gate unavailable · ${relatedMatches} related-artist ` +
          `${relatedMatches === 1 ? "match" : "matches"} prioritized`
        : "Language gate unavailable · preserving model order";
      return;
    }
    status.textContent = `${languageName(seedLanguage)} lyrics · ` +
      `${sameLanguage} exact ${sameLanguage === 1 ? "match" : "matches"} in top ` +
      `${qualityHeadSize}` +
      `${relatedMatches ? ` · ${relatedMatches} related prioritized` : ""}` +
      `${differentLanguage ? ` · ${differentLanguage} different hidden` : ""}` +
      `${unavailable ? ` · ${unavailable} no-lyrics hidden` : ""}` +
      `${lookupFailures ? ` · ${lookupFailures} checks failed` : ""}`;
    if (!displayed.length) {
      Spicetify.showNotification(
        `No exact ${languageName(seedLanguage)}-language matches were found in the quality head.`,
        true
      );
    }
  }

  function revealResolvedRow(page, results, tracks, index, seedLanguage) {
    const row = page.querySelector(`.sa-row[data-index="${index}"]`);
    if (!row || page !== activePage?.view) return;
    const track = tracks[index] || null;
    renderResultRow(row, results[index], index, track);
    if (!seedLanguage || track?.soundalikeLanguage === seedLanguage) {
      row.hidden = false;
      row.style.order = String(index + 1);
    }
  }

  async function hydrateSpotifyRows(
    results,
    page,
    tracks,
    seedLanguage,
    relatedArtistsPromise,
    timing
  ) {
    const startedAt = nowMs();
    const resolvedRelatedArtists = Promise.resolve(relatedArtistsPromise)
      .catch((error) => {
        console.warn("[soundalike] Related-artist ordering failed.", error);
        return new Set();
      });
    if (!Spicetify.GraphQL.Definitions.searchModalResults) {
      console.warn("[soundalike] Spotify artwork lookup is unavailable in this client.");
      const relatedArtistUris = await resolvedRelatedArtists;
      applyLanguageGate(
        page,
        results,
        tracks,
        seedLanguage,
        relatedArtistUris
      );
      return;
    }
    let cursor = 0;
    let enriched = 0;
    async function worker() {
      while (cursor < results.length) {
        const index = cursor++;
        if (index in tracks) {
          revealResolvedRow(page, results, tracks, index, seedLanguage);
          continue;
        }
        try {
          const track = await findSpotifyTrack(results[index]);
          tracks[index] = track;
          revealResolvedRow(page, results, tracks, index, seedLanguage);
          if (track) enriched++;
        } catch (error) {
          tracks[index] = null;
          revealResolvedRow(page, results, tracks, index, seedLanguage);
          console.warn(
            `[soundalike] Spotify metadata lookup failed for ${results[index].title}`,
            error
          );
        }
      }
    }
    const workerCount = Math.min(SPOTIFY_ENRICH_WORKERS, results.length);
    await Promise.all(Array.from({ length: workerCount }, worker));
    const relatedArtistUris = await resolvedRelatedArtists;
    if (page === activePage?.view) {
      applyLanguageGate(
        page,
        results,
        tracks,
        seedLanguage,
        relatedArtistUris
      );
    }
    console.log(
      `[soundalike] Spotify metadata settled in ${elapsedMs(startedAt)} ms ` +
      `(${enriched}/${results.length} rows, ` +
      `${timing ? `${elapsedMs(timing.startedAt)} ms total` : "total unavailable"}).`
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
      ? `${languageName(seedLanguage)} lyrics · checking quality head`
      : "Checking Spotify metadata and artist context";
    const modelLabel = data.method === "dual_sonic64_guardrail"
      ? "Dual-Sonic64"
      : "Production model";
    const tags = [
      modelLabel,
      "V5 strict + artist context",
      v.tempo,
      v.dynamics,
      v.low_end,
      v.tone,
    ]
      .map((t) => `<span class="sa-tag">${esc(t)}</span>`)
      .join("") +
      `<span class="sa-tag sa-language-status">${esc(languageStatus)}</span>`;
    const rows = data.results
      .map(
        (x, i) => `
      <div class="sa-row" data-index="${i}" data-q="${esc(x.title + " " + x.artist)}" hidden>
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
        .sa-list{display:flex;flex-direction:column;padding-right:4px}
        .sa-language-loading{padding:18px 6px;color:var(--spice-subtext,#b3b3b3)}
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
        .sa-album-link{color:inherit;text-decoration:none}
        .sa-album-link:hover,.sa-album-link:focus-visible{color:var(--spice-text,#fff);text-decoration:underline;outline:none}
        .sa-bpm{color:var(--spice-subtext,#b3b3b3);font-variant-numeric:tabular-nums;text-align:right}
        .sa-play{grid-area:1/1;display:none;width:30px;height:30px;border:0;border-radius:50%;background:transparent;color:var(--spice-text,#fff);font-size:15px;cursor:pointer}
        .sa-row-content:hover .sa-rank,.sa-row-content:focus-within .sa-rank{display:none}
        .sa-row-content:hover .sa-play,.sa-play:focus-visible{display:grid;place-items:center}
        .sa-play:hover,.sa-play:focus-visible{background:var(--spice-button,#1ed760);color:#000;outline:none}
        .sa-row-content:hover .sa-title,.sa-row-content:focus-visible .sa-title{color:var(--spice-text,#fff)}
        .sa-menu-loading{padding:8px 12px;color:var(--spice-subtext,#b3b3b3);white-space:nowrap}
        .sa-feedback{max-width:680px;margin:24px 0 0;padding:16px;border:1px solid rgba(255,255,255,.12);border-radius:8px;background:rgba(255,255,255,.04)}
        .sa-feedback[hidden],.sa-feedback-details[hidden],.sa-feedback-question[hidden],.sa-feedback-actions[hidden],.sa-feedback-preference[hidden]{display:none}
        .sa-feedback fieldset{border:0;margin:0;padding:0}
        .sa-feedback legend{font-weight:700;margin-bottom:10px}
        .sa-feedback-options,.sa-feedback-reasons,.sa-feedback-actions{display:flex;flex-wrap:wrap;gap:8px}
        .sa-feedback-choice,.sa-feedback-reason{display:inline-flex;align-items:center;gap:6px;min-height:36px;padding:7px 11px;border:1px solid rgba(255,255,255,.16);border-radius:999px;cursor:pointer}
        .sa-feedback-choice:focus-within,.sa-feedback-reason:focus-within{outline:2px solid var(--spice-button,#1ed760);outline-offset:2px}
        .sa-feedback-details{margin-top:14px}
        .sa-feedback-reasons{margin-bottom:12px}
        .sa-feedback-note-label{display:block;font-weight:600;margin-bottom:6px}
        .sa-feedback-note{box-sizing:border-box;width:100%;min-height:72px;resize:vertical;padding:9px 10px;border:1px solid rgba(255,255,255,.18);border-radius:6px;background:var(--spice-card,#242424);color:var(--spice-text,#fff);font:inherit}
        .sa-feedback-help{display:flex;justify-content:space-between;gap:12px;color:var(--spice-subtext,#b3b3b3);font-size:12px;margin-top:4px}
        .sa-feedback-actions{align-items:center;margin-top:14px}
        .sa-feedback-actions button{min-height:36px;border-radius:999px;padding:7px 14px;font:inherit;font-weight:700;cursor:pointer}
        .sa-feedback-send{border:0;background:var(--spice-button,#1ed760);color:#000}
        .sa-feedback-send:disabled{cursor:not-allowed;opacity:.55}
        .sa-feedback-dismiss{border:1px solid rgba(255,255,255,.2);background:transparent;color:var(--spice-text,#fff)}
        .sa-feedback-preference p{font-weight:700;margin:0 0 10px}
        .sa-feedback-preference-actions{display:flex;gap:8px}
        .sa-feedback-preference button{min-height:36px;border-radius:999px;padding:7px 14px;font:inherit;font-weight:700;cursor:pointer}
        .sa-feedback-keep-showing{border:0;background:var(--spice-button,#1ed760);color:#000}
        .sa-feedback-stop-showing{border:1px solid rgba(255,255,255,.2);background:transparent;color:var(--spice-text,#fff)}
        .sa-feedback-status{min-height:20px;color:var(--spice-subtext,#b3b3b3)}
        .sa-feedback-success{color:var(--spice-button,#1ed760)}
        .sa-feedback-error{color:#f29d9d}
        @media(max-width:760px){
          .sa-list-head,.sa-row-content{grid-template-columns:32px 44px minmax(0,1fr) 56px}
          .sa-album,.sa-list-head .sa-album-head{display:none}
          .sa-feedback-help{display:block}
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
      <div class="sa-list">
        <div class="sa-language-loading">Checking Spotify language and artist context before showing results…</div>
        ${rows}
      </div>
      <section class="sa-feedback" hidden aria-labelledby="sa-feedback-question">
        <fieldset class="sa-feedback-question">
          <legend id="sa-feedback-question">How close were these matches?</legend>
          <div class="sa-feedback-options">
            <label class="sa-feedback-choice"><input type="radio" name="sa-feedback-rating" value="good"> Good</label>
            <label class="sa-feedback-choice"><input type="radio" name="sa-feedback-rating" value="mixed"> Mixed</label>
            <label class="sa-feedback-choice"><input type="radio" name="sa-feedback-rating" value="off"> Off</label>
          </div>
        </fieldset>
        <div class="sa-feedback-details" hidden>
          <fieldset>
            <legend>What felt off? <span class="sa-kicker">(optional, choose up to two)</span></legend>
            <div class="sa-feedback-reasons">
              <label class="sa-feedback-reason"><input type="checkbox" name="sa-feedback-reason" value="style"> style</label>
              <label class="sa-feedback-reason"><input type="checkbox" name="sa-feedback-reason" value="mood_energy"> mood/energy</label>
              <label class="sa-feedback-reason"><input type="checkbox" name="sa-feedback-reason" value="tempo"> tempo</label>
              <label class="sa-feedback-reason"><input type="checkbox" name="sa-feedback-reason" value="vocals_language"> vocals/language</label>
              <label class="sa-feedback-reason"><input type="checkbox" name="sa-feedback-reason" value="instruments_timbre"> instruments/timbre</label>
            </div>
          </fieldset>
          <label class="sa-feedback-note-label" for="sa-feedback-note">Optional note</label>
          <textarea id="sa-feedback-note" class="sa-feedback-note" maxlength="280" aria-describedby="sa-feedback-privacy sa-feedback-count"></textarea>
          <div class="sa-feedback-help">
            <span id="sa-feedback-privacy">Please don’t include personal information.</span>
            <span id="sa-feedback-count" class="sa-feedback-count">0/280</span>
          </div>
        </div>
        <div class="sa-feedback-preference" hidden role="group" aria-labelledby="sa-feedback-preference-question">
          <p id="sa-feedback-preference-question">Do you want this survey to keep showing after future searches?</p>
          <div class="sa-feedback-preference-actions">
            <button class="sa-feedback-keep-showing" type="button">Yes</button>
            <button class="sa-feedback-stop-showing" type="button">No</button>
          </div>
        </div>
        <div class="sa-feedback-actions">
          <button class="sa-feedback-send" type="button" disabled>Send feedback</button>
          <button class="sa-feedback-dismiss" type="button">Not now</button>
          <span class="sa-feedback-status" role="status" aria-live="polite"></span>
        </div>
      </section>`;

    wrap.querySelectorAll(".sa-row").forEach((row, index) => {
      row.hidden = Boolean(seedLanguage);
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
      resultsState.seedLanguage,
      resultsState.relatedArtistsPromise,
      resultsState.timing
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

  function showResultsPage(data, seedTrack, relatedArtistsPromise, timing) {
    resultsState = {
      data,
      seedTrack,
      seedLanguage: seedTrack?.soundalikeLanguage || null,
      relatedArtistsPromise,
      timing,
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
  void localServerReady();

  function prewarmFallbackRecommender() {
    if (!window.requestIdleCallback) return;
    window.requestIdleCallback(async () => {
      const params = new URLSearchParams({
        query: "Blinding Lights — The Weeknd",
        n: "1",
        diversity: "0",
        v: HOSTED_API_VERSION,
        language_policy: LANGUAGE_POLICY,
        ranking_policy: RANKING_POLICY,
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
