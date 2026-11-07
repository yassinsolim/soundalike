// soundalike — Spicetify extension
// Adds a right-click "Find soundalikes" item to any track in the Spotify
// desktop client. Results open on a navigable Spotify page with up to three
// lists: "Best matches" (the hosted any-song ranking, shown by default),
// "Same vibe" and "Sounds like" (both from the same hosted response; with the
// local companion, Sounds like comes from the local soundalike server and
// Same vibe is not shown). Playing a result plays the shown list on in
// order, like a playlist that is never saved.
//
// Install (requires a patchable desktop Spotify app): follow this directory's
// README for Marketplace or manual setup.

(function soundalike() {
  const RUNTIME_SEMANTIC_VERSION = "2.6.0";
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
  // Best matches and Sounds like entries share this bucket.
  const MAX_RECOMMENDATION_CACHE_SIZE = 100;
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
  // Spotify links this app URI to RESULTS_PATH. Results play as one context
  // under it, so the next song is the next row.
  const RESULTS_CONTEXT_URI = "spotify:app:soundalike";
  // Spotify drops songs queued just as a song starts playing.
  const LATE_QUEUE_DELAY_MS = 1500;
  const BEST_MATCHES_ENDPOINT = `${FALLBACK_HOSTED_SERVER}/api/anysong/similar`;
  const BEST_MATCHES_POLICY = "anysong-best-rrf-language-vibe-sounds-v2";
  const BEST_MATCHES_TIMEOUT_MS = 30000;
  const BEST_MATCHES_RETRIES = 2;
  const BEST_MATCHES_RETRY_DELAY_MS = 1500;
  // How long a ready Sounds like list waits for a slower Best matches list
  // before the page opens with Best matches still loading.
  const BEST_MATCHES_PATIENCE_MS = 4000;
  const RESULT_VIEWS = [
    ["best", "Best matches"],
    ["vibe", "Same vibe"],
    ["sounds", "Sounds like"],
  ];
  const BEST_MATCHES_BLURB =
    "Likely covers, samples and shared melodies first, then similar-sounding " +
    "songs by related artists from the same scene.";
  const SAME_VIBE_BLURB =
    "Songs from these matches that share this song's tempo and overall " +
    "sound, one per artist.";
  const BEST_MATCHES_OFFER =
    "Your local companion keeps searches on this computer. Best matches only " +
    "run on the hosted matcher, so getting them sends this song's title and " +
    "artist to soundalike.yassin.app.";
  const SOUNDS_LIKE_BLURB =
    "Songs with the same overall sound: low end, dynamics and tone.";
  // The hosted Sounds like list comes from the GPU engine's MuQ-MuLan model.
  const HOSTED_SOUNDS_LIKE_BLURB =
    "Songs with the same overall sound: instruments, production and mood.";
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
  // Spicetify releases before 2.46.1 can't draw extension menu items on
  // Spotify 1.3.5 or newer: once any item is registered, every right-click
  // menu opens empty.
  const MENU_FIXED_SPICETIFY_VERSION = [2, 46, 1];
  const MENU_BREAKING_SPOTIFY_VERSION = [1, 3, 5];
  let localStatus = { available: false, checkedAt: 0 };
  let pendingLocalProbe;
  let nativeContextChain;
  let warnedAboutNativeMenus = false;
  let resultsState = null;
  let activePage = null;
  // Rows matched after their list started playing: { last, uris }.
  let lateMatches = null;
  let routeMountTimer;
  let cacheSaveTimer;
  let searchSequence = 0;
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

  function delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
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

  function setupFeedbackPrompt(page, data, displayed, session = null) {
    const panel = page.querySelector(".sa-feedback");
    if (
      !panel ||
      panel.dataset?.ready === "true" ||
      session?.feedbackClosed ||
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
    // Re-rendering the page (tab switches, navigating back) must not re-ask a
    // question that was already answered or put away for this search.
    const close = () => {
      if (session) session.feedbackClosed = true;
    };

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
      close();
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
        close();
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

  function bestMatchesMessage(status) {
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

  function bestMatchesError(status, message = bestMatchesMessage(status)) {
    const error = new Error(message);
    error.status = status;
    return error;
  }

  // Keeps only what the page shows, so cached lists stay small.
  function compactBestMatches(body) {
    const query = body?.query || {};
    const best = Array.isArray(body?.results?.best) ? body.results.best : [];
    const results = best
      .map((item) => ({
        title: String(item?.title || "").trim(),
        artist: String(item?.artist || "").trim(),
        matchNote: item?.ranks && typeof item.ranks === "object"
          ? Object.entries(item.ranks)
            .filter(([list, rank]) =>
              /^[a-z]{1,24}$/.test(list) && Number.isInteger(rank) && rank > 0
            )
            .map(([list, rank]) => `${list} #${rank}`)
            .join(" · ")
          : "",
      }))
      .filter((item) => item.title && item.artist)
      .slice(0, DISPLAY_RESULT_COUNT);
    // Servers without Same vibe send no vibe list; its rows carry no ranks.
    const vibe = (Array.isArray(body?.results?.vibe) ? body.results.vibe : [])
      .map((item) => ({
        title: String(item?.title || "").trim(),
        artist: String(item?.artist || "").trim(),
      }))
      .filter((item) => item.title && item.artist)
      .slice(0, DISPLAY_RESULT_COUNT);
    // The hosted Sounds like list; servers before it send none.
    const sounds = (Array.isArray(body?.results?.sounds) ? body.results.sounds : [])
      .map((item) => ({
        title: String(item?.title || "").trim(),
        artist: String(item?.artist || "").trim(),
        bpm: Number.isFinite(item?.bpm) && item.bpm > 0 ? item.bpm : null,
      }))
      .filter((item) => item.title && item.artist)
      .slice(0, DISPLAY_RESULT_COUNT);
    const total = Number(body?.timings_ms?.total);
    return {
      query: {
        title: String(query.title || ""),
        artist: String(query.artist || ""),
        source: query.source === "live" ? "live" : "catalog",
        language: String(query.language || "unknown"),
        languageMode: query.language_mode === "all" ? "all" : "same",
        hidden: Number.isFinite(Number(query.hidden)) ? Math.max(0, Number(query.hidden)) : 0,
        bpm: Number.isFinite(query.bpm) && query.bpm > 0 ? query.bpm : null,
      },
      results,
      vibe,
      sounds,
      totalMs: Number.isFinite(total) && total > 0 ? Math.round(total) : 0,
      serverCached: body?.cached === true,
    };
  }

  async function requestBestMatchesUncached(title, artist, langMode = "same") {
    // The policy keeps HTTP-cached responses from before a format change out.
    const params = new URLSearchParams({
      title,
      artist,
      k: String(DISPLAY_RESULT_COUNT),
      lang: langMode,
      policy: BEST_MATCHES_POLICY,
    });
    for (let attempt = 0; ; attempt++) {
      let response;
      try {
        response = await fetchWithTimeout(
          `${BEST_MATCHES_ENDPOINT}?${params}`,
          { cache: "default" },
          BEST_MATCHES_TIMEOUT_MS
        );
      } catch (error) {
        throw bestMatchesError(error?.name === "AbortError" ? 504 : 0);
      }
      let body = null;
      try {
        body = await response.json();
      } catch {
        body = null;
      }
      // 503 means every analysis slot is taken; a short wait usually frees one.
      if (response.status === 503 && attempt < BEST_MATCHES_RETRIES) {
        await delay(BEST_MATCHES_RETRY_DELAY_MS * (attempt + 1));
        continue;
      }
      if (!response.ok || !body?.ok) {
        throw bestMatchesError(response.ok ? 0 : response.status);
      }
      const data = compactBestMatches(body);
      if (!data.results.length) {
        throw bestMatchesError(0, "No best matches came back for this song.");
      }
      return data;
    }
  }

  function bestMatchesCacheId(seedUri, langMode = "same") {
    return `best:${BEST_MATCHES_POLICY}:${langMode}:${seedUri}`;
  }

  function savedBestMatches(seedUri, langMode = "same") {
    const cached = readCacheEntry("recommendations", bestMatchesCacheId(seedUri, langMode));
    return Array.isArray(cached?.results) && cached.results.length
      ? { data: cached, cached: true }
      : null;
  }

  async function requestBestMatches(title, artist, seedUri, langMode = "same") {
    const saved = savedBestMatches(seedUri, langMode);
    if (saved) return saved;
    const cacheId = bestMatchesCacheId(seedUri, langMode);
    if (pendingRecommendations.has(cacheId)) {
      return pendingRecommendations.get(cacheId);
    }
    const request = requestBestMatchesUncached(title, artist, langMode)
      .then((data) => {
        writeCacheEntry("recommendations", cacheId, data);
        return { data, cached: false };
      })
      .finally(() => pendingRecommendations.delete(cacheId));
    pendingRecommendations.set(cacheId, request);
    return request;
  }

  async function loadSeedTrack(seedUri) {
    const cached = readCacheEntry("spotifyTracks", seedUri);
    if (cached) return cached;
    const metadata = await Spicetify.GraphQL.Request(
      Spicetify.GraphQL.Definitions.getTrack,
      { uri: seedUri }
    );
    const seedTrack = metadata?.data?.trackUnion;
    if (seedTrack) writeCacheEntry("spotifyTracks", seedUri, compactSpotifyTrack(seedTrack));
    return seedTrack;
  }

  function seedArtistName(track) {
    return track?.firstArtist?.items?.[0]?.profile?.name || "";
  }

  function seedArtistNames(track) {
    const names = [
      ...(track?.firstArtist?.items || []),
      ...(track?.otherArtists?.items || []),
    ]
      .map((artist) => artist?.profile?.name)
      .filter(Boolean);
    return [...new Set(names)];
  }

  function soundsLikeError(notification, message) {
    const error = new Error(message);
    error.notification = notification;
    return error;
  }

  // The original single-list search: language check, related artists and the
  // local or hosted recommender. It now feeds the "Sounds like" view.
  async function requestSoundsLike(seedTrack, artist, seedUri) {
    let data;
    let relatedArtistsPromise = Promise.resolve(new Set());
    let languageSeed = seedTrack;
    try {
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
      languageSeed = {
        ...seedTrack,
        uri: seedTrack.uri || seedUri,
        soundalikeLanguage: seedLanguageState.language,
        soundalikeLanguageStatus: seedLanguageState.status,
      };
      writeCacheEntry("spotifyTracks", seedUri, compactSpotifyTrack(languageSeed));
      data = recommendation.data;
      if (data && typeof data === "object") {
        data.__soundalikeSource = recommendation.source;
        data.__soundalikeApiVersion = recommendation.apiVersion ||
          (recommendation.source === "local" ? "local" : "unknown");
      }
    } catch (e) {
      const message = e?.message || "recommendation service unavailable";
      throw soundsLikeError(`soundalike failed: ${message}`, message);
    }
    if (!data || !data.ok) {
      const message = (data && data.error) || "No match found.";
      throw soundsLikeError(message, message);
    }
    return {
      data,
      seedLanguage: languageSeed.soundalikeLanguage || null,
      relatedArtistsPromise,
    };
  }

  // Hosted Sounds like comes with Best matches from the GPU engine. The older
  // Sounds like service answers only when that response has no sounds list,
  // for example because the engine couldn't get the song. Null means the
  // Best matches response has the list.
  async function soundsAfterBest(best, seedTrack, artist, seedUri) {
    await best.promise;
    if (best.ok && best.value.data.sounds?.length) return null;
    return requestSoundsLike(seedTrack, artist, seedUri);
  }

  function settledRequest(promise) {
    const request = { done: false, ok: false, value: undefined, error: undefined };
    request.promise = Promise.resolve(promise).then(
      (value) => {
        Object.assign(request, { done: true, ok: true, value });
      },
      (error) => {
        Object.assign(request, { done: true, ok: false, error });
      }
    );
    return request;
  }

  function pendingResultList() {
    return { status: "pending", tracks: [] };
  }

  function applyBestOutcome(state, request) {
    state.best = request.ok
      ? {
          status: "ready",
          data: request.value.data,
          cached: request.value.cached === true,
          tracks: [],
          vibeTracks: [],
          soundsTracks: [],
        }
      : {
          status: "error",
          error: request.error?.message || bestMatchesMessage(0),
          tracks: [],
        };
    // Same vibe is hosted-only. A failed language reload keeps the tab it had.
    if (request.ok) {
      state.hasVibe = !state.companion && Boolean(request.value.data.vibe?.length);
      // Once the hosted response carries Sounds like, the tab stays on it.
      if (!state.companion && request.value.data.sounds?.length) state.soundsFromBest = true;
    }
  }

  function applySoundsOutcome(state, request) {
    // Hosted Sounds like came with Best matches, so there is no separate list.
    if (request.ok && !request.value) return;
    state.sounds = request.ok
      ? {
          status: "ready",
          data: request.value.data,
          seedLanguage: request.value.seedLanguage,
          relatedArtistsPromise: request.value.relatedArtistsPromise,
          tracks: [],
          feedbackClosed: false,
        }
      : {
          status: "error",
          error: request.error?.message || "No match found.",
          tracks: [],
        };
  }

  async function findSoundalikes(uris) {
    const startedAt = nowMs();
    const searchId = ++searchSequence;
    const id = uris[0].split(":track:")[1];
    const seedUri = `spotify:track:${id}`;
    Spicetify.showNotification("Finding soundalikes…");
    let seedTrack;
    let artist;
    try {
      seedTrack = await loadSeedTrack(seedUri);
      artist = seedArtistName(seedTrack);
      if (!seedTrack?.name || !artist) {
        throw new Error("Spotify did not return track metadata.");
      }
    } catch (e) {
      if (searchId === searchSequence) {
        Spicetify.showNotification(
          `soundalike failed: ${e?.message || "recommendation service unavailable"}`, true);
      }
      return;
    }
    const seedReadyAt = nowMs();
    // The local companion keeps searches on this computer, but Best matches
    // only run on the hosted matcher, so its users opt in for each search.
    const askFirst = await localServerReady();
    const best = askFirst
      ? null
      : settledRequest(requestBestMatches(seedTrack.name, artist, seedUri));
    const sounds = settledRequest(best
      ? soundsAfterBest(best, seedTrack, artist, seedUri)
      : requestSoundsLike(seedTrack, artist, seedUri));
    if (best) {
      // Best matches is the default view. Open as soon as it settles, or after
      // a few seconds with it still loading.
      let patienceTimer;
      const patience = new Promise((resolve) => {
        patienceTimer = setTimeout(resolve, BEST_MATCHES_PATIENCE_MS);
      });
      await Promise.race([best.promise, patience]);
      clearTimeout(patienceTimer);
      if (best.done && !best.ok) await sounds.promise;
    } else {
      await sounds.promise;
    }
    if (searchId !== searchSequence) return;
    if (best?.done && !best.ok && !sounds.ok) {
      Spicetify.showNotification(
        `${sounds.error?.notification || sounds.error?.message || "No match found."} ` +
        `${best.error?.message || bestMatchesMessage(0)}`,
        true
      );
      return;
    }
    const initialReadyMs = elapsedMs(startedAt);
    // Hosted searches fall back to Sounds like when Best matches failed; local
    // ones open on Sounds like unless it failed and the Best offer is all left.
    const showSounds = best ? best.done && !best.ok : sounds.ok;
    const state = {
      searchId,
      seedTrack,
      seedUri,
      view: showSounds ? "sounds" : "best",
      viewChosen: false,
      showAllLanguages: false,
      companion: askFirst,
      hasVibe: false,
      soundsFromBest: false,
      timing: { startedAt, initialReadyMs },
      best: best ? pendingResultList() : offeredBestList(seedUri),
      sounds: pendingResultList(),
    };
    if (best?.done) applyBestOutcome(state, best);
    if (sounds.done) applySoundsOutcome(state, sounds);
    console.log(
      `[soundalike] results page ready in ${initialReadyMs} ms ` +
      `(seed metadata ${Math.max(0, Math.round(seedReadyAt - startedAt))} ms).`
    );
    showResultsPage(state);
    if (best && !best.done) {
      best.promise.then(() => settleResultsView(searchId, "best", best));
    }
    if (!sounds.done) {
      sounds.promise.then(() => settleResultsView(searchId, "sounds", sounds));
    }
  }

  // A Best matches list saved by an earlier search needs no request, so it
  // shows straight away; otherwise the tab offers to ask the hosted matcher.
  function offeredBestList(seedUri) {
    const saved = savedBestMatches(seedUri);
    return saved
      ? { status: "ready", data: saved.data, cached: true, tracks: [] }
      : { status: "offer", tracks: [] };
  }

  function requestOfferedBestMatches() {
    const state = resultsState;
    if (!state || state.best.status !== "offer") return;
    const { searchId, seedTrack, seedUri } = state;
    state.best = pendingResultList();
    const request = settledRequest(
      requestBestMatches(seedTrack.name, seedArtistName(seedTrack), seedUri,
        state.showAllLanguages ? "all" : "same")
    );
    request.promise.then(() => settleResultsView(searchId, "best", request));
    refreshResultsPage(true);
  }

  function toggleBestLanguageMode() {
    const state = resultsState;
    if (!state || state.best.status !== "ready") return;
    state.showAllLanguages = !state.showAllLanguages;
    state.viewChosen = true;
    const { searchId, seedTrack, seedUri } = state;
    state.best = pendingResultList();
    refreshResultsPage(true);
    const request = settledRequest(requestBestMatches(
      seedTrack.name,
      seedArtistName(seedTrack),
      seedUri,
      state.showAllLanguages ? "all" : "same"
    ));
    request.promise.then(() => settleResultsView(searchId, "best", request));
  }

  // Applies a list that finished after the page opened.
  function settleResultsView(searchId, kind, request) {
    const state = resultsState;
    if (!state || state.searchId !== searchId) return;
    if (kind === "best") applyBestOutcome(state, request);
    else applySoundsOutcome(state, request);
    // Failed Best matches give way to a Sounds like list, whichever came first.
    if (
      state.best.status === "error" &&
      state.view === "best" &&
      !state.viewChosen &&
      state.sounds.status === "ready"
    ) {
      state.view = "sounds";
      refreshResultsPage();
      return;
    }
    if (state.view === "vibe" && !state.hasVibe) state.view = "best";
    const fromBest = state.view === "vibe" || (state.view === "sounds" && state.soundsFromBest);
    if (state.view === kind || (kind === "best" && fromBest)) {
      refreshResultsPage();
    }
  }

  // Rebuilding the page would drop focus inside it to the body, so it moves to
  // the selected tab. Focus elsewhere in Spotify stays where it is.
  function refreshResultsPage(focusTab = false) {
    if (!activePage?.container) return;
    if (Spicetify.Platform.History.location?.pathname !== RESULTS_PATH) return;
    const focused = document.activeElement;
    const keepFocus = focusTab || Boolean(focused && activePage.host?.contains?.(focused));
    renderResultsPage(activePage.container);
    if (keepFocus) focusSelectedTab();
  }

  function focusSelectedTab() {
    const view = resultsState?.view;
    activePage?.view?.querySelector?.(`.sa-tab[data-view="${view}"]`)?.focus?.();
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

  function playingTrackUri() {
    const data = Spicetify.Player?.data;
    return data?.item?.uri || data?.track?.uri || null;
  }

  // Like Spotify's own lists, the page marks the song that is playing.
  function markPlayingRows(page = activePage?.view) {
    const uri = playingTrackUri();
    page?.querySelectorAll?.(".sa-row").forEach((row) => {
      row.classList?.toggle?.("sa-row-playing", Boolean(uri) && row.dataset?.uri === uri);
    });
  }

  // The Spotify songs the page shows, in its order, each once.
  function shownResultUris(page = activePage?.view) {
    const rows = Array.from(page?.querySelectorAll?.(".sa-row") || [])
      .map((row, position) => ({
        uri: row.dataset?.uri,
        hidden: row.hidden,
        order: Number(row.style?.order) || Number(row.dataset?.index) + 1 || 0,
        position,
      }))
      .filter((row) => row.uri && !row.hidden);
    rows.sort((left, right) => left.order - right.order || left.position - right.position);
    return [...new Set(rows.map((row) => row.uri))];
  }

  // Plays the shown songs from this one as an unsaved list, so the next row
  // plays next, the way Spotify's song radio does. False when it can't.
  async function playShownResults(uri) {
    const player = Spicetify.Platform?.PlayerAPI;
    const uris = shownResultUris();
    const index = uris.indexOf(uri);
    if (typeof player?.play !== "function" || index < 0) return false;
    lateMatches = null;
    await player.play(
      {
        uri: RESULTS_CONTEXT_URI,
        pages: [{ items: uris.map((item) => ({ uri: item })) }],
      },
      { featureIdentifier: "soundalike" },
      { skipTo: { index, uri } }
    );
    followLateMatches(uris);
    return true;
  }

  // A started list can't grow, so rows matched to a Spotify song after it
  // started are queued, in page order, once the list's last song plays.
  function followLateMatches(played) {
    const page = activePage;
    if (!page?.settled) return;
    const late = { last: played.at(-1), uris: [] };
    lateMatches = late;
    page.settled.then(() => {
      if (lateMatches !== late) return;
      late.uris = shownResultUris(page.view).filter((uri) => !played.includes(uri));
      if (late.uris.length) void queueLateMatches();
      else lateMatches = null;
    });
  }

  function playingLastLateSong(late, player) {
    const state = player.getState?.();
    return state?.context?.uri === RESULTS_CONTEXT_URI && state.item?.uri === late.last;
  }

  // Waits for the list's last song to settle in, queues the late songs, and
  // queues them again once if Spotify dropped them.
  async function queueLateMatches() {
    const late = lateMatches;
    const player = Spicetify.Platform?.PlayerAPI;
    if (!late?.uris.length || late.queueing || typeof player?.addToQueue !== "function") return;
    if (!playingLastLateSong(late, player)) return;
    late.queueing = true;
    try {
      for (let attempt = 0; attempt < 2; attempt += 1) {
        await delay(LATE_QUEUE_DELAY_MS);
        if (lateMatches !== late || !playingLastLateSong(late, player)) return;
        const queued = new Set((player.getQueue?.()?.queued || []).map((item) => item.uri));
        if (late.uris.every((uri) => queued.has(uri))) break;
        await player.addToQueue(late.uris.map((uri) => ({ uri })));
      }
      lateMatches = null;
    } catch (error) {
      console.warn("[soundalike] Songs matched after the list started could not be queued.", error);
    } finally {
      late.queueing = false;
    }
  }

  async function activateResult(result, track) {
    if (track?.uri && typeof Spicetify.Player?.playUri === "function") {
      try {
        if (await playShownResults(track.uri)) return;
      } catch (error) {
        console.warn(
          "[soundalike] Spotify would not play the results as a list; playing the song alone.",
          error
        );
      }
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

  function renderResultRow(
    row,
    result,
    index,
    track = null,
    rank = index + 1,
    { bpm = true } = {}
  ) {
    const artist = spotifyArtists(track, result.artist);
    const image = track ? artworkUrl(track) : "";
    const matchNote = result.matchNote ? ` · ${result.matchNote}` : "";
    row.dataset.q = `${result.title} ${result.artist}`;
    if (track?.uri) row.dataset.uri = track.uri;
    row.classList?.toggle?.(
      "sa-row-playing",
      Boolean(track?.uri) && track.uri === playingTrackUri()
    );

    const React = Spicetify.React;
    if (!(React?.createElement && Spicetify.ReactDOM?.createRoot)) {
      const artistNode = row.querySelector(".sa-artist");
      if (artistNode) artistNode.textContent = `${artist}${matchNote}`;
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
        matchNote
          ? React.createElement(
            "div",
            { className: "sa-artist" },
            artist,
            React.createElement("span", { className: "sa-why" }, matchNote)
          )
          : React.createElement("div", { className: "sa-artist" }, artist)
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
      bpm
        ? React.createElement("div", { className: "sa-bpm" }, formatBpm(result.bpm))
        : null
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
    if (names[code]) return names[code];
    try {
      return new Intl.DisplayNames(["en"], { type: "language" }).of(code);
    } catch {
      return code?.toUpperCase() || "Unknown";
    }
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
    if (page === activePage?.view && resultsState?.sounds?.data) {
      setupFeedbackPrompt(
        page,
        resultsState.sounds.data,
        displayed.map(({ index }) => results[index]),
        resultsState.sounds
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

  function revealBestRow(page, results, tracks, index, bpm = false) {
    const row = page.querySelector(`.sa-row[data-index="${index}"]`);
    if (!row || page !== activePage?.view) return;
    renderResultRow(row, results[index], index, tracks[index] || null, index + 1, { bpm });
    row.hidden = false;
  }

  // Best matches, Same vibe and hosted Sounds like arrive already ranked, so
  // Spotify only adds artwork, album and playback. The lyrics-language gate
  // and feedback belong to the companion's Sounds like list.
  async function hydrateBestRows(
    results,
    page,
    tracks,
    timing,
    label = "Best matches",
    { bpm = false } = {}
  ) {
    const startedAt = nowMs();
    if (!Spicetify.GraphQL.Definitions.searchModalResults) {
      console.warn("[soundalike] Spotify artwork lookup is unavailable in this client.");
      return;
    }
    let cursor = 0;
    async function worker() {
      while (cursor < results.length) {
        const index = cursor++;
        if (!(index in tracks)) {
          try {
            tracks[index] = await findSpotifyTrack(results[index]);
          } catch (error) {
            tracks[index] = null;
            console.warn(
              `[soundalike] Spotify metadata lookup failed for ${results[index].title}`,
              error
            );
          }
        }
        revealBestRow(page, results, tracks, index, bpm);
      }
    }
    const workerCount = Math.min(SPOTIFY_ENRICH_WORKERS, results.length);
    await Promise.all(Array.from({ length: workerCount }, worker));
    const enriched = results.filter((_, index) => tracks[index]).length;
    console.log(
      `[soundalike] ${label} metadata settled in ${elapsedMs(startedAt)} ms ` +
      `(${enriched}/${results.length} rows, ` +
      `${timing ? `${elapsedMs(timing.startedAt)} ms total` : "total unavailable"}).`
    );
  }

  function feedbackMarkup() {
    return `
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
  }

  // Same vibe is only offered once a hosted Best matches response carried it.
  function resultViews(state) {
    return RESULT_VIEWS.filter(([view]) => view !== "vibe" || state.hasVibe);
  }

  function resultTabsMarkup(state) {
    const tabs = resultViews(state).map(([view, label]) => {
      const selected = state.view === view;
      return `<button class="sa-tab" type="button" role="tab" id="sa-tab-${view}" ` +
        `data-view="${view}" aria-controls="sa-panel" aria-selected="${selected}" ` +
        `tabindex="${selected ? 0 : -1}">${esc(label)}</button>`;
    });
    return `<div class="sa-tabs" role="tablist" aria-label="Result lists">${tabs.join("")}</div>`;
  }

  function tagsMarkup(tags) {
    return tags
      .filter(Boolean)
      .map((tag) => `<span class="sa-tag">${esc(tag)}</span>`)
      .join("");
  }

  function listStateMarkup(list, pendingText) {
    if (list.status === "error") {
      return `<div class="sa-state sa-state-error" role="alert">${esc(list.error)}</div>`;
    }
    return `<div class="sa-state" role="status">${esc(pendingText)}</div>`;
  }

  // True when a service's song is the same one the user picked in Spotify.
  function sameSeedSong(title, artist, seedTitle, seedArtists) {
    if (normalizeLabel(title) !== normalizeLabel(seedTitle)) return false;
    const label = ` ${normalizeLabel(artist)} `;
    return seedArtists.some((name) => {
      const seed = normalizeLabel(name);
      return Boolean(seed) && label.includes(` ${seed} `);
    });
  }

  function resultRowsMarkup(results, { hidden, notes }) {
    return results
      .map(
        (x, i) => `
      <div class="sa-row" data-index="${i}" data-q="${esc(x.title + " " + x.artist)}"${hidden ? " hidden" : ""}>
        <div class="sa-rank">${i + 1}</div>
        <div class="sa-cover" aria-hidden="true"><span>♫</span></div>
        <div class="sa-meta">
          <div class="sa-title">${esc(x.title)}</div>
          <div class="sa-artist">${esc(x.artist)}${notes && x.matchNote
            ? `<span class="sa-why"> · ${esc(x.matchNote)}</span>`
            : ""}</div>
        </div>
        <div class="sa-open">›</div>
      </div>`
      )
      .join("");
  }

  // Best matches and Same vibe come from one response, so they share its notes,
  // tags and language switch.
  function matchedToNote(state, query, seedArtists) {
    return query.title &&
      query.artist &&
      !sameSeedSong(query.title, query.artist, state.seedTrack?.name, seedArtists)
      ? ` Matched to “${query.title}” by ${query.artist}.`
      : "";
  }

  function sourceTags(best) {
    const { query, totalMs } = best.data;
    const cached = best.cached || best.data.serverCached;
    return [
      query.source !== "live"
        ? "In our library"
        : cached ? "Heard live earlier" : "Heard live just now",
      !cached && totalMs > 0 ? `Matched in ${(totalMs / 1000).toFixed(1)} s` : "",
    ];
  }

  function languageFilterMarkup(query, status) {
    const allLanguages = query.languageMode === "all";
    const unknown = query.language === "unknown";
    const control = unknown ? "" :
      `<button class="sa-language-toggle" type="button" aria-pressed="${allLanguages}">${allLanguages ? "Use same-language matches" : "Show all languages"}</button>`;
    const text = unknown
      ? "Language uncertain; showing all languages."
      : `${languageName(query.language)} · ${status}`;
    return `<div class="sa-language-filter" role="status"><span>${esc(text)}</span>${control}</div>`;
  }

  function bestPanelMarkup(state, seedArtists) {
    const best = state.best;
    let blurb = BEST_MATCHES_BLURB;
    if (best.status === "offer") {
      const note = !state.viewChosen && state.sounds.status === "error"
        ? `<p class="sa-note">${esc(state.sounds.error)}</p>`
        : "";
      return `<p class="sa-blurb">${esc(blurb)}</p>${note}
      <div class="sa-offer">
        <p>${esc(BEST_MATCHES_OFFER)}</p>
        <button class="sa-best-offer" type="button">Get best matches</button>
      </div>`;
    }
    if (best.status !== "ready") {
      return `<p class="sa-blurb">${esc(blurb)}</p>` + listStateMarkup(
        best,
        "Listening for best matches… songs outside the library take a few seconds."
      );
    }
    const { query, results } = best.data;
    blurb += matchedToNote(state, query, seedArtists);
    const hidden = Math.max(0, Number(query.hidden) || 0);
    const languageStatus = hidden
      ? `${hidden} other-language ${hidden === 1 ? "match" : "matches"} ` +
        `${query.languageMode === "all" ? "in" : "hidden from"} the top ${DISPLAY_RESULT_COUNT}`
      : `no other-language matches in the top ${DISPLAY_RESULT_COUNT}`;
    const tags = tagsMarkup(["Melody + samples", ...sourceTags(best)]);
    return `<p class="sa-blurb">${esc(blurb)}</p>
      <div class="sa-tags">${tags}</div>
      ${languageFilterMarkup(query, languageStatus)}
      <div class="sa-list-head"><div>#</div><div></div><div>Title</div><div class="sa-album-head">Album</div></div>
      <div class="sa-list">${resultRowsMarkup(results, { hidden: false, notes: true })}
      </div>`;
  }

  // Best's hidden count doesn't describe this list, so its status only says
  // whether other languages are left out.
  function vibePanelMarkup(state, seedArtists) {
    const best = state.best;
    if (best.status !== "ready") {
      return `<p class="sa-blurb">${esc(SAME_VIBE_BLURB)}</p>` +
        listStateMarkup(best, "Finding songs with the same vibe…");
    }
    const { query, vibe } = best.data;
    const blurb = SAME_VIBE_BLURB + matchedToNote(state, query, seedArtists);
    const tags = tagsMarkup([
      "Tempo + sound",
      query.bpm ? `${Math.round(query.bpm)} BPM` : "",
      ...sourceTags(best),
    ]);
    const languageStatus = query.languageMode === "all"
      ? "all languages"
      : "other languages left out";
    return `<p class="sa-blurb">${esc(blurb)}</p>
      <div class="sa-tags">${tags}</div>
      ${languageFilterMarkup(query, languageStatus)}
      <div class="sa-list-head"><div>#</div><div></div><div>Title</div><div class="sa-album-head">Album</div></div>
      <div class="sa-list">${resultRowsMarkup(vibe, { hidden: false, notes: false })}
      </div>`;
  }

  // Hosted Sounds like comes with Best matches, so it shares their language
  // switch; its rows keep the BPM column.
  function hostedSoundsPanelMarkup(state, seedArtists) {
    const best = state.best;
    if (best.status !== "ready") {
      return `<p class="sa-blurb">${esc(HOSTED_SOUNDS_LIKE_BLURB)}</p>` +
        listStateMarkup(best, "Finding songs that sound alike…");
    }
    const { query, sounds } = best.data;
    if (!sounds?.length) {
      return `<p class="sa-blurb">${esc(HOSTED_SOUNDS_LIKE_BLURB)}</p>` + listStateMarkup(
        { status: "error", error: "No sound-alike songs came back for this song." },
        ""
      );
    }
    const blurb = HOSTED_SOUNDS_LIKE_BLURB + matchedToNote(state, query, seedArtists);
    const tags = tagsMarkup([
      "Overall sound",
      query.bpm ? `${Math.round(query.bpm)} BPM` : "",
      ...sourceTags(best),
    ]);
    const languageStatus = query.languageMode === "all"
      ? "all languages"
      : "other languages left out";
    return `<p class="sa-blurb">${esc(blurb)}</p>
      <div class="sa-tags">${tags}</div>
      ${languageFilterMarkup(query, languageStatus)}
      <div class="sa-list-head"><div>#</div><div></div><div>Title</div><div class="sa-album-head">Album</div><div class="sa-bpm">BPM</div></div>
      <div class="sa-list">${resultRowsMarkup(sounds, { hidden: false, notes: false })}
      </div>`;
  }

  function soundsPanelMarkup(state, seedArtists) {
    if (state.soundsFromBest) return hostedSoundsPanelMarkup(state, seedArtists);
    const sounds = state.sounds;
    let blurb = SOUNDS_LIKE_BLURB;
    const note = state.view === "sounds" &&
      !state.viewChosen &&
      state.best.status === "error"
      ? `<p class="sa-note">${esc(`${state.best.error} Showing Sounds like instead.`)}</p>`
      : "";
    if (sounds.status !== "ready") {
      return `<p class="sa-blurb">${esc(blurb)}</p>${note}` +
        listStateMarkup(sounds, "Finding songs that sound alike…");
    }
    const data = sounds.data;
    const s = data.seed || {}, v = data.vibe || {};
    if (
      s.title &&
      !sameSeedSong(s.title, s.artist, state.seedTrack?.name, seedArtists)
    ) {
      blurb += ` Based on “${s.title}” by ${s.artist}, the closest song in that library.`;
    }
    const languageStatus = sounds.seedLanguage
      ? `${languageName(sounds.seedLanguage)} lyrics · checking quality head`
      : "Checking Spotify metadata and artist context";
    const modelLabel = data.method === "dual_sonic64_guardrail"
      ? "Dual-Sonic64"
      : "Production model";
    const tags = tagsMarkup([
      modelLabel,
      "V5 strict + artist context",
      v.tempo,
      v.dynamics,
      v.low_end,
      v.tone,
    ]) + `<span class="sa-tag sa-language-status">${esc(languageStatus)}</span>`;
    return `<p class="sa-blurb">${esc(blurb)}</p>${note}
      <div class="sa-tags">${tags}</div>
      <div class="sa-list-head"><div>#</div><div></div><div>Title</div><div class="sa-album-head">Album</div><div class="sa-bpm">BPM</div></div>
      <div class="sa-list">
        <div class="sa-language-loading">Checking Spotify language and artist context before showing results…</div>
        ${resultRowsMarkup(data.results, { hidden: true, notes: false })}
      </div>${feedbackMarkup()}`;
  }

  function shownView(state) {
    return resultViews(state).some(([key]) => key === state.view) ? state.view : "best";
  }

  // The list a view shows. Best matches, Same vibe and hosted Sounds like all
  // come from the Best matches response; only the companion's Sounds like (or
  // the older hosted service, as a fallback) has its own.
  function viewList(state, view) {
    const best = state.best;
    if (view === "sounds" && !state.soundsFromBest) {
      const sounds = state.sounds;
      return { list: sounds, results: sounds.data?.results, tracks: sounds.tracks, separate: true };
    }
    if (view === "vibe") {
      return { list: best, results: best.data?.vibe, tracks: (best.vibeTracks ||= []), label: "Same vibe" };
    }
    if (view === "sounds") {
      return {
        list: best,
        results: best.data?.sounds,
        tracks: (best.soundsTracks ||= []),
        label: "Sounds like",
        bpm: true,
      };
    }
    return { list: best, results: best.data?.results, tracks: best.tracks, label: "Best matches" };
  }

  function buildResultsPage(state) {
    const wrap = document.createElement("div");
    const view = shownView(state);
    const seedTrack = state.seedTrack;
    const seedImage = artworkUrl(seedTrack);
    const seedArtists = seedArtistNames(seedTrack);
    const sounds = state.sounds;
    let kicker = view === "vibe"
      ? "SAME VIBE · HOSTED LIBRARY"
      : "BEST MATCHES · HOSTED LIBRARY";
    if (view === "sounds") {
      kicker = state.soundsFromBest
        ? "SOUNDS LIKE · HOSTED LIBRARY"
        : sounds.status === "ready"
          ? `SOUNDS LIKE · ${sounds.data.__soundalikeSource === "local"
            ? "LOCAL ENGINE"
            : "HOSTED LIBRARY"}`
          : "SOUNDS LIKE";
    }
    wrap.className = "sa-results";

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
        .sa-album .sa-album-link{color:inherit;text-decoration:none}
        .sa-album .sa-album-link:hover,.sa-album .sa-album-link:focus-visible{color:var(--spice-text,#fff);text-decoration:underline;outline:none}
        .sa-bpm{color:var(--spice-subtext,#b3b3b3);font-variant-numeric:tabular-nums;text-align:right}
        .sa-play{grid-area:1/1;display:none;width:30px;height:30px;border:0;border-radius:50%;background:transparent;color:var(--spice-text,#fff);font-size:15px;cursor:pointer}
        .sa-row-content:hover .sa-rank,.sa-row-content:focus-within .sa-rank{display:none}
        .sa-row-content:hover .sa-play,.sa-play:focus-visible{display:grid;place-items:center}
        .sa-play:hover,.sa-play:focus-visible{background:var(--spice-button,#1ed760);color:#000;outline:none}
        .sa-row-content:hover .sa-title,.sa-row-content:focus-visible .sa-title{color:var(--spice-text,#fff)}
        .sa-row.sa-row-playing .sa-title,.sa-row.sa-row-playing .sa-rank{color:var(--spice-button,#1ed760)}
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
        .sa-tabs{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 14px}
        .sa-tab{min-height:32px;padding:6px 14px;border:0;border-radius:999px;background:rgba(255,255,255,.08);color:var(--spice-text,#fff);font:inherit;font-weight:600;cursor:pointer}
        .sa-tab:hover{background:rgba(255,255,255,.14)}
        .sa-tab[aria-selected="true"]{background:var(--spice-text,#fff);color:var(--spice-main,#121212)}
        .sa-tab:focus-visible{outline:2px solid var(--spice-button,#1ed760);outline-offset:2px}
        .sa-language-filter{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:0 0 14px;color:var(--spice-subtext,#b3b3b3);font-size:13px}
        .sa-language-toggle{min-height:32px;border:1px solid rgba(255,255,255,.2);border-radius:999px;padding:5px 12px;background:transparent;color:var(--spice-text,#fff);font:inherit;font-weight:600;cursor:pointer}
        .sa-language-toggle:hover{border-color:var(--spice-button,#1ed760)}
        .sa-language-toggle:focus-visible{outline:2px solid var(--spice-button,#1ed760);outline-offset:2px}
        .sa-blurb{max-width:680px;margin:0 0 10px;color:var(--spice-subtext,#b3b3b3)}
        .sa-note{max-width:680px;margin:0 0 12px;color:var(--spice-text,#fff)}
        .sa-why{color:var(--spice-subtext,#b3b3b3)}
        .sa-state{padding:18px 6px;color:var(--spice-subtext,#b3b3b3)}
        .sa-state-error{color:#f29d9d}
        .sa-offer{max-width:680px;padding:16px;border:1px solid rgba(255,255,255,.12);border-radius:8px;background:rgba(255,255,255,.04)}
        .sa-offer p{margin:0 0 12px;color:var(--spice-subtext,#b3b3b3)}
        .sa-best-offer{min-height:36px;border:0;border-radius:999px;padding:7px 14px;background:var(--spice-button,#1ed760);color:#000;font:inherit;font-weight:700;cursor:pointer}
        .sa-best-offer:focus-visible{outline:2px solid var(--spice-text,#fff);outline-offset:2px}
        .sa-view-best .sa-list-head,.sa-view-best .sa-row-content,.sa-view-vibe .sa-list-head,.sa-view-vibe .sa-row-content{grid-template-columns:32px 44px minmax(180px,1.6fr) minmax(120px,1fr)}
        @media(max-width:760px){
          .sa-list-head,.sa-row-content{grid-template-columns:32px 44px minmax(0,1fr) 56px}
          .sa-view-best .sa-list-head,.sa-view-best .sa-row-content,.sa-view-vibe .sa-list-head,.sa-view-vibe .sa-row-content{grid-template-columns:32px 44px minmax(0,1fr)}
          .sa-album,.sa-list-head .sa-album-head{display:none}
          .sa-feedback-help{display:block}
        }
      </style>
      <div class="sa-seed">
        ${seedImage
          ? `<img src="${esc(seedImage)}" alt="">`
          : `<div class="sa-seed-fallback" aria-hidden="true">♫</div>`}
        <div>
          <div class="sa-kicker">${esc(kicker)}</div>
          <div class="sa-seed-title">${esc(seedTrack?.name)}</div>
          <div class="sa-seed-artist">${esc(seedArtists.join(", "))}</div>
        </div>
      </div>
      ${resultTabsMarkup(state)}
      <div class="sa-panel sa-view-${view}" id="sa-panel" role="tabpanel" aria-labelledby="sa-tab-${view}">
        ${view === "best"
          ? bestPanelMarkup(state, seedArtists)
          : view === "vibe"
            ? vibePanelMarkup(state, seedArtists)
            : soundsPanelMarkup(state, seedArtists)}
      </div>`;

    const shown = viewList(state, view);
    const results = (shown.list.status === "ready" && shown.results) || [];
    const tracks = shown.tracks || [];
    wrap.querySelectorAll(".sa-row").forEach((row) => {
      const index = Number(row.dataset?.index);
      const result = results[index];
      if (!result) return;
      if (!shown.separate) {
        row.hidden = false;
        renderResultRow(row, result, index, tracks[index] || null, index + 1, {
          bpm: Boolean(shown.bpm),
        });
        return;
      }
      row.hidden = Boolean(shown.list.seedLanguage);
      renderResultRow(row, result, index, tracks[index] || null);
    });
    return wrap;
  }

  function selectResultsView(view, focus = false) {
    const state = resultsState;
    if (!state || !resultViews(state).some(([key]) => key === view)) return;
    state.viewChosen = true;
    if (state.view !== view) {
      state.view = view;
      refreshResultsPage(focus);
    } else if (focus) {
      focusSelectedTab();
    }
  }

  function bindResultTabs(page) {
    const tabs = Array.from(page.querySelectorAll(".sa-tab") || []);
    tabs.forEach((tab, index) => {
      tab.onclick = () => selectResultsView(tab.dataset.view, true);
      tab.onkeydown = (event) => {
        let next = null;
        if (event.key === "ArrowRight") next = tabs[(index + 1) % tabs.length];
        else if (event.key === "ArrowLeft") {
          next = tabs[(index - 1 + tabs.length) % tabs.length];
        } else if (event.key === "Home") next = tabs[0];
        else if (event.key === "End") next = tabs[tabs.length - 1];
        if (!next) return;
        event.preventDefault?.();
        selectResultsView(next.dataset.view, true);
      };
    });
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
    const state = resultsState;
    const page = buildResultsPage(state);
    const mounted = createResultsHost(container, page);
    activePage = { container, view: page, ...mounted };
    nativeContextChain = undefined;
    bindResultTabs(page);
    const offer = page.querySelector(".sa-best-offer");
    if (offer) offer.onclick = requestOfferedBestMatches;
    const languageToggle = page.querySelector(".sa-language-toggle");
    if (languageToggle) languageToggle.onclick = toggleBestLanguageMode;
    const view = shownView(state);
    const shown = viewList(state, view);
    if (!shown.separate) {
      if (state.best.status !== "ready") return;
      const { label } = shown;
      activePage.settled = hydrateBestRows(
        shown.results || [],
        page,
        shown.tracks,
        state.timing,
        label,
        { bpm: Boolean(shown.bpm) }
      ).catch((error) => {
        console.error(`[soundalike] ${label} enrichment failed`, error);
      });
      return;
    }
    if (state.sounds.status !== "ready") return;
    activePage.settled = hydrateSpotifyRows(
      state.sounds.data.results,
      page,
      state.sounds.tracks,
      state.sounds.seedLanguage,
      state.sounds.relatedArtistsPromise,
      state.timing
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

  function showResultsPage(state) {
    resultsState = state;
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

  function versionParts(version) {
    const match = /^v?(\d+)\.(\d+)\.(\d+)/.exec(String(version ?? "").trim());
    return match ? match.slice(1).map(Number) : null;
  }

  function versionAtLeast(parts, minimum) {
    for (let index = 0; index < minimum.length; index += 1) {
      if (parts[index] !== minimum[index]) return parts[index] > minimum[index];
    }
    return true;
  }

  // Only the known-broken pairing is skipped. Unknown or unreadable versions
  // keep the menu item.
  function menuBlockedBySpicetify() {
    const spicetify = versionParts(Spicetify.Config?.version);
    const spotify = versionParts(Spicetify.Platform?.version);
    return Boolean(
      spicetify &&
      spotify &&
      !versionAtLeast(spicetify, MENU_FIXED_SPICETIFY_VERSION) &&
      versionAtLeast(spotify, MENU_BREAKING_SPOTIFY_VERSION)
    );
  }

  // Spotify's now-playing song menus (the player bar title) carry only
  // canToggleCanvas, and Spicetify up to 2.46.4 passes them no song, so the
  // main item can't appear there (spicetify/cli#4039). These menus are always
  // about the song that is playing. Once Spicetify passes the song, the main
  // item covers them and this one stays out.
  function nowPlayingMenuTrack(props) {
    if (!props || typeof props !== "object" || !Object.hasOwn(props, "canToggleCanvas")) {
      return null;
    }
    if (Spicetify.ContextMenuV2.parseProps(props)) return null;
    const uri = playingTrackUri();
    return typeof uri === "string" && uri.startsWith("spotify:track:") ? uri : null;
  }

  function registerNowPlayingMenuItem() {
    const menus = Spicetify.ContextMenuV2;
    if (typeof menus?.Item !== "function" || typeof menus.parseProps !== "function") return;
    try {
      window.__soundalikeNowPlayingMenuItem = new menus.Item({
        children: "Find soundalikes",
        leadingIcon: "enhance",
        shouldAdd: (props) => Boolean(nowPlayingMenuTrack(props)),
        // Spicetify hands the clicked menu's context ({ props, ... }).
        onClick: (context) => {
          const uri = nowPlayingMenuTrack(context?.props);
          if (uri) findSoundalikes([uri]);
        },
      });
      window.__soundalikeNowPlayingMenuItem.register();
    } catch (error) {
      console.warn("[soundalike] Find soundalikes could not join the player bar menu.", error);
    }
  }

  if (menuBlockedBySpicetify()) {
    // Leave Spotify's own menu working and explain the fix instead.
    Spicetify.showNotification(
      `Find soundalikes is off: Spicetify ${Spicetify.Config.version} can't ` +
        `add right-click items on this Spotify version. Run "spicetify update" ` +
        "to turn it on.",
      false,
      10000
    );
    console.warn(
      `[soundalike] Spicetify ${Spicetify.Config.version} empties every ` +
        `right-click menu on Spotify ${Spicetify.Platform.version} once an ` +
        "extension item is registered, so Find soundalikes was not added. " +
        'Run "spicetify update".'
    );
    return;
  }

  Spicetify.Platform.History.listen(onHistoryChange);
  Spicetify.Player?.addEventListener?.("songchange", () => {
    markPlayingRows();
    void queueLateMatches();
  });

  window.__soundalikeContextMenuItem = new Spicetify.ContextMenu.Item(
    "Find soundalikes",
    findSoundalikes,
    onlyTracks,
    "enhance" // Spicetify built-in icon
  );
  window.__soundalikeContextMenuItem.register();
  registerNowPlayingMenuItem();
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
