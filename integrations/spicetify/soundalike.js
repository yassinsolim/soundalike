// soundalike — Spicetify extension
// Adds a right-click "Find soundalikes" item to any track in the Spotify
// desktop client. Results open on a navigable Spotify page with three lists
// from one hosted response: "Best matches" (likely covers and samples, then
// other artists' songs a model ranks closest from five signals, shown by
// default), "Listeners also play" (other artists' songs this song's listeners
// play) and "Melody match" (the melody and sample matches fused into one
// ranking). Each list shows 20 songs and Show more adds the next ones. With
// the local companion, or when the hosted matcher can't take the song, a
// "Sounds like" list from the
// soundalike recommender joins them. Playing a result plays the shown list on
// in order, like a playlist that is never saved. Each hosted row's "Stems"
// button opens the seed and that match split into instruments and compared
// part by part (the hosted /api/anysong/dissect endpoint). The page stays a
// place to return to: a top bar button and Spotify's Back and Forward reopen
// it as it was left, and Recent lists the last searches.
//
// Install (requires a patchable desktop Spotify app): follow this directory's
// README for Marketplace or manual setup.

(function soundalike() {
  const RUNTIME_SEMANTIC_VERSION = "2.16.0";
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
  // Best matches and Sounds like entries share this bucket. A hosted entry
  // holds every list at full length, so the bucket stays small.
  const MAX_RECOMMENDATION_CACHE_SIZE = 60;
  const MAX_SPOTIFY_CACHE_SIZE = 500;
  const HOSTED_API_VERSION = "4";
  const LANGUAGE_POLICY = "spotify-lyrics-strict-v2";
  const RANKING_POLICY = "model-quality-v1";
  const FEEDBACK_SELECTION_POLICY =
    "top-20-strict-language-related-artist-model-quality-v1";
  // Lists shown with Audio only on skip the related-artist ordering, so their
  // feedback says so and isn't mixed with the listener-aware lists'.
  const FEEDBACK_AUDIO_ONLY_SELECTION_POLICY =
    "top-20-strict-language-audio-only-model-quality-v1";
  // Hosted lists show this many songs, and as many again with each Show more.
  const DISPLAY_RESULT_COUNT = 20;
  const RECOMMENDATION_POOL_SIZE = DISPLAY_RESULT_COUNT;
  // Rows asked of the hosted matcher: the most it gives. A longer list of a
  // song is the same ranking cut later, so one answer covers every Show more,
  // and the songs Spotify doesn't have, whose places the next ones take.
  const HOSTED_RESULT_COUNT = 50;
  // The hosted lists start hidden and fill in as their songs are found on
  // Spotify (hydrateBestRows); this line says so, then names the songs that
  // weren't found.
  const LIST_NOTE_MARKUP =
    '<div class="sa-more-row"><button class="sa-more" type="button" hidden>Show more</button></div>' +
    '<p class="sa-list-note" role="status">Finding these songs on Spotify…</p>';
  const LANGUAGE_LOOKUP_ATTEMPTS = 2;
  const LANGUAGE_RETRY_DELAY_MS = 350;
  const RELATED_ARTIST_TIMEOUT_MS = 1500;
  const SPOTIFY_ENRICH_WORKERS = 4;
  // Hosted rows appear in rank order, so a slow Spotify lookup would hold back
  // every row below it; past this it shows unmatched until its song arrives.
  const SPOTIFY_LOOKUP_PATIENCE_MS = 5000;
  const SPOTIFY_TRACK_SEARCH_LIMIT = 20;
  const SPOTIFY_TRACK_SEARCH_HASH =
    "59ee4a659c32e9ad894a71308207594a65ba67bb6b632b183abe97303a51fa55";
  const RESULTS_PATH = "/soundalike";
  // Spotify links this app URI to RESULTS_PATH. Results play as one context
  // under it, so the next song is the next row.
  const RESULTS_CONTEXT_URI = "spotify:app:soundalike";
  // Spotify drops songs queued just as a song starts playing.
  const LATE_QUEUE_DELAY_MS = 1500;
  // Songs this extension queued; Spotify keeps its queue across restarts.
  const LATE_QUEUE_KEY = "soundalike:late-queue:v1";
  // The hosted tab last chosen, kept for later searches.
  const RESULT_VIEW_KEY = "soundalike:view:v1";
  const REMEMBERED_VIEWS = ["best", "listeners", "melody"];
  // The last searches, newest first: { uri, title, artist, image }.
  const HISTORY_KEY = "soundalike:history:v1";
  const HISTORY_SIZE = 20;
  // Where the page goes: Spotify's main view panel, else (older clients) the
  // main view's container. The class names come from Spicetify's class map,
  // which themes are written against too.
  const MAIN_VIEW_PANEL = ".Root__main-view, #main-view";
  const MAIN_VIEW_SELECTORS = [
    ".Root__main-view",
    "#main-view",
    '[data-testid="main-view-container"]',
    ".main-view-container",
    "main",
  ];
  // The player bar and, where a theme floats it outside that box, its content.
  const PLAYER_BAR_SELECTOR = ".Root__now-playing-bar, .main-nowPlayingBar-container";
  const PAGE_BOTTOM_PADDING = 56;
  const BEST_MATCHES_ENDPOINT = `${FALLBACK_HOSTED_SERVER}/api/anysong/similar`;
  const BEST_MATCHES_POLICY = "anysong-main-consensus-v17";
  // Match a part: a server that can rank a part of the 30-second preview says
  // so in every answer (query.parts), so the control needs no request of its
  // own and stays hidden with servers that can't. Spans sit on a 5-second
  // grid and are 10 to 20 seconds long.
  const PART_STEP_SECONDS = 5;
  const PART_LONGEST_SECONDS = 20;
  const PART_DEFAULT = [10, 20];
  // Where previews usually start in the song, for songs the server gives no place for.
  const USUAL_PREVIEW_OFFSET_SECONDS = 48;
  const PART_PLAY_WAIT_MS = 3000;
  const PART_PLAY_POLL_MS = 250;
  // Further than this from the part, the listener moved in the song themselves.
  const PART_PLAY_SLACK_MS = 1500;
  const PART_INTRO =
    "Soundalike hears a 30-second preview. Pick the part of it you care about.";
  const PART_LISTS_NOTE =
    "Best matches, Listeners also play and Melody match follow the part; Sounds like always uses the whole song. " +
    "Listeners also play still starts from the whole song's listeners; only how close each song sounds follows the part.";
  const PART_PLACE_NOTE =
    "Song times assume the preview sits where previews usually do; in a few songs it is elsewhere.";
  const PART_KEYS_NOTE =
    "Drag the part or its edges. Keyboard: arrow keys move it, Shift with an arrow key changes its length.";
  const PART_PLAY_TITLE =
    "Plays the song in Spotify from about where this part starts, then pauses or returns to where you were in it";
  // Audio only: the hosted lists come without the scene and listener layer
  // (scene=0), and the companion's Sounds like without related-artist ordering.
  const AUDIO_ONLY_KEY = "soundalike:audio-only:v1";
  const BEST_MATCHES_TIMEOUT_MS = 30000;
  const BEST_MATCHES_RETRIES = 2;
  const BEST_MATCHES_RETRY_DELAY_MS = 1500;
  // Stems: the hosted matcher splits a seed and a match into instruments and
  // measures each part. A song it hasn't dissected before takes GPU time, so
  // the wait is longer and a busy dissector (503) is retried a few times.
  const DISSECT_ENDPOINT = `${FALLBACK_HOSTED_SERVER}/api/anysong/dissect`;
  const DISSECT_TIMEOUT_MS = 90000;
  const DISSECT_RETRIES = 3;
  const DISSECT_RETRY_DELAY_MS = 2000;
  const DISSECT_CACHE_SIZE = 40;
  const DISSECT_FRAMES = 300; // 30 s lanes at 10 frames a second
  const DISSECT_STEMS = [
    ["vocals", "Vocals"],
    ["drums", "Drums"],
    ["bass", "Bass"],
    ["guitar", "Guitar"],
    ["piano", "Keys"],
    ["other", "Synths & other"],
  ];
  const DISSECT_DRUMS = [["kick", "Kick"], ["snare", "Snare"], ["hats", "Hats"]];
  // Raw facet values before calibration: cosine-like similarities (1 = alike)
  // or distances in the units below.
  const DISSECT_SIMILARITIES = new Set([
    "vocals", "drums", "bass", "guitar", "piano", "other",
    "groove", "tempo", "harmony", "bass_motion",
  ]);
  const DISSECT_BLURB =
    "Both songs split into vocals, drums, bass, guitar, keys and the rest, " +
    "then measured part by part: loudness over time, the beat grid, drum hits, " +
    "the sung and bass melody lines, and a one-bar drum pattern.";
  const DISSECT_PENDING =
    "Splitting both songs into instruments… a song that hasn't been dissected " +
    "before takes up to half a minute.";
  // The seed's ISRC lets the matcher find songs Deezer's search doesn't list
  // where the server runs, and the exact recording. The lookup is optional:
  // past this the request goes by title and artist alone.
  const ISRC_LOOKUP_TIMEOUT_MS = 1500;
  const ISRC_PATTERN = /^[A-Z]{2}[A-Z0-9]{3}\d{7}$/;
  const SPOTIFY_ID_ALPHABET =
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
  // How long a ready Sounds like list waits for a slower Best matches list
  // before the page opens with Best matches still loading.
  const BEST_MATCHES_PATIENCE_MS = 4000;
  const RESULT_VIEWS = [
    ["best", "Best matches"],
    ["listeners", "Listeners also play"],
    ["melody", "Melody match"],
    ["sounds", "Sounds like"],
  ];
  const BEST_MATCHES_BLURB =
    "Likely covers, samples and shared melodies first, then other artists' " +
    "songs a model ranks closest, from five signals: sound, scene, playlists, " +
    "melody and samples. Turn on Audio only to rank by sound alone.";
  // Servers whose main list follows sound and the listener ranker.
  const NUDGED_BEST_MATCHES_BLURB =
    "Likely covers, samples and shared melodies first, then the songs that " +
    "sound most alike, nudged toward artists this song's listeners also play. " +
    "Turn on Audio only to rank by sound alone.";
  const AUDIO_ONLY_BEST_MATCHES_BLURB =
    "Likely covers, samples and shared melodies first, then the songs that " +
    "sound most alike.";
  // Servers without the main list send Best matches with listener picks in it.
  const MIXED_BEST_MATCHES_BLURB =
    "Likely covers, samples and shared melodies first, then similar-sounding " +
    "songs by artists this song's listeners also play, marked Listeners also " +
    "play. Turn on Audio only to leave those out.";
  const LISTENERS_BLURB =
    "Songs by other artists that this song's listeners also play, one per " +
    "artist. The likelier its listeners are to play a song, and the closer " +
    "it sounds, the higher it ranks.";
  // The hosted "best" list: its audio head is the melody and sample matches
  // fused into one ranking; with listener data, listener picks follow it.
  const MELODY_BLURB =
    "Likely covers, samples and shared melodies first, from the melody and " +
    "sample matches fused into one ranking, then similar-sounding songs by " +
    "artists this song's listeners also play, marked Listeners also play. " +
    "Turn on Audio only to leave those out.";
  const AUDIO_ONLY_MELODY_BLURB =
    "Likely covers, samples and shared melodies: the melody and sample " +
    "matches fused into one ranking, by audio alone.";
  const AUDIO_ONLY_LISTENERS_NOTE =
    "Audio only is on, and this list comes from what this song's listeners " +
    "also play. Turn Audio only off to see it.";
  const NO_LISTENERS_NOTE =
    "We don't know what this artist's listeners play yet, so there's no list " +
    "here. Best matches still has this song's closest matches.";
  // A Best row the listener ranker placed, rather than the audio alone.
  const LISTENER_PICK_LABEL = "Listeners also play";
  const LISTENER_PICK_TITLE =
    "Placed by what this song's listeners also play, not by audio alone";
  const LISTENER_PICK_ARIA =
    "placed by what this song's listeners also play rather than by audio alone";
  const AUDIO_ONLY_TITLE =
    "Rank by audio alone, leaving out what this song's listeners also play";
  const BEST_MATCHES_OFFER =
    "Your local companion keeps searches on this computer. Best matches only " +
    "run on the hosted matcher, so getting them sends this song's title, " +
    "artist and recording code (ISRC) to soundalike.yassin.app.";
  const SOUNDS_LIKE_BLURB =
    "Songs with the same overall sound: low end, dynamics and tone.";
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
  // Set when Spotify's right-click menu failed to render around a row.
  let nativeMenusBroken = false;
  let RowBoundary;
  let resultsState = null;
  // Searches of this session by seed, as their pages were left.
  const keptStates = new Map();
  let latestRestored = false;
  let activePage = null;
  // The open Stems view over the results list, and dissections by song pair.
  let dissectView = null;
  const dissections = new Map();
  // Rows matched after their list started playing: { last, uris }.
  let lateMatches = null;
  let routeMountTimer;
  let cacheSaveTimer;
  // While Play this part runs: the part's place in the song, and where the
  // song was when it was already the current one, to put back afterwards.
  let partPlay = null;
  // A fresh answer said the server can match a part of the preview.
  let serverMatchesParts = false;
  let searchSequence = 0;
  const pendingRecommendations = new Map();
  const pendingIsrcLookups = new Map();
  const pendingSpotifyTracks = new Map();
  const pendingRelatedArtists = new Map();
  const relatedArtistCache = new Map();
  let persistentCache;
  let fallbackTrackSearchDefinition;

  // Two copies of the runtime can load: a manual install of this file and the
  // one Marketplace's bootstrap injects. The first to start claims the slot
  // before the readiness wait, so the other steps aside on every start
  // instead of whichever polls first.
  if (!window.__soundalikeRuntimeVersion) window.__soundalikeRuntimeVersion = RUNTIME_SEMANTIC_VERSION;

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
  if (
    window.__soundalikeRuntimeVersion !== RUNTIME_SEMANTIC_VERSION ||
    window.__soundalikeContextMenuItem
  ) return;
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

  function savedAudioOnly() {
    try {
      return Spicetify.LocalStorage?.get(AUDIO_ONLY_KEY) === "true";
    } catch {
      return false;
    }
  }

  function saveAudioOnly(audioOnly) {
    try {
      if (audioOnly) Spicetify.LocalStorage?.set(AUDIO_ONLY_KEY, "true");
      else Spicetify.LocalStorage?.remove(AUDIO_ONLY_KEY);
    } catch {}
  }

  // The hosted tab a new search opens on: the one last chosen.
  function savedResultView() {
    try {
      const view = Spicetify.LocalStorage?.get(RESULT_VIEW_KEY);
      return REMEMBERED_VIEWS.includes(view) ? view : "best";
    } catch {
      return "best";
    }
  }

  function saveResultView(view) {
    if (!REMEMBERED_VIEWS.includes(view)) return;
    try {
      Spicetify.LocalStorage?.set(RESULT_VIEW_KEY, view);
    } catch {}
  }

  function searchHistory() {
    try {
      const saved = JSON.parse(Spicetify.LocalStorage?.get(HISTORY_KEY) || "[]");
      return Array.isArray(saved)
        ? saved
          .filter((item) =>
            typeof item?.uri === "string" &&
            item.uri.startsWith("spotify:track:") &&
            typeof item.title === "string")
          .slice(0, HISTORY_SIZE)
        : [];
    } catch {
      return [];
    }
  }

  // The newest search leads; a song searched again moves back to the front.
  function rememberSearch(state) {
    const entry = {
      uri: state.seedUri,
      title: String(state.seedTrack?.name || ""),
      artist: seedArtistNames(state.seedTrack).join(", "),
      image: artworkUrl(state.seedTrack),
    };
    const earlier = searchHistory().filter((item) => item.uri !== entry.uri);
    try {
      Spicetify.LocalStorage?.set(
        HISTORY_KEY,
        JSON.stringify([entry, ...earlier].slice(0, HISTORY_SIZE))
      );
    } catch {}
  }

  // Clear forgets every search but the one on screen.
  function clearSearchHistory() {
    try {
      Spicetify.LocalStorage?.remove(HISTORY_KEY);
    } catch {}
    keptStates.clear();
    if (resultsState) keptStates.set(resultsState.seedUri, resultsState);
  }

  function keepState(state) {
    keptStates.delete(state.seedUri);
    keptStates.set(state.seedUri, state);
    for (const uri of keptStates.keys()) {
      if (keptStates.size <= HISTORY_SIZE) break;
      keptStates.delete(uri);
    }
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

  function feedbackPayload(data, displayed, installNonce, sessionNonce, form, audioOnly = false) {
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
      selection_policy: audioOnly
        ? FEEDBACK_AUDIO_ONLY_SELECTION_POLICY
        : FEEDBACK_SELECTION_POLICY,
      source: data.__soundalikeSource === "local" ? "local" : "hosted",
      selection,
      reasons: detailed ? form.reasons() : [],
      note: detailed ? form.note().slice(0, 280) : "",
    };
  }

  // audioOnly is the mode the displayed list was ordered in.
  function setupFeedbackPrompt(page, data, displayed, session = null, audioOnly = false) {
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
          form,
          audioOnly
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
        "[soundalike] Primary hosted endpoint is unavailable or outdated; using web app fallback.",
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
        "[soundalike] Cacheable web app endpoint failed; using legacy endpoint.",
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

  function deezerId(value) {
    const id = Number(value);
    return Number.isSafeInteger(id) && id > 0 ? id : null;
  }

  // The server's sound match: 0 to 100, the percentile of a song's sound score
  // among the matches its lists show for typical songs. Its overall match (the
  // main list and Listeners also play) scores the list's own ranking the same
  // way and never rises down the list.
  function matchPercent(value) {
    return Number.isInteger(value) && value >= 0 && value <= 100 ? value : null;
  }

  // Keeps only what the page shows, so cached lists stay small. Deezer ids stay
  // so a match can be dissected next to its seed (Stems).
  function compactBestMatches(body) {
    const query = body?.query || {};
    const withId = (item, compact) => {
      const id = deezerId(item?.id);
      const match = matchPercent(item?.match);
      const overall = matchPercent(item?.overall);
      const kept = {
        ...compact,
        ...(match === null ? {} : { match }),
        ...(overall === null ? {} : { overall }),
      };
      return id ? { id, ...kept } : kept;
    };
    const channelRanks = (item) => item?.ranks && typeof item.ranks === "object"
      ? Object.entries(item.ranks)
        .filter(([list, rank]) =>
          /^[a-z]{1,24}$/.test(list) && Number.isInteger(rank) && rank > 0
        )
        .map(([list, rank]) => `${list} #${rank}`)
        .join(" · ")
      : "";
    const compactList = (name, extra) =>
      (Array.isArray(body?.results?.[name]) ? body.results[name] : [])
        .map((item) => withId(item, {
          title: String(item?.title || "").trim(),
          artist: String(item?.artist || "").trim(),
          ...extra(item),
        }))
        .filter((item) => item.title && item.artist)
        .slice(0, HOSTED_RESULT_COUNT);
    // What placed a row: the audio head, the sound order, the five-signal
    // model (consensus) or the listener ranker.
    const basis = (item, allowed) =>
      allowed.includes(item?.basis) ? { basis: item.basis } : {};
    // Servers before the main list send Best matches with listener picks in it;
    // the Best matches tab falls back to that list.
    const results = compactList("best", (item) => ({
      matchNote: channelRanks(item),
      ...basis(item, ["audio", "listeners"]),
    }));
    // Only the audio head's rows keep the channel ranks that put them there.
    const main = compactList("main", (item) => ({
      matchNote: item?.basis === "audio" ? channelRanks(item) : "",
      ...basis(item, ["audio", "sound", "consensus"]),
    }));
    const listeners = compactList("listeners", () => ({}));
    const total = Number(body?.timings_ms?.total);
    return {
      query: withId(query, {
        title: String(query.title || ""),
        artist: String(query.artist || ""),
        source: query.source === "live" ? "live" : "catalog",
        language: String(query.language || "unknown"),
        // The seed had no language of its own; the server guessed one.
        ...(query.language_guessed === true ? { languageGuessed: true } : {}),
        // The server's guess gates the main list alone; the query's language stays "unknown".
        ...(/^[a-z]{2,3}$/.test(String(query.main_language_guess || ""))
          ? { mainLanguageGuess: String(query.main_language_guess) } : {}),
        languageMode: query.language_mode === "all" ? "all" : "same",
        bpm: Number.isFinite(query.bpm) && query.bpm > 0 ? query.bpm : null,
        // "unknown_artist": no listener data, so Listeners also play is empty.
        ...(["on", "off", "unknown_artist"].includes(query.scene) ? { scene: query.scene } : {}),
        // The server can match a part of the preview: its limits, and where
        // the preview starts in the song when it knows.
        ...(compactParts(query.parts) ? { parts: compactParts(query.parts) } : {}),
        // This answer is for a part: [start, end] seconds of the preview.
        ...(compactPart(query.part) ? { part: compactPart(query.part) } : {}),
      }),
      results,
      main,
      listeners,
      totalMs: Number.isFinite(total) && total > 0 ? Math.round(total) : 0,
      serverCached: body?.cached === true,
    };
  }

  // The Best matches tab's list: the main list, or Best matches from a server
  // without it.
  function bestRows(data) {
    return data?.main?.length ? data.main : data?.results || [];
  }

  function compactParts(parts) {
    if (!parts || typeof parts !== "object") return null;
    const total = Number(parts.preview_s) || 30;
    const min = Math.max(PART_STEP_SECONDS, Number(parts.min_s) || 10);
    const max = Math.min(PART_LONGEST_SECONDS, Number(parts.max_s) || PART_LONGEST_SECONDS);
    if (!(min <= max && max <= total)) return null;
    const offset = Number.isFinite(parts.song_offset_s) && parts.song_offset_s >= 0
      ? parts.song_offset_s
      : null;
    return { total, min, max, offset };
  }

  function compactPart(part) {
    const start = Number(part?.start);
    const end = Number(part?.end);
    return part && Number.isFinite(start) && Number.isFinite(end) && start >= 0 && end > start
      ? [start, end]
      : null;
  }

  const partText = (part) => `${part[0]}-${part[1]}`;

  // part: [start, end] seconds of the preview to match instead of all of it.
  async function requestBestMatchesUncached(
    title,
    artist,
    langMode = "same",
    isrc = null,
    audioOnly = false,
    part = null
  ) {
    // The policy keeps HTTP-cached responses from before a format change out.
    const params = new URLSearchParams({
      title,
      artist,
      k: String(HOSTED_RESULT_COUNT),
      lang: langMode,
      policy: BEST_MATCHES_POLICY,
    });
    if (audioOnly) params.set("scene", "0");
    if (isrc) params.set("isrc", isrc);
    if (part) params.set("part", partText(part));
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
      // A server without Match a part refuses the span: ask for the whole preview.
      if (params.has("part") && (response.status === 400 || response.status === 404)) {
        params.delete("part");
        continue;
      }
      if (!response.ok || !body?.ok) {
        throw bestMatchesError(response.ok ? 0 : response.status);
      }
      const data = compactBestMatches(body);
      if (!bestRows(data).length) {
        throw bestMatchesError(0, "No best matches came back for this song.");
      }
      if (data.query.parts) serverMatchesParts = true;
      return data;
    }
  }

  // Audio-only and listener-aware lists are saved apart, so they never mix,
  // and so is the list of each part of the preview.
  function bestMatchesCacheId(seedUri, langMode = "same", audioOnly = false, part = null) {
    return `best:${BEST_MATCHES_POLICY}:${audioOnly ? "audio" : "listeners"}:${langMode}:${seedUri}` +
      (part ? `:part:${partText(part)}` : "");
  }

  // Lists saved before Stems have no Deezer ids, so they are fetched once more;
  // so are lists saved before the server could match a part, once it can.
  function savedBestMatches(seedUri, langMode = "same", audioOnly = false, part = null) {
    const cached = readCacheEntry(
      "recommendations",
      bestMatchesCacheId(seedUri, langMode, audioOnly, part)
    );
    return bestRows(cached).length && deezerId(cached.query?.id) &&
      (cached.query.parts || !serverMatchesParts)
      ? { data: cached, cached: true }
      : null;
  }

  async function requestBestMatches(
    title, artist, seedUri, langMode = "same", audioOnly = false, part = null
  ) {
    const saved = savedBestMatches(seedUri, langMode, audioOnly, part);
    if (saved) return saved;
    const cacheId = bestMatchesCacheId(seedUri, langMode, audioOnly, part);
    if (pendingRecommendations.has(cacheId)) {
      return pendingRecommendations.get(cacheId);
    }
    const request = lookupSeedIsrc(seedUri)
      .then((isrc) => requestBestMatchesUncached(title, artist, langMode, isrc, audioOnly, part))
      .then((data) => {
        // A server without Match a part answered for the whole preview.
        writeCacheEntry(
          "recommendations",
          data.query.part ? cacheId : bestMatchesCacheId(seedUri, langMode, audioOnly),
          data
        );
        return { data, cached: false };
      })
      .finally(() => pendingRecommendations.delete(cacheId));
    pendingRecommendations.set(cacheId, request);
    return request;
  }

  function spotifyIdToHex(id) {
    const viaSpicetify = Spicetify.URI?.idToHex?.(id);
    if (typeof viaSpicetify === "string" && /^[0-9a-f]{32}$/i.test(viaSpicetify)) {
      return viaSpicetify.toLowerCase();
    }
    if (typeof BigInt !== "function" || !/^[0-9A-Za-z]{1,22}$/.test(id || "")) return null;
    let value = 0n;
    for (const char of id) {
      value = value * 62n + BigInt(SPOTIFY_ID_ALPHABET.indexOf(char));
    }
    const hex = value.toString(16);
    return hex.length > 32 ? null : hex.padStart(32, "0");
  }

  function isrcFrom(value) {
    const code = String(value || "").replace(/[\s-]/g, "").toUpperCase();
    return ISRC_PATTERN.test(code) ? code : null;
  }

  function withTimeout(promise, timeoutMs) {
    let timer;
    return Promise.race([
      promise,
      new Promise((resolve) => {
        timer = setTimeout(() => resolve(undefined), timeoutMs);
      }),
    ]).finally(() => clearTimeout(timer));
  }

  // Spotify's metadata service answers in protobuf on current clients (Cosmos
  // then fails to parse it), so read the reply as text: JSON when it is JSON,
  // else the ISRC that follows the "isrc" tag in the binary reply.
  async function metadataIsrc(gid) {
    let token = Spicetify.Platform?.Session?.accessToken;
    try {
      const state = await Spicetify.Platform?.AuthorizationAPI?.getState?.();
      token = state?.token?.accessToken || token;
    } catch {
      // keep the session token
    }
    if (!token || typeof fetch !== "function") return undefined;
    const response = await fetch(
      `https://spclient.wg.spotify.com/metadata/4/track/${gid}?market=from_token`,
      { headers: { Authorization: `Bearer ${token}`, Accept: "application/json" } }
    );
    if (!response.ok) return undefined;
    const text = await response.text();
    try {
      const meta = JSON.parse(text);
      return isrcFrom((meta?.external_id || []).find?.(
        (entry) => String(entry?.type).toLowerCase() === "isrc")?.id);
    } catch {
      const found = text.match(/isrc[\s\S]{1,3}([A-Z]{2}[A-Z0-9]{3}[0-9]{7})/);
      return found ? isrcFrom(found[1]) : null;
    }
  }

  // Spotify's GraphQL track query has no ISRC, so ask its metadata service
  // and the Web API at once and take the first ISRC either gives.
  async function fetchSeedIsrc(trackId) {
    const gid = spotifyIdToHex(trackId);
    const sources = [
      gid && metadataIsrc(gid),
      Spicetify.CosmosAsync.get(`https://api.spotify.com/v1/tracks/${trackId}`)
        .then((track) => isrcFrom(track?.external_ids?.isrc)),
    ].filter(Boolean).map((source) => source.catch(() => undefined));
    const answers = sources.map((source) => source.then((isrc) => isrc || Promise.reject()));
    try {
      return await Promise.any(answers);
    } catch {
      // Both answered without one (null: remembered) or failed (undefined).
      const settled = await Promise.all(sources);
      return settled.every((value) => value === null) ? null : undefined;
    }
  }

  function lookupSeedIsrc(seedUri) {
    const trackId = seedUri?.split(":track:")[1];
    if (!trackId || !Spicetify.CosmosAsync?.get) return Promise.resolve(null);
    const cached = readCacheEntry("spotifyTracks", `isrc:${seedUri}`);
    if (cached) return Promise.resolve(cached.isrc || null);
    if (!pendingIsrcLookups.has(seedUri)) {
      const lookup = withTimeout(fetchSeedIsrc(trackId), ISRC_LOOKUP_TIMEOUT_MS)
        .then((isrc) => {
          if (isrc !== undefined) writeCacheEntry("spotifyTracks", `isrc:${seedUri}`, { isrc });
          return isrc || null;
        })
        .catch(() => null)
        .finally(() => pendingIsrcLookups.delete(seedUri));
      pendingIsrcLookups.set(seedUri, lookup);
    }
    return pendingIsrcLookups.get(seedUri);
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

  // The older Sounds like service stands in when the hosted matcher can't take
  // the song, for example because it couldn't get the audio. Null means Best
  // matches came back, so no stand-in is needed.
  async function soundsAfterBest(best, seedTrack, artist, seedUri) {
    await best.promise;
    if (best.ok) return null;
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
          listenerTracks: [],
        }
      : {
          status: "error",
          error: request.error?.message || bestMatchesMessage(0),
          tracks: [],
        };
  }

  function applySoundsOutcome(state, request) {
    // Best matches came back, so no Sounds like list was asked for.
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

  // The menu items' handler. Spicetify passes them more than the song.
  function findSoundalikes(uris) {
    return searchSoundalikes(uris);
  }

  // quiet: a search reopened from Recent or Spotify's Back says nothing while
  // it loads. navigate: false keeps Spotify's history as it is.
  async function searchSoundalikes(uris, { quiet = false, navigate = true } = {}) {
    const startedAt = nowMs();
    const searchId = ++searchSequence;
    const id = uris[0].split(":track:")[1];
    const seedUri = `spotify:track:${id}`;
    if (!quiet) Spicetify.showNotification("Finding soundalikes…");
    let seedTrack;
    let artist;
    lookupSeedIsrc(seedUri);
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
    const audioOnly = savedAudioOnly();
    const best = askFirst
      ? null
      : settledRequest(requestBestMatches(seedTrack.name, artist, seedUri, "same", audioOnly));
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
      seedTrack,
      seedUri,
      view: showSounds ? "sounds" : savedResultView(),
      viewChosen: false,
      showAllLanguages: false,
      audioOnly,
      companion: askFirst,
      timing: { startedAt, initialReadyMs },
      best: best ? pendingResultList() : offeredBestList(seedUri, audioOnly),
      sounds: pendingResultList(),
      // Songs shown per hosted list once Show more was used.
      limits: {},
    };
    if (best?.done) applyBestOutcome(state, best);
    if (sounds.done) applySoundsOutcome(state, sounds);
    console.log(
      `[soundalike] results page ready in ${initialReadyMs} ms ` +
      `(seed metadata ${Math.max(0, Math.round(seedReadyAt - startedAt))} ms).`
    );
    showResultsPage(state, navigate);
    if (best && !best.done) {
      best.promise.then(() => settleBestMatches(state, audioOnly, best));
    }
    if (!sounds.done) {
      sounds.promise.then(() => settleResultsView(state, "sounds", sounds));
    }
  }

  // A Best matches list saved by an earlier search needs no request, so it
  // shows straight away; otherwise the tab offers to ask the hosted matcher.
  function offeredBestList(seedUri, audioOnly = false) {
    const saved = savedBestMatches(seedUri, "same", audioOnly);
    return saved
      ? { status: "ready", data: saved.data, cached: true, tracks: [] }
      : { status: "offer", tracks: [] };
  }

  // Asks for this search's Best matches in the page's language and audio
  // modes, for the part of the preview it matches. A list that arrives after
  // the mode or the part changed again is dropped.
  function requestModeBestMatches(state) {
    const { seedTrack, seedUri, audioOnly } = state;
    const part = state.part || null;
    state.best = pendingResultList();
    state.limits = {};
    const request = settledRequest(requestBestMatches(
      seedTrack.name,
      seedArtistName(seedTrack),
      seedUri,
      state.showAllLanguages ? "all" : "same",
      audioOnly,
      part
    ));
    request.promise.then(() => settleBestMatches(state, audioOnly, request, part));
  }

  function settleBestMatches(state, audioOnly, request, part = null) {
    if (state.audioOnly !== audioOnly || !sameSpan(state.part, part)) return;
    // A server without Match a part answered for the whole preview.
    if (part && request.ok && !request.value.data.query.part) state.part = null;
    settleResultsView(state, "best", request);
  }

  function requestOfferedBestMatches() {
    const state = resultsState;
    if (!state || state.best.status !== "offer") return;
    requestModeBestMatches(state);
    refreshResultsPage(true);
  }

  function toggleBestLanguageMode() {
    const state = resultsState;
    if (!state || state.best.status !== "ready") return;
    state.showAllLanguages = !state.showAllLanguages;
    state.viewChosen = true;
    requestModeBestMatches(state);
    refreshResultsPage(true);
  }

  // Audio only is saved for later searches too. The open search asks again in
  // the new mode, except for a companion search's Best matches that weren't
  // asked for yet; a list saved in that mode still shows straight away.
  function toggleAudioOnly() {
    const state = resultsState;
    if (!state) return;
    state.audioOnly = !state.audioOnly;
    saveAudioOnly(state.audioOnly);
    state.viewChosen = true;
    if (state.best.status === "offer") {
      state.best = offeredBestList(state.seedUri, state.audioOnly);
    } else {
      requestModeBestMatches(state);
    }
    refreshResultsPage(true);
    activePage?.view?.querySelector?.(".sa-audio-only")?.focus?.();
  }

  // ---- Match a part -------------------------------------------------------
  // The header's "Match a part" opens a 30-second timeline of the preview with
  // a span to move; Apply asks for the hosted lists of that span. The choice
  // belongs to the open search and is not saved.
  const sameSpan = (left, right) =>
    (left ? partText(left) : "") === (right ? partText(right) : "");

  function clockText(seconds) {
    const whole = Math.max(0, Math.round(seconds));
    return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
  }

  // What the search's server can match, kept while a list reloads; null with a
  // server that can't match parts.
  function partLimits(state) {
    if (state.best.status === "ready") state.partsInfo = state.best.data.query?.parts || null;
    return state.partsInfo || null;
  }

  // "0:58–1:13 of the song", or of the preview when its place is unknown.
  function spanLabel(state, part) {
    const offset = partLimits(state)?.offset ?? null;
    const base = offset === null ? 0 : offset;
    return `${clockText(base + part[0])}–${clockText(base + part[1])} of the ${
      offset === null ? "preview" : "song"
    }`;
  }

  // Keeps a span on the step grid, inside the preview and within the lengths.
  function fitSpan(start, end, limits) {
    const snap = (value) => Math.round(value / PART_STEP_SECONDS) * PART_STEP_SECONDS;
    const length = Math.min(limits.max, Math.max(limits.min, snap(end - start)));
    const from = Math.min(limits.total - length, Math.max(0, snap(start)));
    return [from, from + length];
  }

  function partOpenMarkup(state, view) {
    if (view === "sounds" || state.best.status === "offer" || !partLimits(state)) return "";
    return `<button class="sa-part-open" type="button" aria-expanded="${Boolean(state.partDraft)}" ` +
      `aria-controls="sa-part">Match a part <span class="sa-beta">beta</span></button>`;
  }

  function partChipMarkup(state, view) {
    if (!state.part) return "";
    return `<div class="sa-part-chip" role="status"><span>Matching ${esc(spanLabel(state, state.part))}` +
      `${view === "sounds" ? " (not on Sounds like)" : ""}</span> · ` +
      `<button class="sa-part-clear" type="button" title="Match the whole preview again">whole preview</button></div>`;
  }

  function partPanelMarkup(state, view) {
    const limits = partLimits(state);
    const draft = state.partDraft;
    if (!draft || !limits || view === "sounds") return "";
    const base = limits.offset === null ? 0 : limits.offset;
    const lengths = [];
    for (let length = limits.min; length <= limits.max; length += PART_STEP_SECONDS) lengths.push(length);
    const ticks = [];
    for (let at = 0; at <= limits.total; at += 10) ticks.push(at);
    const label = `${spanLabel(state, [draft.start, draft.end])} · ${draft.end - draft.start} s`;
    const unchanged = sameSpan(state.part, [draft.start, draft.end]);
    return `<section class="sa-part" id="sa-part" aria-label="Match a part of the preview">
        <p class="sa-part-intro"><span class="sa-beta">New · beta</span> ${esc(PART_INTRO)}</p>
        <div class="sa-part-track">
          <div class="sa-part-span" role="slider" tabindex="0" aria-label="Part of the preview to match" aria-valuemin="0" aria-valuemax="${limits.total - limits.min}" aria-valuenow="${draft.start}" aria-valuetext="${esc(label)}" style="left:${(100 * draft.start) / limits.total}%;width:${(100 * (draft.end - draft.start)) / limits.total}%"><i class="sa-part-edge" data-edge="start"></i><i class="sa-part-edge" data-edge="end"></i></div>
        </div>
        <div class="sa-part-ticks" aria-hidden="true">${ticks.map((at) => `<span>${clockText(base + at)}</span>`).join("")}</div>
        <p class="sa-part-label">${esc(label)}</p>
        <div class="sa-part-controls">
          <span class="sa-part-lengths" role="group" aria-label="Length of the part">${lengths.map((length) =>
            `<button class="sa-part-length" type="button" data-length="${length}" aria-pressed="${length === draft.end - draft.start}">${length} s</button>`
          ).join("")}</span>
          <button class="sa-part-play" type="button" title="${esc(PART_PLAY_TITLE)}">Play this part</button>
          <button class="sa-part-apply" type="button"${unchanged ? " disabled" : ""}>Apply</button>
          <button class="sa-part-cancel" type="button">Cancel</button>
        </div>
        <p class="sa-part-note">${esc(PART_LISTS_NOTE)} ${esc(PART_PLACE_NOTE)} ${esc(PART_KEYS_NOTE)}</p>
      </section>`;
  }

  function togglePartPicker() {
    const state = resultsState;
    const limits = state && partLimits(state);
    if (!limits) return;
    stopPartPlay(true);
    if (state.partDraft) {
      state.partDraft = null;
    } else {
      const [start, end] = fitSpan(...(state.part || PART_DEFAULT), limits);
      state.partDraft = { start, end };
    }
    refreshResultsPage();
    activePage?.view?.querySelector?.(state.partDraft ? ".sa-part-span" : ".sa-part-open")?.focus?.();
  }

  // part: [start, end] to match that span, null to match the whole preview again.
  function applyPart(part) {
    const state = resultsState;
    if (!state || state.best.status === "offer") return;
    // A part that is playing plays on to its end while its lists load.
    state.part = part;
    state.partDraft = null;
    state.viewChosen = true;
    requestModeBestMatches(state);
    refreshResultsPage(true);
    activePage?.view?.querySelector?.(part ? ".sa-part-clear" : ".sa-part-open")?.focus?.();
  }

  // Moves the open picker's span and redraws it in place: rebuilding the page
  // would end a drag and drop keyboard focus.
  function movePartDraft(start, end) {
    const state = resultsState;
    const limits = state && partLimits(state);
    const draft = state?.partDraft;
    if (!draft || !limits) return;
    const [from, to] = fitSpan(start, end, limits);
    if (from === draft.start && to === draft.end) return;
    draft.start = from;
    draft.end = to;
    const page = activePage?.view;
    const label = `${spanLabel(state, [from, to])} · ${to - from} s`;
    const span = page?.querySelector?.(".sa-part-span");
    if (span) {
      if (span.style) {
        span.style.left = `${(100 * from) / limits.total}%`;
        span.style.width = `${(100 * (to - from)) / limits.total}%`;
      }
      span.setAttribute?.("aria-valuenow", String(from));
      span.setAttribute?.("aria-valuetext", label);
    }
    const text = page?.querySelector?.(".sa-part-label");
    if (text) text.textContent = label;
    Array.from(page?.querySelectorAll?.(".sa-part-length") || []).forEach((button) => {
      button.setAttribute?.("aria-pressed", String(Number(button.dataset?.length) === to - from));
    });
    const apply = page?.querySelector?.(".sa-part-apply");
    if (apply) apply.disabled = sameSpan(state.part, [from, to]);
  }

  // Arrow keys move the part a step; with Shift they change its length.
  function onPartKey(event) {
    const state = resultsState;
    const limits = state && partLimits(state);
    const draft = state?.partDraft;
    if (!draft || !limits) return;
    const { start, end } = draft;
    const step = event.key === "ArrowRight" || event.key === "ArrowUp"
      ? PART_STEP_SECONDS
      : event.key === "ArrowLeft" || event.key === "ArrowDown"
        ? -PART_STEP_SECONDS
        : 0;
    if (step) {
      if (!event.shiftKey) movePartDraft(start + step, end + step);
      else {
        // A longer part that would pass the end of the preview grows backwards.
        const length = Math.min(limits.max, Math.max(limits.min, end - start + step));
        const from = Math.min(start, limits.total - length);
        movePartDraft(from, from + length);
      }
    } else if (event.key === "Home") movePartDraft(0, end - start);
    else if (event.key === "End") movePartDraft(limits.total - (end - start), limits.total);
    else return;
    event.preventDefault?.();
  }

  // Dragging the span moves it and dragging an edge resizes it; a press
  // outside the span brings the span's middle there.
  function bindPartDrag(track) {
    let drag = null;
    const seconds = (event) => {
      const box = track.getBoundingClientRect?.();
      const total = partLimits(resultsState)?.total || 30;
      return box?.width > 0 ? ((event.clientX - box.left) / box.width) * total : 0;
    };
    const move = (event) => {
      const state = resultsState;
      const limits = state && partLimits(state);
      const draft = state?.partDraft;
      if (!drag || !draft || !limits) return;
      const at = seconds(event);
      const { start, end } = draft;
      const within = (length) => Math.min(limits.max, Math.max(limits.min, length));
      if (drag.mode === "move") movePartDraft(at - drag.grab, at - drag.grab + (end - start));
      else if (drag.mode === "start") movePartDraft(Math.max(0, end - within(end - at)), end);
      else movePartDraft(start, Math.min(limits.total, start + within(at - start)));
    };
    track.onpointerdown = (event) => {
      const draft = resultsState?.partDraft;
      if (!draft) return;
      const at = seconds(event);
      const edge = event.target?.dataset?.edge;
      const inside = at >= draft.start && at <= draft.end;
      drag = {
        mode: edge === "start" || edge === "end" ? edge : "move",
        grab: inside ? at - draft.start : (draft.end - draft.start) / 2,
      };
      track.setPointerCapture?.(event.pointerId);
      event.preventDefault?.();
      if (!inside && drag.mode === "move") move(event);
    };
    track.onpointermove = move;
    track.onpointerup = track.onpointercancel = () => {
      drag = null;
    };
    // Without this a flicked touch drag ends in a fling, and the browser
    // takes the next tap (on Apply, say) as the tap that stops it.
    track.addEventListener?.("touchmove", (event) => event.preventDefault(), { passive: false });
  }

  // The panel plays no previews, so the part is played in Spotify: the song
  // from about where the part starts, until the part is over. The preview's
  // place in the song is the server's, or the usual one for a song of a minute
  // and a half or more. A song that was already the current one goes back to
  // where it was, playing or paused as it was; a song started for the part is
  // paused. Nothing is put back once the listener pauses, moves in the song
  // or plays something else: the player is theirs again.
  async function playPart() {
    const state = resultsState;
    const draft = state?.partDraft;
    if (!draft) return;
    const { start, end } = draft;
    // Playing the part again keeps what was there before the first time.
    const before = partPlay && partPlay.state === state
      ? partPlay.before
      : seedIsCurrent(state) && typeof Spicetify.Player?.getProgress === "function"
        ? {
          position: Math.max(0, Number(Spicetify.Player.getProgress()) || 0),
          playing: Boolean(Spicetify.Player.isPlaying?.()),
        }
        : null;
    stopPartPlay(false);
    if (!seedIsCurrent(state)) {
      await playSeedSong(state);
      for (let waited = 0; !seedIsCurrent(state) && waited < PART_PLAY_WAIT_MS; waited += 100) {
        await delay(100);
      }
    }
    if (state !== resultsState || !state.partDraft || !seedIsCurrent(state)) return;
    const known = partLimits(state)?.offset ?? null;
    const duration = Number(Spicetify.Player?.getDuration?.()) || 0;
    const offset = known !== null
      ? known
      : duration >= 90000 ? USUAL_PREVIEW_OFFSET_SECONDS : null;
    if (offset === null || typeof Spicetify.Player?.seek !== "function") {
      Spicetify.showNotification("Soundalike doesn't know where the preview sits in this song.", true);
      return;
    }
    const run = {
      state,
      before,
      from: Math.round((offset + start) * 1000),
      to: Math.round((offset + end) * 1000),
      started: false,
      polls: 0,
      timer: null,
    };
    partPlay = run;
    Spicetify.Player.seek(run.from);
    if (!Spicetify.Player.isPlaying?.()) Spicetify.Player.play?.();
    run.timer = setTimeout(watchPartPlay, PART_PLAY_POLL_MS, run);
  }

  function watchPartPlay(run) {
    if (partPlay !== run) return;
    // Another song plays now.
    if (!seedIsCurrent(run.state)) return stopPartPlay(false);
    run.polls += 1;
    const playing = Boolean(Spicetify.Player.isPlaying?.());
    const progress = Number(Spicetify.Player.getProgress?.());
    // A player that tells no position is followed by the clock.
    const at = Number.isFinite(progress) ? progress : run.from + run.polls * PART_PLAY_POLL_MS;
    const inPart = at >= run.from - PART_PLAY_SLACK_MS && at <= run.to + PART_PLAY_SLACK_MS;
    if (!run.started) {
      // Spotify takes a moment to move to the part and play.
      if (playing && inPart) {
        run.started = true;
        run.polls = 0;
      } else if (run.polls * PART_PLAY_POLL_MS >= PART_PLAY_WAIT_MS) {
        stopPartPlay(true);
        Spicetify.showNotification("Spotify didn't start playing this part.", true);
        return;
      }
    } else if (!playing || !inPart) {
      // Paused or moved in the song by the listener.
      return stopPartPlay(false);
    } else if (at >= run.to) {
      return stopPartPlay(true);
    }
    run.timer = setTimeout(watchPartPlay, PART_PLAY_POLL_MS, run);
  }

  // Ends Play this part. restore: put the song back as it was before the part
  // (see playPart); otherwise the player is left as it is.
  function stopPartPlay(restore) {
    const run = partPlay;
    if (!run) return;
    clearTimeout(run.timer);
    partPlay = null;
    if (!restore || !seedIsCurrent(run.state)) return;
    const playing = Boolean(Spicetify.Player.isPlaying?.());
    if (run.before) {
      Spicetify.Player.seek(run.before.position);
      if (run.before.playing && !playing) Spicetify.Player.play?.();
    }
    if (playing && !run.before?.playing) Spicetify.Player.pause?.();
  }

  function bindPartPicker(page, state) {
    const open = page.querySelector(".sa-part-open");
    if (open) open.onclick = togglePartPicker;
    const clear = page.querySelector(".sa-part-clear");
    if (clear) clear.onclick = () => applyPart(null);
    const span = page.querySelector(".sa-part-span");
    if (!state.partDraft || !span) return;
    span.onkeydown = onPartKey;
    const track = page.querySelector(".sa-part-track");
    if (track) bindPartDrag(track);
    Array.from(page.querySelectorAll(".sa-part-length") || []).forEach((button) => {
      button.onclick = () =>
        movePartDraft(state.partDraft.start, state.partDraft.start + Number(button.dataset.length));
    });
    const play = page.querySelector(".sa-part-play");
    if (play) play.onclick = playPart;
    const apply = page.querySelector(".sa-part-apply");
    if (apply) apply.onclick = () => applyPart([state.partDraft.start, state.partDraft.end]);
    const cancel = page.querySelector(".sa-part-cancel");
    if (cancel) cancel.onclick = togglePartPicker;
  }

  // Applies a list that finished after the page opened. A search that isn't
  // on screen any more still gets its list, for when it is reopened.
  function settleResultsView(state, kind, request) {
    const shown = state === resultsState;
    const tabsBefore = resultViews(state).map(([view]) => view).join();
    if (kind === "best") applyBestOutcome(state, request);
    else applySoundsOutcome(state, request);
    if (!shown) return;
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
    // Best matches, Listeners also play and Melody match share the hosted response.
    const list = shownView(state) === "sounds" ? "sounds" : "best";
    const tabsAfter = resultViews(state).map(([view]) => view).join();
    if (list === kind || tabsBefore !== tabsAfter) refreshResultsPage();
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

  // language: false skips the lyrics-language lookup, which only the Sounds
  // like list's language gate reads; a song saved without one gains it the
  // first time that list asks.
  async function findSpotifyTrack(result, language = true) {
    const cacheId = `result:${normalizeLabel(result.title)}::${normalizeLabel(result.artist)}`;
    const pendingId = language ? cacheId : `${cacheId}::plain`;
    if (pendingSpotifyTracks.has(pendingId)) return pendingSpotifyTracks.get(pendingId);
    const request = (async () => {
      const cached = readCacheEntry("spotifyTracks", cacheId);
      if (cached) {
        if (!language || languageStateFromTrack(cached)) return cached;
        const languageState = await getSpotifyTrackLanguageState(cached.uri);
        const refreshed = withLanguageState(cached, languageState);
        if (languageState.status !== "error") {
          writeCacheEntry("spotifyTracks", cacheId, compactSpotifyTrack(refreshed));
        }
        return refreshed;
      }
      const track = await findSpotifyTrackUncached(result, language);
      if (track) {
        writeCacheEntry("spotifyTracks", cacheId, compactSpotifyTrack(track));
      }
      return track;
    })().finally(() => pendingSpotifyTracks.delete(pendingId));
    pendingSpotifyTracks.set(pendingId, request);
    return request;
  }

  async function findSpotifyTrackUncached(result, language = true) {
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
    const languagePromise = language ? getSpotifyTrackLanguageState(match.uri) : null;
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
    if (!languagePromise) return { ...resolvedTrack };
    return withLanguageState({
      ...resolvedTrack,
    }, await languagePromise);
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

  const PLAY_PATH = "M7.05 3.606 20.54 11.394a.7.7 0 0 1 0 1.212L7.05 20.394A.7.7 0 0 1 6 19.788V4.212a.7.7 0 0 1 1.05-.606";
  const PAUSE_BAR = "a.7.7 0 0 0-.7.7v16.6a.7.7 0 0 0 .7.7h2.6a.7.7 0 0 0 .7-.7V3.7a.7.7 0 0 0-.7-.7z";
  const PAUSE_PATH = `M5.7 3${PAUSE_BAR}m10 0${PAUSE_BAR}`;

  // The header's play button is for the seed song alone: it pauses and
  // resumes the seed whenever the seed is the current song, from any context.
  function seedIsCurrent(state = resultsState) {
    const uri = playingTrackUri();
    return Boolean(uri && state?.seedUri) && uri === state.seedUri;
  }

  function seedPlayLabel(state = resultsState) {
    const name = state?.seedTrack?.name || "this song";
    if (!seedIsCurrent(state)) return `Play ${name}, then this list`;
    return Spicetify.Player?.isPlaying?.() ? `Pause ${name}` : `Resume ${name}`;
  }

  // Like Spotify's own lists, the page marks the song that is playing.
  function markPlayingRows(page = activePage?.view) {
    const uri = playingTrackUri();
    const playing = Boolean(Spicetify.Player?.isPlaying?.());
    page?.querySelectorAll?.(".sa-row").forEach((row) => {
      const current = Boolean(uri) && row.dataset?.uri === uri;
      row.classList?.toggle?.("sa-row-playing", current);
      const button = row.querySelector?.(".sa-play");
      if (button) {
        button.classList?.toggle?.("sa-paused-icon", current && playing);
        const label = button.getAttribute?.("aria-label") || "";
        button.setAttribute?.("aria-label", label.replace(/^(Play|Pause) /, current && playing ? "Pause " : "Play "));
      }
    });
    const main = page?.querySelector?.(".sa-seed-play");
    if (main) {
      const label = seedPlayLabel();
      main.classList?.toggle?.("sa-paused-icon", seedIsCurrent() && playing);
      main.setAttribute?.("aria-label", label);
      main.setAttribute?.("title", label);
    }
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
  function playShownResults(uri) {
    const uris = shownResultUris();
    return playResultsList(uris, uris.indexOf(uri));
  }

  async function playResultsList(uris, index) {
    const player = Spicetify.Platform?.PlayerAPI;
    if (typeof player?.play !== "function" || index < 0 || index >= uris.length) return false;
    lateMatches = null;
    await dropLateQueue(player);
    await player.play(
      {
        uri: RESULTS_CONTEXT_URI,
        pages: [{ items: uris.map((item) => ({ uri: item })) }],
      },
      { featureIdentifier: "soundalike" },
      { skipTo: { index, uri: uris[index] } }
    );
    followLateMatches(uris);
    return true;
  }

  // The header's play button: the seed song, then the shown list, like a song
  // radio that starts from its song. Plays the seed alone when Spotify refuses.
  async function playSeedSong(state) {
    const seedUri = state?.seedUri;
    if (!seedUri) return;
    const uris = [seedUri, ...shownResultUris().filter((uri) => uri !== seedUri)];
    try {
      if (await playResultsList(uris, 0)) return;
    } catch (error) {
      console.warn(
        "[soundalike] Spotify would not play the song with its results; playing it alone.",
        error
      );
    }
    try {
      await Spicetify.Player.playUri(seedUri);
    } catch (error) {
      console.error(`[soundalike] Could not play ${seedUri}`, error);
      Spicetify.showNotification(`Could not play ${state.seedTrack?.name || "this song"}.`, true);
    }
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

  function lateQueuedUris() {
    try {
      const value = JSON.parse(Spicetify.LocalStorage?.get(LATE_QUEUE_KEY) || "[]");
      return Array.isArray(value) ? value.filter((uri) => typeof uri === "string") : [];
    } catch {
      return [];
    }
  }

  function rememberLateQueued(uris) {
    try {
      if (uris.length) {
        Spicetify.LocalStorage?.set(LATE_QUEUE_KEY, JSON.stringify(uris.slice(-200)));
      } else {
        Spicetify.LocalStorage?.remove(LATE_QUEUE_KEY);
      }
    } catch {}
  }

  // Queued songs play before a list's next song, so the ones queued for an
  // earlier list leave the queue once another list starts or playback leaves
  // the list. A failure here never stops the new list from playing.
  async function dropLateQueue(player = Spicetify.Platform?.PlayerAPI) {
    const ours = new Set(lateQueuedUris());
    if (!ours.size) return;
    rememberLateQueued([]);
    if (typeof player?.removeFromQueue !== "function") return;
    try {
      const stale = (player.getQueue?.()?.queued || [])
        .filter((item) => ours.has(item?.uri))
        .map(({ uri, uid }) => (uid ? { uri, uid } : { uri }));
      if (stale.length) await player.removeFromQueue(stale);
    } catch (error) {
      console.warn("[soundalike] Songs queued for an earlier list could not be removed.", error);
    }
  }

  // Queued songs keep the list as their context, so another context means
  // the listener moved on.
  function leftResultsList() {
    const context = Spicetify.Platform?.PlayerAPI?.getState?.()?.context?.uri;
    if (context && context !== RESULTS_CONTEXT_URI) void dropLateQueue();
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
        rememberLateQueued([...new Set([...lateQueuedUris(), ...late.uris])]);
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

  // The bar shows the overall match when the server sends it (it follows the
  // list's order); the sound match alone moves to the tooltip. A consensus
  // row's match scores the model's ranking, not its sound, so it has none.
  function matchCell(React, result) {
    const consensus = result.basis === "consensus";
    const match = matchPercent(result.match);
    const overall = matchPercent(result.overall) ?? (consensus ? match : null);
    const shown = overall ?? match;
    if (shown === null) return React.createElement("div", { className: "sa-match" });
    const sound = match === null || consensus
      ? "" : `Sound match ${match}%: sounds more alike than ${match}% of typical matches`;
    const label = overall === null ? sound
      : `Match ${overall}%: how strongly Soundalike ranks it for this song${sound ? `. ${sound}` : ""}`;
    return React.createElement(
      "div",
      { className: "sa-match", title: label, role: "img", "aria-label": label },
      React.createElement(
        "span",
        { className: "sa-match-bar" },
        React.createElement("i", { style: { width: `${shown}%` } })
      ),
      React.createElement("span", { className: "sa-match-value" }, `${shown}%`)
    );
  }

  // React drops a whole root when anything in it throws while rendering, which
  // leaves a row empty. Spotify's right-click menu is the only part of a row
  // that isn't ours, so a row whose menu fails falls back to the row alone.
  function rowBoundary(React) {
    if (RowBoundary !== undefined) return RowBoundary;
    RowBoundary = null;
    if (typeof React.Component !== "function") return RowBoundary;
    RowBoundary = class SoundalikeRowBoundary extends React.Component {
      constructor(props) {
        super(props);
        this.state = { failed: false };
      }

      static getDerivedStateFromError() {
        return { failed: true };
      }

      componentDidCatch(error) {
        if (nativeMenusBroken) return;
        nativeMenusBroken = true;
        console.warn(
          "[soundalike] Spotify's right-click menu failed around a result; rows show without it.",
          error
        );
      }

      render() {
        return this.state.failed ? this.props.fallback : this.props.children;
      }
    };
    return RowBoundary;
  }

  function renderResultRow(
    row,
    result,
    index,
    track = null,
    rank = index + 1,
    { bpm = true, stems = null } = {}
  ) {
    const artist = spotifyArtists(track, result.artist);
    const image = track ? artworkUrl(track) : "";
    const matchNote = result.matchNote ? ` · ${result.matchNote}` : "";
    const listenerPick = result.basis === "listeners";
    row.dataset.q = `${result.title} ${result.artist}`;
    if (track?.uri) row.dataset.uri = track.uri;
    row.classList?.toggle?.(
      "sa-row-playing",
      Boolean(track?.uri) && track.uri === playingTrackUri()
    );

    const React = Spicetify.React;
    if (!(React?.createElement && Spicetify.ReactDOM?.createRoot)) {
      const rankNode = row.querySelector(".sa-rank");
      if (rankNode) rankNode.textContent = String(rank);
      const artistNode = row.querySelector(".sa-artist");
      if (artistNode) {
        artistNode.textContent =
          `${artist}${listenerPick ? ` · ${LISTENER_PICK_LABEL}` : ""}${matchNote}`;
      }
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
    // The row keeps its grid class; this one adds the Stems column.
    row.classList?.toggle?.("sa-row-stems", Boolean(stems));
    const dissectable = Boolean(stems) && deezerId(result.id) && result.id !== stems.id;
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
        "aria-label": (track?.uri
          ? `Play ${result.title} by ${artist}`
          : `Search Spotify for ${result.title} by ${result.artist}`) +
          (listenerPick ? `, ${LISTENER_PICK_ARIA}` : ""),
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
              if (track?.uri === playingTrackUri()) Spicetify.Player.togglePlay();
              else activate();
            },
          },
          track?.uri ? React.createElement("svg", { viewBox: "0 0 24 24", "aria-hidden": true },
            React.createElement("path", { className: "sa-icon-play", d: PLAY_PATH }),
            React.createElement("path", { className: "sa-icon-pause", d: PAUSE_PATH })) : "\u203A"
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
        listenerPick
          ? React.createElement(
            "div",
            { className: "sa-artist" },
            artist,
            React.createElement(
              "span",
              { className: "sa-why" },
              " · ",
              React.createElement(
                "span",
                { className: "sa-basis", title: LISTENER_PICK_TITLE },
                LISTENER_PICK_LABEL
              ),
              matchNote
            )
          )
          : matchNote
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
        : null,
      stems ? matchCell(React, result) : null,
      stems
        ? React.createElement(
          "div",
          { className: "sa-dx-cell" },
          dissectable
            ? React.createElement(
              "button",
              {
                className: "sa-dx-open",
                type: "button",
                title: `Compare ${result.title} with ${stems.title || "this song"}, instrument by instrument`,
                "aria-label": `Stems: compare ${result.title} with ${stems.title || "this song"}`,
                onClick: (event) => {
                  event.stopPropagation();
                  openDissection(stems, result, track);
                },
                onDoubleClick: (event) => event.stopPropagation(),
                // Enter on the button opens Stems, not the row's playback.
                onKeyDown: (event) => event.stopPropagation(),
              },
              "Stems"
            )
            : null
        )
        : null
    );
    const menu = nativeMenusBroken ? null : nativeTrackMenu(track);
    const native = menu
      ? withNativeContext(
        React.createElement(Spicetify.ReactComponent.RightClickMenu, { menu }, child))
      : null;
    const Boundary = native ? rowBoundary(React) : null;
    row.__soundalikeRoot.render(
      Boundary ? React.createElement(Boundary, { fallback: child }, native) : native || child
    );
    setTimeout(() => markPlayingRows(), 0);
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
        resultsState.sounds,
        resultsState.audioOnly === true
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

  // A row that can't be drawn the rich way still shows as its plain markup.
  function revealBestRow(page, results, tracks, index, rank, bpm = false, stems = null) {
    const row = page.querySelector(`.sa-row[data-index="${index}"]`);
    if (!row || page !== activePage?.view) return;
    try {
      renderResultRow(row, results[index], index, tracks[index] || null, rank, { bpm, stems });
    } catch (error) {
      console.warn(`[soundalike] Could not draw the row for ${results[index].title}`, error);
    }
    row.hidden = false;
  }

  // Names the songs left out because Spotify doesn't have them. Each one still
  // opens a Spotify search, in case the match was missed.
  function showLeftOutSongs(page, results, leftOut) {
    const note = page.querySelector(".sa-list-note");
    if (!note || page !== activePage?.view) return;
    if (!leftOut.length) {
      note.hidden = true;
      return;
    }
    const songs = leftOut.flatMap((index, position) => {
      const result = results[index];
      const button = document.createElement("button");
      button.type = "button";
      button.className = "sa-left-out";
      button.textContent = `${result.title} · ${result.artist}`;
      button.title = `Search Spotify for ${result.title} by ${result.artist}`;
      button.onclick = () => activateResult(result, null);
      return position ? [document.createTextNode(", "), button] : [button];
    });
    note.replaceChildren("Not found on Spotify, so left out: ", ...songs);
    note.hidden = false;
  }

  // Best matches, Listeners also play and Melody match arrive already ranked,
  // so Spotify adds artwork, album and playback, and decides what is shown:
  // rows appear in rank order once found on Spotify, a song Spotify doesn't
  // have is left out, and the next row takes its place, up to limit rows. A
  // row whose lookup failed, or is too slow to wait for, still shows with a
  // Spotify search instead of playback (a slow one gains its song if it
  // arrives), and so do all rows when this client can't search. Once one
  // lookup is that slow, Spotify's search is slow or stuck, so no later row
  // waits for its lookup. The lyrics-language gate and feedback belong to the
  // companion's Sounds like list.
  async function hydrateBestRows(
    results,
    page,
    tracks,
    timing,
    label = "Best matches",
    { bpm = false, stems = null, limit = DISPLAY_RESULT_COUNT } = {}
  ) {
    const startedAt = nowMs();
    const searchable = Boolean(Spicetify.GraphQL.Definitions.searchModalResults);
    if (!searchable) {
      console.warn("[soundalike] Spotify artwork lookup is unavailable in this client.");
    }
    const unmatched = new Set();
    const usable = new Set();
    const ranks = new Map();
    const leftOut = [];
    let placed = 0;
    let failures = 0;
    let slow = false;
    function place() {
      while (
        placed < results.length &&
        (!searchable || slow || placed in tracks || unmatched.has(placed))
      ) {
        const index = placed++;
        if (slow && !(index in tracks)) unmatched.add(index);
        if (searchable && !tracks[index] && !unmatched.has(index)) {
          leftOut.push(index);
        } else if (ranks.size < limit) {
          ranks.set(index, ranks.size + 1);
          revealBestRow(page, results, tracks, index, ranks.get(index), bpm, stems);
        }
      }
    }
    function lookUp(index) {
      return findSpotifyTrack(results[index], false).then(
        (track) => {
          tracks[index] = track;
          // Shown unmatched while its lookup was slow; now it can play.
          if (track && ranks.has(index)) {
            revealBestRow(page, results, tracks, index, ranks.get(index), bpm, stems);
          }
        },
        (error) => {
          failures++;
          unmatched.add(index);
          console.warn(
            `[soundalike] Spotify metadata lookup failed for ${results[index].title}`,
            error
          );
        }
      );
    }
    let cursor = 0;
    // A worker moves on once a lookup outlasts the patience, so a search that
    // never answers can't hold the list; rows already shown are still looked up.
    async function worker() {
      while (cursor < results.length && (usable.size < limit || ranks.has(cursor))) {
        const index = cursor++;
        if (!(index in tracks)) {
          let patience;
          const waited = new Promise((resolve) => {
            patience = setTimeout(() => {
              slow = true;
              unmatched.add(index);
              resolve();
            }, SPOTIFY_LOOKUP_PATIENCE_MS);
          });
          await Promise.race([lookUp(index).then(() => clearTimeout(patience)), waited]);
        }
        if (tracks[index] || unmatched.has(index)) usable.add(index);
        place();
      }
    }
    if (searchable) {
      const workerCount = Math.min(SPOTIFY_ENRICH_WORKERS, results.length);
      await Promise.all(Array.from({ length: workerCount }, worker));
    }
    place();
    if (!ranks.size && leftOut.length) {
      // Spotify finding none of them points at its search, not at the songs.
      console.warn(`[soundalike] Spotify found none of the ${label} songs; showing them unmatched.`);
      leftOut.splice(0).slice(0, limit).forEach((index) => {
        ranks.set(index, ranks.size + 1);
        revealBestRow(page, results, tracks, index, ranks.get(index), bpm, stems);
      });
    }
    showLeftOutSongs(page, results, leftOut);
    // Show more stays while the list has songs that are neither shown nor
    // known to be missing from Spotify.
    const more = page.querySelector(".sa-more");
    if (more && page === activePage?.view) {
      more.hidden = ranks.size + leftOut.length >= results.length;
      more.disabled = false;
      more.textContent = "Show more";
    }
    console.log(
      `[soundalike] ${label} metadata settled in ${elapsedMs(startedAt)} ms ` +
      `(shown ${ranks.size}, not on Spotify ${leftOut.length}, failed lookups ${failures}, ` +
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

  // Sounds like is the local companion's list, or the stand-in for Best matches
  // the hosted matcher couldn't make; hosted searches show it once it settles.
  function resultViews(state) {
    const best = state.best;
    // Melody match is the hosted "best" list, which servers without the main
    // list already show as Best matches.
    const melody = best.status !== "ready" ||
      Boolean(best.data.main?.length && best.data.results?.length);
    return RESULT_VIEWS.filter(([view]) =>
      view === "melody"
        ? melody
        : view !== "sounds" || state.companion || state.sounds.status !== "pending"
    );
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

  function listHeadMarkup({ bpm = false, stems = false } = {}) {
    return `<div class="sa-list-head${stems ? " sa-with-stems" : ""}"><div>#</div><div></div>` +
      `<div>Title</div><div class="sa-album-head">Album</div>` +
      `${bpm ? '<div class="sa-bpm">BPM</div>' : ""}` +
      `${stems ? '<div class="sa-match-head">Match</div><div class="sa-dx-head">Compare</div>' : ""}</div>`;
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
      .map((x, i) => {
        const pick = notes && x.basis === "listeners"
          ? ` · <span class="sa-basis" title="${esc(LISTENER_PICK_TITLE)}">${esc(LISTENER_PICK_LABEL)}</span>`
          : "";
        const why = pick || (notes && x.matchNote)
          ? `<span class="sa-why">${pick}${notes && x.matchNote ? ` · ${esc(x.matchNote)}` : ""}</span>`
          : "";
        return `
      <div class="sa-row" data-index="${i}" data-q="${esc(x.title + " " + x.artist)}"${hidden ? " hidden" : ""}>
        <div class="sa-rank">${i + 1}</div>
        <div class="sa-cover" aria-hidden="true"><span>♫</span></div>
        <div class="sa-meta">
          <div class="sa-title">${esc(x.title)}</div>
          <div class="sa-artist">${esc(x.artist)}${why}</div>
        </div>
        <div class="sa-open">›</div>
      </div>`;
      })
      .join("");
  }

  // Best matches and Listeners also play come from one response, so they share
  // its notes, tags and language switch.
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

  // Audio only drops the listener ranker from every hosted list, so each of
  // them shows the switch: next to the language one once a list is ready.
  function audioOnlyToggleMarkup(state) {
    return `<button class="sa-audio-only" type="button" aria-pressed="${Boolean(state.audioOnly)}" ` +
      `title="${esc(AUDIO_ONLY_TITLE)}">Audio only</button>`;
  }

  function audioOnlyRowMarkup(state) {
    return `<div class="sa-audio-only-row">${audioOnlyToggleMarkup(state)}</div>`;
  }

  // The server's query.hidden counts other-language songs in its fused order of
  // every channel, not in the main list or Listeners also play, so both tabs
  // only say whether other languages are left out. A main-only guess
  // (mainLanguageGuess) gates the main list alone.
  function languageFilterMarkup(query, state, { main = false } = {}) {
    const allLanguages = query.languageMode === "all";
    const mainGuess = main && query.language === "unknown" ? query.mainLanguageGuess || "" : "";
    const language = mainGuess || query.language;
    const unknown = language === "unknown";
    const control = unknown ? "" :
      `<button class="sa-language-toggle" type="button" aria-pressed="${allLanguages}">${allLanguages ? "Use same-language matches" : "Show all languages"}</button>`;
    const text = unknown
      ? "Language uncertain; showing all languages."
      : `${languageName(language)}${query.languageGuessed || mainGuess ? " (best guess)" : ""} · ${allLanguages ? "all languages" : "other languages left out"}`;
    return `<div class="sa-language-filter" role="status"><span>${esc(text)}</span>` +
      `<span class="sa-filter-actions">${control}${audioOnlyToggleMarkup(state)}</span></div>`;
  }

  function bestPanelMarkup(state, seedArtists) {
    const best = state.best;
    const ready = best.status === "ready";
    const mixed = ready && !best.data.main?.length;
    // The five-signal model placed rows; it needs no listener data.
    const consensus = ready && bestRows(best.data).some((row) => row.basis === "consensus");
    const soundOnly = state.audioOnly ||
      (ready && !consensus && best.data.query?.scene === "unknown_artist");
    let blurb = soundOnly
      ? AUDIO_ONLY_BEST_MATCHES_BLURB
      : mixed ? MIXED_BEST_MATCHES_BLURB
        : ready && !consensus ? NUDGED_BEST_MATCHES_BLURB : BEST_MATCHES_BLURB;
    if (best.status === "offer") {
      const note = !state.viewChosen && state.sounds.status === "error"
        ? `<p class="sa-note">${esc(state.sounds.error)}</p>`
        : "";
      return `<p class="sa-blurb">${esc(blurb)}</p>${note}${audioOnlyRowMarkup(state)}
      <div class="sa-offer">
        <p>${esc(BEST_MATCHES_OFFER)}</p>
        <button class="sa-best-offer" type="button">Get best matches</button>
      </div>`;
    }
    if (best.status !== "ready") {
      return `<p class="sa-blurb">${esc(blurb)}</p>${audioOnlyRowMarkup(state)}` + listStateMarkup(
        best,
        "Listening for best matches… songs outside the library take a few seconds."
      );
    }
    const { query } = best.data;
    const results = bestRows(best.data);
    blurb += matchedToNote(state, query, seedArtists);
    const tags = tagsMarkup([
      mixed ? "Melody + samples" : consensus ? "Five signals" : "Melody, samples + sound",
      ...sourceTags(best),
    ]);
    return `<p class="sa-blurb">${esc(blurb)}</p>
      <div class="sa-tags">${tags}</div>
      ${languageFilterMarkup(query, state, { main: true })}
      ${listHeadMarkup({ stems: Boolean(dissectSeed(state, results)) })}
      <div class="sa-list">${resultRowsMarkup(results, { hidden: true, notes: true })}
      </div>${LIST_NOTE_MARKUP}`;
  }

  // Other artists' songs this song's listeners play, from the Best matches
  // response. Audio only and artists without listener data leave it empty.
  function listenersPanelMarkup(state, seedArtists) {
    const best = state.best;
    if (best.status === "offer") {
      return `<p class="sa-blurb">${esc(LISTENERS_BLURB)}</p>${audioOnlyRowMarkup(state)}
      <div class="sa-offer">
        <p>${esc(BEST_MATCHES_OFFER)}</p>
        <button class="sa-best-offer" type="button">Get best matches</button>
      </div>`;
    }
    if (best.status !== "ready") {
      return `<p class="sa-blurb">${esc(LISTENERS_BLURB)}</p>${audioOnlyRowMarkup(state)}` +
        listStateMarkup(best, "Finding songs this song's listeners also play…");
    }
    const { query, listeners } = best.data;
    if (!listeners?.length) {
      // An empty list has no ranking to describe; the note says why it's empty.
      const note = state.audioOnly ? AUDIO_ONLY_LISTENERS_NOTE : NO_LISTENERS_NOTE;
      return audioOnlyRowMarkup(state) + listStateMarkup({ status: "empty" }, note);
    }
    const blurb = LISTENERS_BLURB + matchedToNote(state, query, seedArtists);
    const tags = tagsMarkup(["What listeners play", ...sourceTags(best)]);
    return `<p class="sa-blurb">${esc(blurb)}</p>
      <div class="sa-tags">${tags}</div>
      ${languageFilterMarkup(query, state)}
      ${listHeadMarkup({ stems: Boolean(dissectSeed(state, listeners)) })}
      <div class="sa-list">${resultRowsMarkup(listeners, { hidden: true, notes: false })}
      </div>${LIST_NOTE_MARKUP}`;
  }

  // The hosted "best" list, as the Best matches tab showed it before the main
  // list: the melody and sample matches fused into one ranking, then, with
  // listener data, the listener picks.
  function melodyPanelMarkup(state, seedArtists) {
    const best = state.best;
    const ready = best.status === "ready";
    const soundOnly = state.audioOnly ||
      (ready && best.data.query?.scene === "unknown_artist");
    const blurb = soundOnly ? AUDIO_ONLY_MELODY_BLURB : MELODY_BLURB;
    if (best.status === "offer") {
      return `<p class="sa-blurb">${esc(blurb)}</p>${audioOnlyRowMarkup(state)}
      <div class="sa-offer">
        <p>${esc(BEST_MATCHES_OFFER)}</p>
        <button class="sa-best-offer" type="button">Get best matches</button>
      </div>`;
    }
    if (!ready) {
      return `<p class="sa-blurb">${esc(blurb)}</p>${audioOnlyRowMarkup(state)}` +
        listStateMarkup(best, "Listening for melody and sample matches…");
    }
    const { query, results } = best.data;
    const tags = tagsMarkup(["Melody + samples", ...sourceTags(best)]);
    return `<p class="sa-blurb">${esc(blurb + matchedToNote(state, query, seedArtists))}</p>
      <div class="sa-tags">${tags}</div>
      ${languageFilterMarkup(query, state)}
      ${listHeadMarkup({ stems: Boolean(dissectSeed(state, results)) })}
      <div class="sa-list">${resultRowsMarkup(results, { hidden: true, notes: true })}
      </div>${LIST_NOTE_MARKUP}`;
  }

  function soundsPanelMarkup(state, seedArtists) {
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
    // Spotify's related artists come from what listeners also play, so Audio
    // only keeps the model's order here too.
    const context = state.audioOnly ? "" : " and artist context";
    const languageStatus = sounds.seedLanguage
      ? `${languageName(sounds.seedLanguage)} lyrics · checking quality head`
      : `Checking Spotify metadata${context}`;
    const modelLabel = data.method === "dual_sonic64_guardrail"
      ? "Dual-Sonic64"
      : "Production model";
    const tags = tagsMarkup([
      modelLabel,
      state.audioOnly ? "V5 strict" : "V5 strict + artist context",
      v.tempo,
      v.dynamics,
      v.low_end,
      v.tone,
    ]) + `<span class="sa-tag sa-language-status">${esc(languageStatus)}</span>`;
    return `<p class="sa-blurb">${esc(blurb)}</p>${note}
      <div class="sa-tags">${tags}</div>${audioOnlyRowMarkup(state)}
      <div class="sa-list-head"><div>#</div><div></div><div>Title</div><div class="sa-album-head">Album</div><div class="sa-bpm">BPM</div></div>
      <div class="sa-list">
        <div class="sa-language-loading">Checking Spotify language${context} before showing results…</div>
        ${resultRowsMarkup(data.results, { hidden: true, notes: false })}
      </div>${feedbackMarkup()}`;
  }

  function shownView(state) {
    return resultViews(state).some(([key]) => key === state.view) ? state.view : "best";
  }

  // The list a view shows. Best matches and Listeners also play both come from
  // the Best matches response; Sounds like (the companion's, or the older
  // hosted service when Best matches fail) has its own.
  function viewList(state, view) {
    const best = state.best;
    if (view === "sounds") {
      const sounds = state.sounds;
      return { list: sounds, results: sounds.data?.results, tracks: sounds.tracks, separate: true };
    }
    if (view === "listeners") {
      return {
        list: best,
        results: best.data?.listeners,
        tracks: (best.listenerTracks ||= []),
        label: "Listeners also play",
      };
    }
    if (view === "melody") {
      return {
        list: best,
        results: best.data?.results,
        tracks: (best.melodyTracks ||= []),
        label: "Melody match",
      };
    }
    return { list: best, results: best.data && bestRows(best.data), tracks: best.tracks, label: "Best matches" };
  }

  function buildResultsPage(state) {
    const wrap = document.createElement("div");
    const view = shownView(state);
    const seedTrack = state.seedTrack;
    const seedImage = artworkUrl(seedTrack);
    const seedArtists = seedArtistNames(seedTrack);
    const sounds = state.sounds;
    let kicker = view === "listeners"
      ? "LISTENERS ALSO PLAY · HOSTED LIBRARY"
      : view === "melody"
        ? "MELODY MATCH · HOSTED LIBRARY"
        : "BEST MATCHES · HOSTED LIBRARY";
    if (view === "sounds") {
      kicker = sounds.status === "ready"
        ? `SOUNDS LIKE · ${sounds.data.__soundalikeSource === "local"
          ? "LOCAL ENGINE"
          : "HOSTED LIBRARY"}`
        : "SOUNDS LIKE";
    }
    wrap.className = "sa-results";

    wrap.innerHTML = `${pageStyleMarkup()}
      <div class="sa-seed">
        ${seedImage
          ? `<img src="${esc(seedImage)}" alt="">`
          : `<div class="sa-seed-fallback" aria-hidden="true">♫</div>`}
        <div>
          <div class="sa-kicker">${esc(kicker)}</div>
          <div class="sa-seed-title">${esc(seedTrack?.name)}</div>
          <div class="sa-seed-artist">${esc(seedArtists.join(", "))}</div>
        </div>
        ${state.seedUri
          ? `<button class="sa-seed-play" type="button" title="${esc(seedPlayLabel(state))}" aria-label="${esc(seedPlayLabel(state))}"><svg viewBox="0 0 24 24" aria-hidden="true"><path class="sa-icon-play" d="${PLAY_PATH}"></path><path class="sa-icon-pause" d="${PAUSE_PATH}"></path></svg></button>`
          : ""}
        ${partOpenMarkup(state, view)}
        ${recentToggleMarkup(state)}
      </div>
      ${partPanelMarkup(state, view)}
      ${partChipMarkup(state, view)}
      ${recentSearchesMarkup(state.seedUri, Boolean(state.recentOpen))}
      ${resultTabsMarkup(state)}
      <div class="sa-panel sa-view-${view}" id="sa-panel" role="tabpanel" aria-labelledby="sa-tab-${view}">
        ${view === "best"
          ? bestPanelMarkup(state, seedArtists)
          : view === "listeners"
            ? listenersPanelMarkup(state, seedArtists)
            : view === "melody"
              ? melodyPanelMarkup(state, seedArtists)
              : soundsPanelMarkup(state, seedArtists)}
      </div>`;

    const shown = viewList(state, view);
    const results = (shown.list.status === "ready" && shown.results) || [];
    const tracks = shown.tracks || [];
    wrap.querySelectorAll(".sa-row").forEach((row) => {
      const index = Number(row.dataset?.index);
      const result = results[index];
      // Hosted rows stay hidden until hydrateBestRows finds them on Spotify.
      if (!result || !shown.separate) return;
      row.hidden = Boolean(shown.list.seedLanguage);
      try {
        renderResultRow(row, result, index, tracks[index] || null);
      } catch (error) {
        console.warn(`[soundalike] Could not draw the row for ${result.title}`, error);
      }
    });
    return wrap;
  }

  // One style sheet for the results page and the page before any search.
  // Colours come from the theme's variables, with Spotify's own as fallbacks.
  function pageStyleMarkup() {
    return `
      <style>
        .sa-results{box-sizing:border-box;min-height:100%;padding:32px clamp(24px,5vw,64px) 56px;font:14px var(--encore-body-font-stack,system-ui,sans-serif);color:var(--spice-text,#fff)}
        .sa-results[hidden]{display:none}
        .sa-seed{display:flex;align-items:center;gap:18px;margin:0 0 18px}
        .sa-seed img,.sa-seed-fallback{width:96px;height:96px;border-radius:6px;object-fit:cover;background:#282828;box-shadow:0 8px 24px rgba(0,0,0,.35)}
        .sa-seed-fallback{display:grid;place-items:center;color:#b3b3b3;font-size:22px}
        .sa-kicker{color:var(--spice-subtext,#b3b3b3);font-size:12px;margin-bottom:2px}
        .sa-seed-title{font-size:clamp(24px,4vw,40px);font-weight:800;line-height:1.1}
        .sa-seed-artist{color:var(--spice-subtext,#b3b3b3);margin-top:2px}
        .sa-seed-play{flex:none;width:56px;height:56px;margin-left:8px;padding:0;border:0;border-radius:50%;background:var(--spice-button,#1ed760);color:#000;line-height:1;cursor:pointer;display:grid;place-items:center}
        .sa-seed-play:hover{transform:scale(1.06)}
        .sa-seed-play svg{width:24px;height:24px;fill:currentColor}
        .sa-seed-play:hover{background:var(--spice-button-active,#3be477)}
        .sa-icon-pause,.sa-paused-icon .sa-icon-play{display:none}
        .sa-paused-icon .sa-icon-pause{display:block}
        .sa-play svg{width:16px;height:16px;fill:currentColor}
        .sa-seed-play:focus-visible{outline:2px solid var(--spice-text,#fff);outline-offset:2px}
        .sa-tags{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px}
        .sa-tag{background:rgba(var(--spice-rgb-text,255,255,255),.08);border-radius:999px;padding:4px 9px;color:var(--spice-subtext,#b3b3b3);font-size:12px}
        .sa-list{display:flex;flex-direction:column;padding-right:4px}
        .sa-list-note{margin:12px 6px 0;color:var(--spice-subtext,#b3b3b3);font-size:13px}
        .sa-list-note[hidden]{display:none}
        .sa-left-out{padding:0;border:0;background:none;color:inherit;font:inherit;text-decoration:underline;cursor:pointer}
        .sa-left-out:hover,.sa-left-out:focus-visible{color:var(--spice-text,#fff)}
        .sa-left-out:focus-visible{outline:2px solid var(--spice-button,#1ed760);outline-offset:2px}
        .sa-language-loading{padding:18px 6px;color:var(--spice-subtext,#b3b3b3)}
        .sa-list-head,.sa-row-content{display:grid;grid-template-columns:32px 44px minmax(180px,1.6fr) minmax(120px,1fr) 64px;align-items:center;column-gap:12px}
        .sa-list-head{height:28px;color:var(--spice-subtext,#b3b3b3);font-size:12px;border-bottom:1px solid rgba(var(--spice-rgb-text,255,255,255),.1);margin-bottom:4px}
        .sa-row{border-radius:6px}
        .sa-row-content{min-height:56px;padding:4px 6px;border-radius:6px;cursor:pointer;outline:none}
        .sa-row-content:hover,.sa-row-content:focus-visible{background:rgba(var(--spice-rgb-text,255,255,255),.1)}
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
        .sa-play:hover,.sa-play:focus-visible{background:transparent;color:var(--spice-text,#fff)}
        .sa-play:focus-visible{outline:2px solid currentColor;outline-offset:2px}
        .sa-row-content:hover .sa-title,.sa-row-content:focus-visible .sa-title{color:var(--spice-text,#fff)}
        .sa-row.sa-row-playing .sa-title,.sa-row.sa-row-playing .sa-rank{color:var(--spice-button,#1ed760)}
        .sa-menu-loading{padding:8px 12px;color:var(--spice-subtext,#b3b3b3);white-space:nowrap}
        .sa-feedback{max-width:680px;margin:24px 0 0;padding:16px;border:1px solid rgba(var(--spice-rgb-text,255,255,255),.12);border-radius:8px;background:rgba(var(--spice-rgb-text,255,255,255),.04)}
        .sa-feedback[hidden],.sa-feedback-details[hidden],.sa-feedback-question[hidden],.sa-feedback-actions[hidden],.sa-feedback-preference[hidden]{display:none}
        .sa-feedback fieldset{border:0;margin:0;padding:0}
        .sa-feedback legend{font-weight:700;margin-bottom:10px}
        .sa-feedback-options,.sa-feedback-reasons,.sa-feedback-actions{display:flex;flex-wrap:wrap;gap:8px}
        .sa-feedback-choice,.sa-feedback-reason{display:inline-flex;align-items:center;gap:6px;min-height:36px;padding:7px 11px;border:1px solid rgba(var(--spice-rgb-text,255,255,255),.16);border-radius:999px;cursor:pointer}
        .sa-feedback-choice:focus-within,.sa-feedback-reason:focus-within{outline:2px solid var(--spice-button,#1ed760);outline-offset:2px}
        .sa-feedback-details{margin-top:14px}
        .sa-feedback-reasons{margin-bottom:12px}
        .sa-feedback-note-label{display:block;font-weight:600;margin-bottom:6px}
        .sa-feedback-note{box-sizing:border-box;width:100%;min-height:72px;resize:vertical;padding:9px 10px;border:1px solid rgba(var(--spice-rgb-text,255,255,255),.18);border-radius:6px;background:var(--spice-card,#242424);color:var(--spice-text,#fff);font:inherit}
        .sa-feedback-help{display:flex;justify-content:space-between;gap:12px;color:var(--spice-subtext,#b3b3b3);font-size:12px;margin-top:4px}
        .sa-feedback-actions{align-items:center;margin-top:14px}
        .sa-feedback-actions button{min-height:36px;border-radius:999px;padding:7px 14px;font:inherit;font-weight:700;cursor:pointer}
        .sa-feedback-send{border:0;background:var(--spice-button,#1ed760);color:#000}
        .sa-feedback-send:disabled{cursor:not-allowed;opacity:.55}
        .sa-feedback-dismiss{border:1px solid rgba(var(--spice-rgb-text,255,255,255),.2);background:transparent;color:var(--spice-text,#fff)}
        .sa-feedback-preference p{font-weight:700;margin:0 0 10px}
        .sa-feedback-preference-actions{display:flex;gap:8px}
        .sa-feedback-preference button{min-height:36px;border-radius:999px;padding:7px 14px;font:inherit;font-weight:700;cursor:pointer}
        .sa-feedback-keep-showing{border:0;background:var(--spice-button,#1ed760);color:#000}
        .sa-feedback-stop-showing{border:1px solid rgba(var(--spice-rgb-text,255,255,255),.2);background:transparent;color:var(--spice-text,#fff)}
        .sa-feedback-status{min-height:20px;color:var(--spice-subtext,#b3b3b3)}
        .sa-feedback-success{color:var(--spice-button,#1ed760)}
        .sa-feedback-error{color:#f29d9d}
        .sa-tabs{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 14px}
        .sa-tab{min-height:32px;padding:6px 14px;border:0;border-radius:999px;background:rgba(var(--spice-rgb-text,255,255,255),.08);color:var(--spice-text,#fff);font:inherit;font-weight:600;cursor:pointer}
        .sa-tab:hover{background:rgba(var(--spice-rgb-text,255,255,255),.14)}
        .sa-tab[aria-selected="true"]{background:var(--spice-text,#fff);color:rgb(var(--spice-rgb-main,18,18,18))}
        .sa-tab:focus-visible{outline:2px solid var(--spice-button,#1ed760);outline-offset:2px}
        .sa-language-filter{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:0 0 14px;color:var(--spice-subtext,#b3b3b3);font-size:13px}
        .sa-language-toggle,.sa-audio-only{min-height:32px;border:1px solid rgba(var(--spice-rgb-text,255,255,255),.2);border-radius:999px;padding:5px 12px;background:transparent;color:var(--spice-text,#fff);font:inherit;font-weight:600;cursor:pointer}
        .sa-language-toggle:hover,.sa-audio-only:hover{border-color:var(--spice-button,#1ed760)}
        .sa-language-toggle:focus-visible,.sa-audio-only:focus-visible{outline:2px solid var(--spice-button,#1ed760);outline-offset:2px}
        .sa-audio-only[aria-pressed="true"]{border-color:var(--spice-button,#1ed760);background:var(--spice-button,#1ed760);color:#000}
        .sa-filter-actions{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:8px}
        .sa-audio-only-row{display:flex;justify-content:flex-end;margin:0 0 14px}
        .sa-basis{text-decoration:underline dotted;text-underline-offset:2px;cursor:help}
        .sa-blurb{max-width:680px;margin:0 0 10px;color:var(--spice-subtext,#b3b3b3)}
        .sa-note{max-width:680px;margin:0 0 12px;color:var(--spice-text,#fff)}
        .sa-why{color:var(--spice-subtext,#b3b3b3)}
        .sa-state{padding:18px 6px;color:var(--spice-subtext,#b3b3b3)}
        .sa-state-error{color:#f29d9d}
        .sa-offer{max-width:680px;padding:16px;border:1px solid rgba(var(--spice-rgb-text,255,255,255),.12);border-radius:8px;background:rgba(var(--spice-rgb-text,255,255,255),.04)}
        .sa-offer p{margin:0 0 12px;color:var(--spice-subtext,#b3b3b3)}
        .sa-best-offer{min-height:36px;border:0;border-radius:999px;padding:7px 14px;background:var(--spice-button,#1ed760);color:#000;font:inherit;font-weight:700;cursor:pointer}
        .sa-best-offer:focus-visible{outline:2px solid var(--spice-text,#fff);outline-offset:2px}
        .sa-view-best .sa-list-head,.sa-view-best .sa-row-content,.sa-view-listeners .sa-list-head,.sa-view-melody .sa-list-head,.sa-view-listeners .sa-row-content,.sa-view-melody .sa-row-content{grid-template-columns:32px 44px minmax(180px,1.6fr) minmax(120px,1fr)}
        .sa-list-head.sa-with-stems,.sa-row-stems .sa-row-content{grid-template-columns:32px 44px minmax(180px,1.6fr) minmax(120px,1fr) 64px 76px 72px}
        .sa-view-best .sa-list-head.sa-with-stems,.sa-view-best .sa-row-stems .sa-row-content,.sa-view-listeners .sa-list-head.sa-with-stems,.sa-view-melody .sa-list-head.sa-with-stems,.sa-view-listeners .sa-row-stems .sa-row-content,.sa-view-melody .sa-row-stems .sa-row-content{grid-template-columns:32px 44px minmax(180px,1.6fr) minmax(120px,1fr) 76px 72px}
        .sa-match{display:flex;align-items:center;gap:6px;min-width:0}
        .sa-match-bar{flex:1 1 auto;min-width:20px;height:6px;border-radius:3px;background:rgba(var(--spice-rgb-text,255,255,255),.14);overflow:hidden}
        .sa-match-bar i{display:block;height:100%;border-radius:3px;background:var(--spice-button,#1ed760)}
        .sa-match-value{flex:0 0 auto;width:32px;text-align:right;color:var(--spice-subtext,#b3b3b3);font-size:12px;font-variant-numeric:tabular-nums}
        .sa-dx-head{text-align:center}
        .sa-dx-cell{display:flex;justify-content:center}
        .sa-dx-open{min-height:28px;padding:4px 11px;border:1px solid rgba(var(--spice-rgb-text,255,255,255),.22);border-radius:999px;background:transparent;color:var(--spice-subtext,#b3b3b3);font:inherit;font-size:12px;font-weight:700;cursor:pointer}
        .sa-dx-open:hover,.sa-dx-open:focus-visible{border-color:var(--spice-button,#1ed760);color:var(--spice-text,#fff);outline:none}
        .sa-dx-top{display:flex;align-items:center;gap:12px;margin:0 0 16px}
        .sa-dx-back,.sa-dx-retry{min-height:32px;border:1px solid rgba(var(--spice-rgb-text,255,255,255),.2);border-radius:999px;padding:5px 14px;background:transparent;color:var(--spice-text,#fff);font:inherit;font-weight:600;cursor:pointer}
        .sa-dx-back:hover,.sa-dx-retry:hover{border-color:var(--spice-button,#1ed760)}
        .sa-dx-back:focus-visible,.sa-dx-retry:focus-visible,.sa-dx-play:focus-visible{outline:2px solid var(--spice-button,#1ed760);outline-offset:2px}
        .sa-dx-title{font-size:clamp(20px,3vw,30px);font-weight:800;line-height:1.15;margin:2px 0 8px}
        .sa-dx-title span{color:var(--spice-subtext,#b3b3b3);font-weight:600}
        .sa-dx-compare{max-width:760px;margin:6px 0 22px;padding:14px 16px;border:1px solid rgba(var(--spice-rgb-text,255,255,255),.12);border-radius:8px;background:rgba(var(--spice-rgb-text,255,255,255),.04)}
        .sa-dx-compare h3,.sa-dx-song h3{font-size:13px;font-weight:700;margin:0 0 10px;color:var(--spice-subtext,#b3b3b3);text-transform:uppercase;letter-spacing:.06em}
        .sa-dx-matched{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:0 0 12px}
        .sa-dx-chip{background:var(--spice-button,#1ed760);color:#000;border-radius:999px;padding:4px 10px;font-size:12px;font-weight:700}
        .sa-dx-facets{display:grid;grid-template-columns:minmax(0,max-content) minmax(48px,1fr) minmax(0,max-content);gap:6px 12px;align-items:center}
        .sa-dx-bar{height:8px;border-radius:999px;background:rgba(var(--spice-rgb-text,255,255,255),.1);overflow:hidden}
        .sa-dx-bar span{display:block;height:100%;border-radius:inherit;background:var(--spice-subtext,#b3b3b3)}
        .sa-dx-strong .sa-dx-bar span{background:var(--spice-button,#1ed760)}
        .sa-dx-strong .sa-dx-facet-label{font-weight:700}
        .sa-dx-facet-value{color:var(--spice-subtext,#b3b3b3);font-variant-numeric:tabular-nums}
        .sa-dx-hint{margin:10px 0 0;color:var(--spice-subtext,#b3b3b3);font-size:12px}
        .sa-dx-songs{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(340px,100%),1fr));gap:22px}
        .sa-dx-song{min-width:0}
        .sa-dx-song-head{display:flex;align-items:center;gap:10px;margin:0 0 8px}
        .sa-dx-song-title{font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
        .sa-dx-song-artist{color:var(--spice-subtext,#b3b3b3);font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
        .sa-dx-song-head>div{min-width:0}
        .sa-dx-role{flex:none;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--spice-subtext,#b3b3b3)}
        .sa-dx-play{flex:none;width:32px;height:32px;border:0;border-radius:50%;background:var(--spice-button,#1ed760);color:#000;font-size:13px;cursor:pointer;display:grid;place-items:center}
        .sa-dx-lane{display:grid;grid-template-columns:96px minmax(0,1fr);align-items:center;column-gap:10px;margin:0 0 4px}
        .sa-dx-lane-label{font-size:12px;font-weight:600;line-height:1.2}
        .sa-dx-lane-label span{display:block;font-weight:400;color:var(--spice-subtext,#b3b3b3)}
        .sa-dx-svg{display:block;width:100%;background:rgba(var(--spice-rgb-text,255,255,255),.03);border-radius:4px}
        .sa-dx-absent{opacity:.4}
        .sa-dx-beat{stroke:rgba(var(--spice-rgb-text,255,255,255),.08);stroke-width:1;vector-effect:non-scaling-stroke}
        .sa-dx-down{stroke:rgba(var(--spice-rgb-text,255,255,255),.24);stroke-width:1;vector-effect:non-scaling-stroke}
        .sa-dx-area{opacity:.85}
        .sa-dx-mix{fill:rgba(var(--spice-rgb-text,255,255,255),.45)}
        .sa-dx-vocals{fill:#ff9f43}.sa-dx-drums{fill:#ef5777}.sa-dx-bass{fill:#8c7ae6}
        .sa-dx-guitar{fill:#2ed573}.sa-dx-piano{fill:#48b0f7}.sa-dx-other{fill:#a4b0be}
        .sa-dx-pitch{fill:none;stroke:#fff;stroke-width:1.6;stroke-linejoin:round;stroke-linecap:round;vector-effect:non-scaling-stroke}
        .sa-dx-hits{fill:none;stroke:#ef5777;stroke-width:2;vector-effect:non-scaling-stroke}
        .sa-dx-axis{display:flex;justify-content:space-between;margin:2px 0 14px 106px;color:var(--spice-subtext,#b3b3b3);font-size:11px;font-variant-numeric:tabular-nums}
        .sa-dx-pattern{display:grid;grid-template-columns:96px repeat(16,minmax(0,1fr));gap:3px;align-items:center;margin:0 0 4px}
        .sa-dx-pattern>span:first-child{font-size:12px;font-weight:600;margin-right:7px}
        .sa-dx-step{height:16px;border-radius:3px;background:rgba(var(--spice-rgb-text,255,255,255),.07);overflow:hidden}
        .sa-dx-step.sa-dx-on-beat{background:rgba(var(--spice-rgb-text,255,255,255),.13)}
        .sa-dx-step i{display:block;height:100%;background:#ef5777}
        .sa-dx-quiet{color:var(--spice-subtext,#b3b3b3);font-size:13px;margin:0}
        .sa-dx-foot{margin:18px 0 0;color:var(--spice-subtext,#b3b3b3);font-size:12px}
        .sa-more-row{display:flex;justify-content:center;margin:14px 0 0}
        .sa-more,.sa-recent-toggle{min-height:32px;border:1px solid rgba(var(--spice-rgb-text,255,255,255),.2);border-radius:999px;padding:5px 16px;background:transparent;color:var(--spice-text,#fff);font:inherit;font-weight:600;cursor:pointer}
        .sa-more[hidden],.sa-recent[hidden]{display:none}
        .sa-more:disabled{border-color:transparent;color:var(--spice-subtext,#b3b3b3);cursor:default}
        .sa-more:hover:enabled,.sa-recent-toggle:hover{border-color:var(--spice-button,#1ed760)}
        .sa-more:focus-visible,.sa-recent-toggle:focus-visible,.sa-recent-clear:focus-visible{outline:2px solid var(--spice-button,#1ed760);outline-offset:2px}
        .sa-recent-toggle{flex:none;align-self:flex-start;margin-left:auto}
        .sa-recent-toggle[aria-expanded="true"]{border-color:var(--spice-button,#1ed760)}
        .sa-recent{max-width:680px;margin:0 0 18px;padding:12px;border:1px solid rgba(var(--spice-rgb-text,255,255,255),.12);border-radius:8px;background:rgba(var(--spice-rgb-text,255,255,255),.04)}
        .sa-recent-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:0 0 8px;color:var(--spice-subtext,#b3b3b3);font-size:12px;font-weight:700}
        .sa-recent-clear{padding:0;border:0;background:none;color:inherit;font:inherit;text-decoration:underline;cursor:pointer}
        .sa-recent-clear:hover{color:var(--spice-text,#fff)}
        .sa-recent-list{display:flex;flex-direction:column;max-height:312px;overflow:auto}
        .sa-recent-item{display:flex;align-items:center;gap:10px;min-height:48px;padding:4px 6px;border:0;border-radius:6px;background:transparent;color:var(--spice-text,#fff);font:inherit;text-align:left;cursor:pointer}
        .sa-recent-item:hover,.sa-recent-item:focus-visible{background:rgba(var(--spice-rgb-text,255,255,255),.1);outline:none}
        .sa-recent-item img,.sa-recent-art{flex:none;width:40px;height:40px;border-radius:4px;object-fit:cover;background:#282828;display:grid;place-items:center;color:#727272}
        .sa-recent-text{display:flex;flex-direction:column;min-width:0}
        .sa-recent-title{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
        .sa-recent-item[aria-current="true"] .sa-recent-title{color:var(--spice-button,#1ed760)}
        .sa-recent-artist{color:var(--spice-subtext,#b3b3b3);font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
        .sa-beta{display:inline-block;border:1px solid rgba(var(--spice-rgb-text,255,255,255),.35);border-radius:999px;padding:0 7px;color:var(--spice-subtext,#b3b3b3);font-size:11px;font-weight:600;vertical-align:1px}
        .sa-part-open,.sa-part-controls button,.sa-part-clear{min-height:32px;border:1px solid rgba(var(--spice-rgb-text,255,255,255),.2);border-radius:999px;padding:5px 14px;background:transparent;color:var(--spice-text,#fff);font:inherit;font-weight:600;cursor:pointer}
        .sa-part-open{flex:none;align-self:center}
        .sa-part-open:hover,.sa-part-open[aria-expanded="true"],.sa-part-controls button:hover:enabled,.sa-part-clear:hover,.sa-part-length[aria-pressed="true"]{border-color:var(--spice-button,#1ed760)}
        .sa-part-open:focus-visible,.sa-part-controls button:focus-visible,.sa-part-clear:focus-visible,.sa-part-span:focus-visible{outline:2px solid var(--spice-button,#1ed760);outline-offset:2px}
        .sa-part{max-width:680px;margin:0 0 18px;padding:14px;border:1px solid rgba(var(--spice-rgb-text,255,255,255),.12);border-radius:8px;background:rgba(var(--spice-rgb-text,255,255,255),.04)}
        .sa-part-intro{margin:0 0 10px}
        .sa-part-track{position:relative;height:48px;border-radius:6px;background:linear-gradient(rgba(var(--spice-rgb-text,255,255,255),.25),rgba(var(--spice-rgb-text,255,255,255),.25)) center/100% 4px no-repeat,rgba(var(--spice-rgb-text,255,255,255),.06);touch-action:none;cursor:pointer;user-select:none}
        .sa-part-span{position:absolute;top:0;bottom:0;box-sizing:border-box;border:2px solid var(--spice-button,#1ed760);border-radius:6px;background:rgba(var(--spice-rgb-button,30,215,96),.22);cursor:grab}
        .sa-part-edge{position:absolute;top:0;bottom:0;width:20px;cursor:ew-resize}
        .sa-part-edge[data-edge="start"]{left:-10px}
        .sa-part-edge[data-edge="end"]{right:-10px}
        .sa-part-edge::after{content:"";position:absolute;top:50%;left:50%;width:4px;height:20px;margin:-10px 0 0 -2px;border-radius:2px;background:var(--spice-button,#1ed760)}
        .sa-part-ticks{display:flex;justify-content:space-between;margin:4px 0 0;color:var(--spice-subtext,#b3b3b3);font-size:11px}
        .sa-part-label{margin:8px 0;font-weight:700}
        .sa-part-controls{display:flex;flex-wrap:wrap;align-items:center;gap:8px}
        .sa-part-lengths{display:inline-flex;gap:6px}
        .sa-part-length[aria-pressed="true"]{background:rgba(var(--spice-rgb-button,30,215,96),.22)}
        .sa-part-apply:enabled{border-color:var(--spice-button,#1ed760);background:var(--spice-button,#1ed760);color:#000}
        .sa-part-controls button:disabled{color:var(--spice-subtext,#b3b3b3);cursor:default}
        .sa-part-note{margin:10px 0 0;color:var(--spice-subtext,#b3b3b3);font-size:12px}
        .sa-part-chip{display:inline-flex;flex-wrap:wrap;align-items:center;gap:6px;margin:0 0 16px;padding:4px 5px 4px 14px;border:1px solid var(--spice-button,#1ed760);border-radius:999px;font-weight:600}
        .sa-part-clear{min-height:28px;padding:3px 12px}
        @container (max-width:760px){
          .sa-list-head,.sa-row-content{grid-template-columns:32px 44px minmax(0,1fr) 56px}
          .sa-view-best .sa-list-head,.sa-view-best .sa-row-content,.sa-view-listeners .sa-list-head,.sa-view-melody .sa-list-head,.sa-view-listeners .sa-row-content,.sa-view-melody .sa-row-content{grid-template-columns:32px 44px minmax(0,1fr)}
          .sa-list-head.sa-with-stems,.sa-row-stems .sa-row-content{grid-template-columns:32px 44px minmax(0,1fr) 56px 64px 72px}
          .sa-view-best .sa-list-head.sa-with-stems,.sa-view-best .sa-row-stems .sa-row-content,.sa-view-listeners .sa-list-head.sa-with-stems,.sa-view-melody .sa-list-head.sa-with-stems,.sa-view-listeners .sa-row-stems .sa-row-content,.sa-view-melody .sa-row-stems .sa-row-content{grid-template-columns:32px 44px minmax(0,1fr) 64px 72px}
          .sa-album,.sa-list-head .sa-album-head{display:none}
          .sa-feedback-help{display:block}
          .sa-dx-lane{grid-template-columns:72px minmax(0,1fr)}
          .sa-dx-pattern{grid-template-columns:72px repeat(16,minmax(0,1fr))}
          .sa-dx-axis{margin-left:82px}
        }
        @container (max-width:460px){
          .sa-list-head.sa-with-stems,.sa-row-stems .sa-row-content{grid-template-columns:32px 44px minmax(0,1fr) 56px 40px 72px}
          .sa-view-best .sa-list-head.sa-with-stems,.sa-view-best .sa-row-stems .sa-row-content,.sa-view-listeners .sa-list-head.sa-with-stems,.sa-view-melody .sa-list-head.sa-with-stems,.sa-view-listeners .sa-row-stems .sa-row-content,.sa-view-melody .sa-row-stems .sa-row-content{grid-template-columns:32px 44px minmax(0,1fr) 40px 72px}
          .sa-match-value{display:none}
        }
      </style>
    `;
  }

  function selectResultsView(view, focus = false) {
    const state = resultsState;
    if (!state || !resultViews(state).some(([key]) => key === view)) return;
    state.viewChosen = true;
    saveResultView(view);
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
    dissectView = null;
    if (!activePage) return;
    // A search comes back scrolled to where it was left.
    if (activePage.state) activePage.state.scrollTop = activePage.host?.scrollTop || 0;
    activePage.view?.querySelectorAll(".sa-row").forEach((row) => {
      row.__soundalikeRoot?.unmount?.();
    });
    activePage.resizeObserver?.disconnect?.();
    activePage.mutationObserver?.disconnect?.();
    if (activePage.syncHost) {
      window.removeEventListener?.("resize", activePage.syncHost);
    }
    activePage.host?.remove?.();
    activePage = null;
    nativeContextChain = undefined;
  }

  // The element the page covers. One without a box is passed over: a theme
  // can collapse the empty route's <main>, which older clients fall back to.
  function resultsContainer() {
    let first = null;
    for (const selector of MAIN_VIEW_SELECTORS) {
      const element = document.querySelector(selector);
      if (!element) continue;
      first ||= element;
      const bounds = element.getBoundingClientRect?.();
      if (!bounds || (bounds.width > 0 && bounds.height > 0)) return element;
    }
    return first;
  }

  function sameBox(left, right) {
    return ["left", "top", "width", "height"].every(
      (side) => Math.abs(left[side] - right[side]) <= 2
    );
  }

  // Inside Spotify's main view panel the page takes the panel's box, its
  // rounded corners and its place under the player bar and menus, whatever a
  // theme does to them. Without the panel (older clients), or when the page
  // doesn't land on the panel's box, it floats over the container instead.
  function createResultsHost(container, view) {
    const host = document.createElement("div");
    host.className = "sa-results-host";
    Object.assign(host.style, {
      overflow: "auto",
      background: "var(--spice-main, #121212)",
      // The narrow layout follows the main view's width, not the window's,
      // so the side panels can't push the Match column out of view.
      containerType: "inline-size",
    });
    host.appendChild(view);
    let inside = Boolean(container.matches?.(MAIN_VIEW_PANEL)) &&
      typeof container.appendChild === "function";
    if (inside) {
      Object.assign(host.style, { position: "absolute", inset: "0", zIndex: "1" });
      container.appendChild(host);
      const box = host.getBoundingClientRect?.();
      const panel = container.getBoundingClientRect?.();
      if (box && panel && !sameBox(box, panel)) {
        host.remove?.();
        inside = false;
      }
    }
    if (!inside) {
      Object.assign(host.style, { position: "fixed", inset: "", zIndex: "10" });
      document.body.appendChild(host);
    }
    const syncHost = () => {
      const bounds = container.getBoundingClientRect?.();
      if (!inside && bounds) {
        host.style.left = `${bounds.left}px`;
        host.style.top = `${bounds.top}px`;
        host.style.width = `${bounds.width}px`;
        host.style.height = `${bounds.height}px`;
      }
      // A player bar that floats over the main view, as several themes have
      // it, must not hide the last rows.
      const box = host.getBoundingClientRect?.();
      const covered = !box ? 0 : Math.max(0, ...Array.from(
        document.querySelectorAll?.(PLAYER_BAR_SELECTOR) || [],
        (element) => {
          const bar = element.getBoundingClientRect();
          return bar.height > 0 &&
            bar.top < box.bottom && bar.bottom > box.top &&
            bar.left < box.right && bar.right > box.left
            ? Math.round(box.bottom - bar.top)
            : 0;
        }
      ));
      view.style.paddingBottom = covered > 0 ? `${PAGE_BOTTOM_PADDING + covered}px` : "";
    };
    syncHost();
    window.addEventListener?.("resize", syncHost);
    const resizeObserver = typeof ResizeObserver === "function"
      ? new ResizeObserver(() => {
        syncHost();
        ensureResultsMounted();
      })
      : null;
    resizeObserver?.observe?.(container);
    const parent = host.parentNode;
    const mutationObserver = parent && typeof MutationObserver === "function"
      ? new MutationObserver(ensureResultsMounted)
      : null;
    mutationObserver?.observe?.(parent, { childList: true });
    return { host, syncHost, resizeObserver, mutationObserver };
  }

  // A theme or Spotify can redraw the main view and drop the page with it, or
  // collapse the element the page covers. It then mounts again on what is
  // there now.
  function ensureResultsMounted() {
    const page = activePage;
    if (!page || Spicetify.Platform.History.location?.pathname !== RESULTS_PATH) return;
    const dropped = page.host.isConnected === false || page.container.isConnected === false;
    const bounds = page.container.getBoundingClientRect?.();
    const collapsed = Boolean(bounds) && !(bounds.width > 0 && bounds.height > 0);
    if (dropped || (collapsed && resultsContainer() !== page.container)) scheduleResultsRoute();
  }

  function viewLimit(state, view) {
    return state.limits?.[view] || DISPLAY_RESULT_COUNT;
  }

  // Fills the shown list's rows from Spotify. Show more calls it again on the
  // same page with a higher limit: found songs are kept, so only new rows load.
  function hydrateShownList(state, page) {
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
        {
          bpm: Boolean(shown.bpm),
          stems: dissectSeed(state, shown.results || []),
          limit: viewLimit(state, view),
        }
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
      state.audioOnly ? new Set() : state.sounds.relatedArtistsPromise,
      state.timing
    ).catch((error) => {
      console.error("[soundalike] Spotify result enrichment failed", error);
    });
  }

  // The hosted answer already holds the longer list, so Show more asks the
  // matcher nothing: the next songs are found on Spotify and join the page.
  function showMoreResults() {
    const state = resultsState;
    const page = activePage;
    if (!state || !page?.view || page.state !== state) return;
    const view = shownView(state);
    state.limits[view] = viewLimit(state, view) + DISPLAY_RESULT_COUNT;
    const more = page.view.querySelector(".sa-more");
    if (more) {
      more.disabled = true;
      more.textContent = "Finding more on Spotify…";
    }
    hydrateShownList(state, page.view);
  }

  function renderResultsPage(container) {
    if (!container || !document.body) return;
    teardownResultsPage();
    const state = resultsState;
    const page = state ? buildResultsPage(state) : buildHomePage();
    const mounted = createResultsHost(container, page);
    activePage = { container, view: page, state, ...mounted };
    nativeContextChain = undefined;
    bindRecentSearches(page);
    if (!state) return;
    bindResultTabs(page);
    const seedPlay = page.querySelector(".sa-seed-play");
    if (seedPlay) seedPlay.onclick = () => {
      if (seedIsCurrent(state)) Spicetify.Player.togglePlay();
      else return playSeedSong(state);
    };
    markPlayingRows(page);
    const offer = page.querySelector(".sa-best-offer");
    if (offer) offer.onclick = requestOfferedBestMatches;
    const languageToggle = page.querySelector(".sa-language-toggle");
    if (languageToggle) languageToggle.onclick = toggleBestLanguageMode;
    const audioOnlyToggle = page.querySelector(".sa-audio-only");
    if (audioOnlyToggle) audioOnlyToggle.onclick = toggleAudioOnly;
    bindPartPicker(page, state);
    const more = page.querySelector(".sa-more");
    if (more) more.onclick = showMoreResults;
    hydrateShownList(state, page);
    if (state.scrollTop > 0) {
      const { host } = activePage;
      const top = state.scrollTop;
      host.scrollTop = top;
      // Rows drawn by React get their height a moment later.
      setTimeout(() => {
        if (activePage?.host === host) host.scrollTop = top;
      }, 60);
    }
  }

  function mountResultsRoute() {
    if (Spicetify.Platform.History.location?.pathname !== RESULTS_PATH) return;
    const container = resultsContainer();
    if (!container) {
      clearTimeout(routeMountTimer);
      routeMountTimer = setTimeout(mountResultsRoute, 100);
      return;
    }
    // Opened without a search this session (the top bar button, or Spotify
    // starting on this page): the latest search comes back, once.
    if (!resultsState && !latestRestored) {
      latestRestored = true;
      const [latest] = searchHistory();
      if (latest) void restoreSearch(latest.uri, false);
    }
    renderResultsPage(container);
  }

  function scheduleResultsRoute() {
    clearTimeout(routeMountTimer);
    routeMountTimer = setTimeout(mountResultsRoute, 0);
  }

  // Each history entry names its search, so Spotify's Back and Forward step
  // through earlier searches instead of showing the latest one again.
  function onHistoryChange(location) {
    const current = Spicetify.Platform.History.location;
    const path = location?.pathname || current?.pathname;
    if (path !== RESULTS_PATH) {
      teardownResultsPage();
      return;
    }
    const seed = (location?.state ?? current?.state)?.seed;
    if (
      typeof seed === "string" &&
      seed.startsWith("spotify:track:") &&
      seed !== resultsState?.seedUri
    ) {
      void restoreSearch(seed, false);
    }
    scheduleResultsRoute();
  }

  // Spotify's history drops a push to the address it is already on, so each
  // search gets its own address under the one path.
  function pushResultsRoute(seedUri) {
    Spicetify.Platform.History.push({
      pathname: RESULTS_PATH,
      search: seedUri ? `?song=${encodeURIComponent(seedUri.split(":track:")[1] || "")}` : "",
      state: { soundalike: true, ...(seedUri ? { seed: seedUri } : {}) },
    });
  }

  function showResultsPage(state, navigate = true) {
    resultsState = state;
    keepState(state);
    rememberSearch(state);
    if (navigate) pushResultsRoute(state.seedUri);
    scheduleResultsRoute();
  }

  // Reopens an earlier search: its page as it was left when this session
  // still has it, else from the saved lists, which asks the matcher nothing
  // while they are saved.
  function restoreSearch(seedUri, navigate = true) {
    const kept = keptStates.get(seedUri);
    if (!kept) return searchSoundalikes([seedUri], { quiet: true, navigate });
    // A search still on its way must not open over this one.
    searchSequence++;
    kept.recentOpen = false;
    showResultsPage(kept, navigate);
    return Promise.resolve();
  }

  // The top bar button: back to the page from anywhere in Spotify.
  function returnToResults() {
    if (Spicetify.Platform.History.location?.pathname === RESULTS_PATH) return;
    pushResultsRoute(resultsState?.seedUri);
  }

  function recentToggleMarkup(state) {
    return searchHistory().length
      ? `<button class="sa-recent-toggle" type="button" aria-expanded="${Boolean(state.recentOpen)}" title="Earlier searches">Recent</button>`
      : "";
  }

  function recentSearchesMarkup(currentUri, open) {
    const items = searchHistory();
    if (!items.length) return "";
    const rows = items.map((item) =>
      `<button class="sa-recent-item" type="button" data-uri="${esc(item.uri)}"${item.uri === currentUri ? ' aria-current="true"' : ""}>` +
      (item.image
        ? `<img src="${esc(item.image)}" alt="" loading="lazy">`
        : '<span class="sa-recent-art" aria-hidden="true">♫</span>') +
      `<span class="sa-recent-text"><span class="sa-recent-title">${esc(item.title)}</span>` +
      `<span class="sa-recent-artist">${esc(item.artist)}</span></span></button>`
    ).join("");
    return `<section class="sa-recent"${open ? "" : " hidden"} aria-label="Recent searches">
        <div class="sa-recent-head"><span>Recent searches</span><button class="sa-recent-clear" type="button">Clear</button></div>
        <div class="sa-recent-list">${rows}</div>
      </section>`;
  }

  function bindRecentSearches(page) {
    const toggle = page.querySelector(".sa-recent-toggle");
    const panel = page.querySelector(".sa-recent");
    if (toggle && panel) toggle.onclick = () => {
      const open = Boolean(panel.hidden);
      panel.hidden = !open;
      toggle.setAttribute?.("aria-expanded", String(open));
      if (resultsState) resultsState.recentOpen = open;
    };
    Array.from(page.querySelectorAll(".sa-recent-item") || []).forEach((item) => {
      item.onclick = () => {
        const uri = item.dataset?.uri;
        if (!uri) return;
        if (uri !== resultsState?.seedUri) return restoreSearch(uri);
        if (toggle) toggle.onclick();
      };
    });
    const clear = page.querySelector(".sa-recent-clear");
    if (clear) clear.onclick = () => {
      clearSearchHistory();
      if (resultsState) resultsState.recentOpen = false;
      scheduleResultsRoute();
    };
  }

  // The page before any search: how to start one, and the earlier searches.
  function buildHomePage() {
    const wrap = document.createElement("div");
    wrap.className = "sa-results";
    wrap.innerHTML = `${pageStyleMarkup()}
      <div class="sa-seed">
        <div>
          <div class="sa-kicker">SOUNDALIKE</div>
          <div class="sa-seed-title">Find soundalikes</div>
        </div>
      </div>
      <p class="sa-blurb">Right-click a song and choose Find soundalikes. Each search stays here, so you can come back to it.</p>
      ${recentSearchesMarkup(null, true)}`;
    return wrap;
  }

  // ---- Stems: a match next to its seed, instrument by instrument ----------

  // The seed as the dissector knows it: the hosted list's song, by Deezer id.
  // Lists whose rows carry no ids (older servers) get no Stems column.
  function dissectSeed(state, results = null) {
    const query = state?.best?.status === "ready" ? state.best.data?.query : null;
    const id = deezerId(query?.id);
    if (!id) return null;
    if (results && !results.some((result) => deezerId(result?.id) && result.id !== id)) return null;
    return {
      id,
      title: query.title || state.seedTrack?.name || "",
      artist: query.artist || seedArtistNames(state.seedTrack).join(", "),
      uri: state.seedUri || null,
    };
  }

  function dissectMessage(status, serverError = "") {
    const reason = String(serverError || "");
    if (status === 404 && reason.includes("not enabled")) {
      return "Stems aren't switched on for the hosted library yet.";
    }
    if (status === 404 && reason.includes("not been dissected")) {
      return "These songs haven't been dissected yet.";
    }
    if (status === 404) return "One of these songs has no Deezer preview to dissect.";
    if (status === 400) return "These songs can't be dissected.";
    if (status === 429) return "You're going a bit fast. Wait a few seconds and try again.";
    if (status === 503) return "The dissector is busy right now. Try again in a moment.";
    if (status === 504) return "Dissecting took too long. Try again.";
    if (status === 502) return "Deezer didn't respond. Try again in a moment.";
    if (status === 422) return "We couldn't decode one of these previews.";
    return "Stems are unavailable right now. Try again in a moment.";
  }

  async function fetchDissection(seedId, matchId) {
    const params = new URLSearchParams({ id: String(seedId), with: String(matchId) });
    for (let attempt = 0; ; attempt++) {
      let response;
      try {
        response = await fetchWithTimeout(
          `${DISSECT_ENDPOINT}?${params}`,
          { cache: "default" },
          DISSECT_TIMEOUT_MS
        );
      } catch (error) {
        const status = error?.name === "AbortError" ? 504 : 0;
        throw bestMatchesError(status, dissectMessage(status));
      }
      let body = null;
      try {
        body = await response.json();
      } catch {
        body = null;
      }
      // 503: the one live-dissection slot is taken; it frees within seconds.
      if (response.status === 503 && attempt < DISSECT_RETRIES) {
        await delay(DISSECT_RETRY_DELAY_MS * (attempt + 1));
        continue;
      }
      const songs = Array.isArray(body?.songs) ? body.songs : [];
      if (
        !response.ok ||
        !body?.ok ||
        songs.length !== 2 ||
        !songs.every((song) => song && typeof song === "object")
      ) {
        const status = response.ok ? 0 : response.status;
        throw bestMatchesError(status, dissectMessage(status, body?.error));
      }
      return body;
    }
  }

  // One request per song pair; a failed one is forgotten so it can be retried.
  function requestDissection(seedId, matchId) {
    const key = `${seedId}:${matchId}`;
    const known = dissections.get(key);
    if (known) {
      dissections.delete(key);
      dissections.set(key, known);
      return known;
    }
    const request = fetchDissection(seedId, matchId);
    request.catch(() => {
      if (dissections.get(key) === request) dissections.delete(key);
    });
    dissections.set(key, request);
    while (dissections.size > DISSECT_CACHE_SIZE) {
      dissections.delete(dissections.keys().next().value);
    }
    return request;
  }

  // Stems replaces the list inside the results page; Back (or Escape) brings
  // the list back where it was.
  function openDissection(seed, result, track) {
    const page = activePage;
    if (!page?.host || !page.view) return;
    closeDissection();
    const panel = document.createElement("div");
    panel.className = "sa-results sa-dx";
    const view = {
      page,
      panel,
      seed,
      result,
      matchUri: track?.uri || null,
      scrollTop: page.host.scrollTop || 0,
      opener: document.activeElement || null,
    };
    dissectView = view;
    panel.onkeydown = (event) => {
      if (event.key !== "Escape") return;
      event.preventDefault?.();
      closeDissection();
    };
    page.view.hidden = true;
    page.host.appendChild(panel);
    page.host.scrollTop = 0;
    loadDissection(view, true);
  }

  function closeDissection() {
    const view = dissectView;
    if (!view) return;
    dissectView = null;
    view.panel.remove?.();
    if (view.page !== activePage) return;
    view.page.view.hidden = false;
    view.page.host.scrollTop = view.scrollTop;
    view.opener?.focus?.();
  }

  function loadDissection(view, focusBack = false) {
    renderDissection(view, { status: "pending" }, focusBack);
    requestDissection(view.seed.id, view.result.id).then(
      (data) => {
        if (dissectView === view) renderDissection(view, { status: "ready", data });
      },
      (error) => {
        if (dissectView !== view) return;
        console.warn("[soundalike] Stems failed", error);
        renderDissection(
          view,
          { status: "error", error: error?.message || dissectMessage(0) },
          true
        );
      }
    );
  }

  function renderDissection(view, state, focusBack = false) {
    const refocus = focusBack || Boolean(view.panel.contains?.(document.activeElement));
    view.state = state;
    view.panel.innerHTML = dissectionMarkup(view, state);
    const back = view.panel.querySelector(".sa-dx-back");
    if (back) {
      back.onclick = closeDissection;
      if (refocus) back.focus?.();
    }
    const retry = view.panel.querySelector(".sa-dx-retry");
    if (retry) retry.onclick = () => loadDissection(view, true);
    (view.panel.querySelectorAll(".sa-dx-play") || []).forEach((button) => {
      button.onclick = () => playDissectedSong(button.dataset.uri, button.dataset.title);
    });
  }

  // Like the list's own rows: the match plays from its place in the list, and
  // the seed plays first with the list after it, as the header button does.
  async function playDissectedSong(uri, title) {
    if (!uri) return;
    const shown = shownResultUris();
    try {
      const played = shown.includes(uri)
        ? await playResultsList(shown, shown.indexOf(uri))
        : await playResultsList([uri, ...shown], 0);
      if (played) return;
    } catch (error) {
      console.warn(
        "[soundalike] Spotify would not play the song with its results; playing it alone.",
        error
      );
    }
    try {
      await Spicetify.Player.playUri(uri);
    } catch (error) {
      console.error(`[soundalike] Could not play ${uri}`, error);
      Spicetify.showNotification(`Could not play ${title || "this song"}.`, true);
    }
  }

  // JSON null means "not measured", which must never read as zero.
  function measured(value) {
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  }

  // Lanes arrive as base64 bytes, one per tenth of a second.
  function dissectBytes(value) {
    if (typeof value !== "string" || typeof atob !== "function") return null;
    try {
      const raw = atob(value);
      return raw.length === DISSECT_FRAMES
        ? Uint8Array.from(raw, (char) => char.charCodeAt(0))
        : null;
    } catch {
      return null;
    }
  }

  function svgNum(value) {
    return Math.round(value * 10) / 10;
  }

  function beatPath(times, seconds) {
    return times
      .filter((time) => Number.isFinite(time) && time >= 0 && time <= seconds)
      .map((time) => `M${svgNum((time / seconds) * DISSECT_FRAMES)} 0V100`)
      .join("");
  }

  // Loudness bytes span 60 dB below full scale (0) to full scale (255).
  function loudnessPath(bytes, height) {
    let path = `M0 ${height}`;
    bytes.forEach((value, index) => {
      path += `L${index + 0.5} ${svgNum(height - (value / 255) * (height - 2))}`;
    });
    return `${path}L${DISSECT_FRAMES} ${height}Z`;
  }

  // Pitch bytes are MIDI notes times two; 0 is unvoiced.
  function pitchPath(bytes, height) {
    const notes = Array.from(bytes, (value) => (value ? value / 2 : null));
    const voiced = notes.filter((note) => note !== null);
    if (voiced.length < 3) return "";
    let low = Math.min(...voiced) - 2;
    let high = Math.max(...voiced) + 2;
    if (high - low < 12) {
      const middle = (high + low) / 2;
      low = middle - 6;
      high = middle + 6;
    }
    let path = "";
    let drawing = false;
    notes.forEach((note, index) => {
      if (note === null) {
        drawing = false;
        return;
      }
      const y = height - 2 - ((note - low) / (high - low)) * (height - 4);
      path += `${drawing ? "L" : "M"}${index + 0.5} ${svgNum(y)}`;
      drawing = true;
    });
    return path;
  }

  function hitsPath(bytes, height) {
    let path = "";
    bytes.forEach((value, index) => {
      if (value >= 48) path += `M${index + 0.5} ${height}V${svgNum(height - (value / 255) * height)}`;
    });
    return path;
  }

  function laneMarkup(label, note, body, grid, { height = 34, absent = false } = {}) {
    return `<div class="sa-dx-lane${absent ? " sa-dx-absent" : ""}">` +
      `<div class="sa-dx-lane-label">${esc(label)}${note ? `<span>${esc(note)}</span>` : ""}</div>` +
      `<svg class="sa-dx-svg" viewBox="0 0 ${DISSECT_FRAMES} ${height}" preserveAspectRatio="none" ` +
      `style="height:${height}px" aria-hidden="true" focusable="false">${grid}${body}</svg></div>`;
  }

  function formatDb(value) {
    return `${value < 0 ? "\u2212" : "+"}${Math.abs(value).toFixed(1)} dB`;
  }

  function patternMarkup(pattern, drumsPresent) {
    const rows = DISSECT_DRUMS.map(([name, label]) => {
      const steps = Array.isArray(pattern?.[name]) ? pattern[name].map(Number) : [];
      return { label, steps: steps.length === 16 && steps.every(Number.isFinite) ? steps : null };
    });
    if (
      !drumsPresent ||
      rows.some((row) => !row.steps) ||
      rows.every((row) => row.steps.every((value) => value <= 0))
    ) {
      return '<p class="sa-dx-quiet">No steady drum pattern in this preview.</p>';
    }
    return '<div role="img" aria-label="Kick, snare and hats on the 16 steps of an average bar">' +
      rows.map(({ label, steps }) =>
        `<div class="sa-dx-pattern"><span>${esc(label)}</span>` +
        steps.map((value, index) =>
          `<span class="sa-dx-step${index % 4 === 0 ? " sa-dx-on-beat" : ""}">` +
          `<i style="opacity:${(Math.max(0, Math.min(255, value)) / 255).toFixed(2)}"></i></span>`
        ).join("") +
        "</div>"
      ).join("") +
      "</div>";
  }

  function dissectedSongMarkup(song, { role, title, artist, uri }) {
    const seconds = measured(song.seconds) > 0 ? song.seconds : 30;
    const times = (list) => (Array.isArray(list) ? list.map(measured) : []);
    const grid = `<path class="sa-dx-beat" d="${beatPath(times(song.beats), seconds)}"/>` +
      `<path class="sa-dx-down" d="${beatPath(times(song.downbeats), seconds)}"/>`;
    const lanes = [];
    const mix = dissectBytes(song.mix);
    if (mix) {
      lanes.push(laneMarkup(
        "Whole mix",
        "",
        `<path class="sa-dx-area sa-dx-mix" d="${loudnessPath(mix, 34)}"/>`,
        grid
      ));
    }
    const stems = Array.isArray(song.stems) ? song.stems : [];
    for (const [name, label] of DISSECT_STEMS) {
      const stem = stems.find((item) => item?.name === name);
      const loudness = stem ? dissectBytes(stem.loudness) : null;
      if (!loudness) continue;
      const present = stem.present === true;
      const pitch = present && (name === "vocals" || name === "bass")
        ? dissectBytes(song.pitch?.[name])
        : null;
      const melody = pitch ? pitchPath(pitch, 34) : "";
      const balance = measured(stem.balance_db);
      const note = !present
        ? "not in this song"
        : balance !== null ? `${formatDb(balance)} vs mix` : "";
      lanes.push(laneMarkup(
        label,
        note,
        `<path class="sa-dx-area sa-dx-${name}" d="${loudnessPath(loudness, 34)}"/>` +
          (melody ? `<path class="sa-dx-pitch" d="${melody}"/>` : ""),
        grid,
        { absent: !present }
      ));
    }
    const drumsPresent = stems.find((item) => item?.name === "drums")?.present === true;
    if (drumsPresent) {
      for (const [name, label] of DISSECT_DRUMS) {
        const hits = dissectBytes(song.drums?.[name]);
        const rate = measured(song.drum_rates?.[name]);
        if (!hits) continue;
        lanes.push(laneMarkup(
          label,
          rate !== null ? `${rate.toFixed(1)} hits/s` : "",
          `<path class="sa-dx-hits" d="${hitsPath(hits, 18)}"/>`,
          grid,
          { height: 18 }
        ));
      }
    }
    const chips = [];
    const bpm = measured(song.bpm);
    if (bpm > 0) chips.push(`${Math.round(bpm)} BPM`);
    if (song.key?.tonic) {
      chips.push(`${song.key.tonic} ${song.key.mode === "minor" ? "minor" : "major"}`);
    }
    if (song.vocals?.median) {
      const range = measured(song.vocals.range_st);
      chips.push(`Vocals around ${song.vocals.median}` +
        (range > 0 ? ` · ${Math.round(range)}-semitone range` : ""));
    }
    if (song.bass?.median) chips.push(`Bass around ${song.bass.median}`);
    const play = uri
      ? `<button class="sa-dx-play" type="button" data-uri="${esc(uri)}" data-title="${esc(title)}" ` +
        `title="Play ${esc(title)}" aria-label="Play ${esc(title)}">▶</button>`
      : "";
    return `<section class="sa-dx-song" aria-label="${esc(`${role}: ${title}`)}">
        <div class="sa-dx-song-head">${play}<div>
          <div class="sa-dx-role">${esc(role)}</div>
          <div class="sa-dx-song-title">${esc(title)}</div>
          <div class="sa-dx-song-artist">${esc(artist)}</div>
        </div></div>
        <div class="sa-tags">${tagsMarkup(chips)}</div>
        ${lanes.join("")}
        <div class="sa-dx-axis" aria-hidden="true"><span>0 s</span><span>10 s</span><span>20 s</span><span>${Math.round(seconds)} s</span></div>
        <h3>One-bar drum pattern</h3>
        ${patternMarkup(song.pattern, drumsPresent)}
      </section>`;
  }

  // Graded facets are percentiles among typical match pairs. Until the
  // dissector is calibrated, raw measures show in their own units.
  function facetReading(facet, graded) {
    if (graded) {
      const grade = measured(facet.grade);
      return grade !== null
        ? {
          fill: grade / 100,
          text: String(Math.round(grade)),
          title: `More alike than ${Math.round(grade)}% of typical match pairs`,
        }
        : null;
    }
    const value = measured(facet.value);
    if (value === null) return null;
    if (DISSECT_SIMILARITIES.has(facet.name)) {
      return { fill: value, text: value.toFixed(2), title: "Similarity; 1.00 is identical" };
    }
    if (facet.name === "key") {
      const steps = Math.round(value);
      return {
        fill: 1 - value / 6,
        text: steps ? `${steps} ${steps === 1 ? "step" : "steps"} apart` : "same key",
        title: "Steps apart on the circle of fifths",
      };
    }
    if (facet.name === "vocal_presence") {
      return {
        fill: 1 - value,
        text: `${Math.round(value * 100)} pts apart`,
        title: "Difference in how much of the preview has singing",
      };
    }
    if (facet.name === "balance") {
      return {
        fill: 1 - value / 40,
        text: `${value.toFixed(1)} dB apart`,
        title: "How differently the parts sit in the mix",
      };
    }
    const semitones = Math.round(value);
    return {
      fill: 1 - value / 12,
      text: `${semitones} ${semitones === 1 ? "semitone" : "semitones"} apart`,
      title: "Distance in semitones",
    };
  }

  function compareMarkup(compare) {
    const graded = compare?.graded === true;
    const facets = (Array.isArray(compare?.facets) ? compare.facets : [])
      .filter((facet) => facet && typeof facet.label === "string" && typeof facet.name === "string")
      .map((facet) => ({ facet, reading: facetReading(facet, graded) }))
      .filter((item) => item.reading);
    if (!facets.length) return "";
    if (graded) facets.sort((left, right) => right.reading.fill - left.reading.fill);
    const matchedOn = new Set(
      graded && Array.isArray(compare.matched_on) ? compare.matched_on : []
    );
    const strong = facets.filter(({ facet }) => matchedOn.has(facet.name));
    const headline = !graded
      ? ""
      : strong.length
        ? `<div class="sa-dx-matched"><span>Matched on</span>${strong
          .map(({ facet }) => `<span class="sa-dx-chip">${esc(facet.label)}</span>`)
          .join("")}</div>`
        : '<p class="sa-dx-hint">No single part stands out; the match comes from the overall sound.</p>';
    const rows = facets.map(({ facet, reading }) => {
      const fill = Math.round(Math.max(0, Math.min(1, reading.fill)) * 100);
      const title = esc(reading.title);
      return `<div class="sa-dx-facet${matchedOn.has(facet.name) ? " sa-dx-strong" : ""}" style="display:contents">` +
        `<span class="sa-dx-facet-label" title="${title}">${esc(facet.label)}</span>` +
        `<span class="sa-dx-bar" title="${title}"><span style="width:${fill}%"></span></span>` +
        `<span class="sa-dx-facet-value" title="${title}">${esc(reading.text)}</span></div>`;
    }).join("");
    const hint = graded
      ? "Grades run 0 to 100 against typical match pairs; 85 or more counts as matched on."
      : "Grades arrive once the dissector is calibrated. Until then these are raw " +
        "measures: similarities run up to 1.00, distances are best at zero.";
    return `<section class="sa-dx-compare" aria-label="How alike, part by part">
        <h3>How alike, part by part</h3>
        ${headline}
        <div class="sa-dx-facets">${rows}</div>
        <p class="sa-dx-hint">${esc(hint)}</p>
      </section>`;
  }

  function dissectionMarkup(view, state) {
    const { seed, result } = view;
    const head = `<div class="sa-dx-top"><button class="sa-dx-back" type="button">← Back to matches</button></div>
      <div class="sa-kicker">STEMS · HOSTED LIBRARY</div>
      <h2 class="sa-dx-title">${esc(seed.title)} <span>vs</span> ${esc(result.title)}</h2>
      <p class="sa-blurb">${esc(DISSECT_BLURB)}</p>`;
    if (state.status === "pending") {
      return `${head}<div class="sa-state" role="status">${esc(DISSECT_PENDING)}</div>`;
    }
    if (state.status === "error") {
      return `${head}<div class="sa-state sa-state-error" role="alert">${esc(state.error)}</div>` +
        '<button class="sa-dx-retry" type="button">Try again</button>';
    }
    const songs = state.data.songs;
    const live = songs.filter((song) => song.source === "live");
    const liveMs = live.reduce((sum, song) => sum + (Number(song.ms) || 0), 0);
    const source = !live.length
      ? "From earlier dissections."
      : liveMs > 0
        ? `Dissected just now in ${(liveMs / 1000).toFixed(1)} s.`
        : "Dissected just now.";
    return `${head}${compareMarkup(state.data.compare)}
      <div class="sa-dx-songs">
        ${dissectedSongMarkup(songs[0], { role: "This song", title: seed.title, artist: seed.artist, uri: seed.uri })}
        ${dissectedSongMarkup(songs[1], { role: "Match", title: result.title, artist: result.artist, uri: view.matchUri })}
      </div>
      <p class="sa-dx-hint">Shaded: each part's loudness over the preview. White line: the sung or bass melody. Bright lines mark bars, faint lines beats.</p>
      <p class="sa-dx-foot">Measured on each song's 30-second Deezer preview. ${esc(source)}</p>`;
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

  Spicetify.Player?.addEventListener?.("onplaypause", () => markPlayingRows());
  Spicetify.Platform.History.listen(onHistoryChange);
  Spicetify.Player?.addEventListener?.("songchange", () => {
    markPlayingRows();
    void queueLateMatches();
    leftResultsList();
  });

  window.__soundalikeContextMenuItem = new Spicetify.ContextMenu.Item(
    "Find soundalikes",
    findSoundalikes,
    onlyTracks,
    "enhance" // Spicetify built-in icon
  );
  window.__soundalikeContextMenuItem.register();
  registerNowPlayingMenuItem();
  registerTopbarButton();
  void localServerReady();
  // Spotify can start on the results page; it then shows the latest search.
  if (Spicetify.Platform.History.location?.pathname === RESULTS_PATH) scheduleResultsRoute();

  // A way back to the results from anywhere in Spotify, next to its own
  // Back and Forward.
  function registerTopbarButton() {
    if (typeof Spicetify.Topbar?.Button !== "function") return;
    try {
      window.__soundalikeTopbarButton = new Spicetify.Topbar.Button(
        "Soundalike results",
        "enhance",
        returnToResults
      );
    } catch (error) {
      console.warn("[soundalike] The top bar button could not be added.", error);
    }
  }

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
          "[soundalike] Web app fallback prewarm was unavailable.",
          error
        );
      }
    }, { timeout: 5000 });
  }

  prewarmFallbackRecommender();

  console.log("[soundalike] extension loaded — right-click a track to try it.");
})();
