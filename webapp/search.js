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

  pick = async item => {
    hideAc();
    input.value = `${item.title} — ${item.artist}`;
    document.querySelector("#out").innerHTML =
      '<div class="state"><span class="spin"></span> matching…</div>';
    try {
      const body = await requestRecommendations(item);
      if (!body.ok) {
        document.querySelector("#out").innerHTML =
          `<div class="state err">${esc(body.error || "No match.")}</div>`;
        return;
      }
      render(body);
    } catch {
      document.querySelector("#out").innerHTML =
        '<div class="state err">Server error.</div>';
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
