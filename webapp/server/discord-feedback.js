const DISCORD_WEBHOOK =
  /^https:\/\/discord\.com\/api\/webhooks\/\d{17,20}\/[A-Za-z0-9._-]{20,}$/u;
const DISCORD_CONTENT_LIMIT = 2000;
const DEFAULT_TIMEOUT_MS = 5000;
const SUPPRESS_EMBEDS = 4;

const SELECTION_LABELS = {
  good: "Good",
  mixed: "Mixed",
  off: "Off",
};

const REASON_LABELS = {
  style: "Style",
  mood_energy: "Mood/energy",
  tempo: "Tempo",
  vocals_language: "Vocals/language",
  instruments_timbre: "Instruments/timbre",
};

function discordText(value, maximumLength = 180) {
  return String(value)
    .slice(0, maximumLength)
    .replace(/([\\`*_~|>#])/gu, "\\$1")
    .replace(/</gu, "\\<")
    .replace(/@/gu, "@\u200b");
}

function reasonText(reasons) {
  return reasons.length
    ? reasons.map((reason) => REASON_LABELS[reason]).join(", ")
    : "None";
}

export function validateDiscordWebhookUrl(value) {
  if (value === undefined || value === "") return null;
  if (typeof value !== "string" || !DISCORD_WEBHOOK.test(value)) {
    throw new Error("Invalid Discord webhook configuration");
  }
  return value;
}

export function formatFeedbackNotification(summary) {
  return [
    "**New Soundalike feedback**",
    `Rating: **${SELECTION_LABELS[summary.selection]}**`,
    `Seed: ${discordText(summary.seed.title)} - ${discordText(summary.seed.artist)}`,
    `Reasons: ${reasonText(summary.reasons)}`,
    `Results shown: ${summary.result_count}`,
    `Receipt: \`${summary.receipt}\``,
  ].join("\n");
}

export function formatFeedbackDigest(summary) {
  const lines = [
    `**Soundalike feedback digest for ${summary.date}**`,
    `Total: **${summary.total}** (Good ${summary.selections.good}, Mixed ${summary.selections.mixed}, Off ${summary.selections.off})`,
  ];
  const reasons = Object.entries(summary.reasons)
    .filter(([, count]) => count > 0)
    .map(([reason, count]) => `${REASON_LABELS[reason]} ${count}`);
  lines.push(`Reasons: ${reasons.length ? reasons.join(", ") : "None"}`);
  if (summary.flagged.length) {
    lines.push("", "**Needs review**");
    for (const item of summary.flagged.slice(0, 5)) {
      lines.push(
        `- ${SELECTION_LABELS[item.selection]}: ${discordText(item.seed.title, 100)} - ${discordText(item.seed.artist, 100)} (${reasonText(item.reasons)})`,
      );
    }
  }
  lines.push("", "Free-text notes remain in private storage.");
  return lines.join("\n").slice(0, DISCORD_CONTENT_LIMIT);
}

export function createDiscordWebhookSender(options = {}) {
  const fetchImpl = options.fetchImpl ?? globalThis.fetch;
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  if (typeof fetchImpl !== "function") {
    throw new Error("Discord webhook transport is unavailable");
  }
  if (
    !Number.isInteger(timeoutMs) ||
    timeoutMs < 100 ||
    timeoutMs > 30_000
  ) {
    throw new Error("Invalid Discord webhook timeout");
  }
  return async function sendDiscordWebhook(content) {
    const webhookUrl = validateDiscordWebhookUrl(
      options.webhookUrl ??
        process.env.SOUNDALIKE_FEEDBACK_DISCORD_WEBHOOK,
    );
    if (!webhookUrl) return false;
    if (
      typeof content !== "string" ||
      content.length < 1 ||
      content.length > DISCORD_CONTENT_LIMIT
    ) {
      throw new Error("Invalid Discord webhook message");
    }
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetchImpl(webhookUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content,
          allowed_mentions: { parse: [] },
          flags: SUPPRESS_EMBEDS,
          username: "Soundalike Feedback",
        }),
        redirect: "error",
        signal: controller.signal,
      });
      if (!response?.ok) {
        throw new Error(
          `Discord webhook request failed (${response?.status ?? "unknown"})`,
        );
      }
      return true;
    } finally {
      clearTimeout(timeout);
    }
  };
}
