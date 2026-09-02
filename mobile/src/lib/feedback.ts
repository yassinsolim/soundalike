import AsyncStorage from "@react-native-async-storage/async-storage";
import * as Crypto from "expo-crypto";

import {
  FEEDBACK_ENDPOINT,
  FEEDBACK_LANGUAGE_POLICY,
  FEEDBACK_SELECTION_POLICY,
  FEEDBACK_SOURCE,
  FEEDBACK_SURVEY_VERSION,
  HOSTED_API_VERSION,
} from "./config";
import type {
  FeedbackReason,
  FeedbackSelection,
  RecommendationSet,
} from "./types";

const INSTALL_NONCE_KEY = "soundalike.install_nonce.v1";
const KNOWN_METHODS = new Set([
  "dual_sonic64_guardrail",
  "sonic64_stable_head",
  "legacy_no_sonic_seed",
]);
const INDEX_VERSION = /^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$/;

export function randomNonce(): string {
  return Array.from(Crypto.getRandomBytes(16))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

const sessionNonce = randomNonce();

/** Stable per-install identifier. It is random and carries no device identity. */
export async function installNonce(): Promise<string> {
  const stored = await AsyncStorage.getItem(INSTALL_NONCE_KEY);
  if (stored && /^[a-f0-9]{32}$/.test(stored)) return stored;
  const created = randomNonce();
  await AsyncStorage.setItem(INSTALL_NONCE_KEY, created);
  return created;
}

export function buildFeedbackPayload(
  set: RecommendationSet,
  selection: FeedbackSelection,
  reasons: FeedbackReason[],
  note: string,
  nonces: { install: string; session: string }
) {
  const isGood = selection === "good";
  return {
    schema_version: 1,
    survey_version: FEEDBACK_SURVEY_VERSION,
    install_nonce: nonces.install,
    session_nonce: nonces.session,
    seed: { title: set.seed.title, artist: set.seed.artist },
    displayed_results: set.results.slice(0, 20).map((result, index) => ({
      position: index + 1,
      title: result.title,
      artist: result.artist,
    })),
    method: KNOWN_METHODS.has(set.method) ? set.method : "unknown",
    index_version: INDEX_VERSION.test(set.indexVersion) ? set.indexVersion : "unknown",
    api_version: HOSTED_API_VERSION,
    language_policy: FEEDBACK_LANGUAGE_POLICY,
    selection_policy: FEEDBACK_SELECTION_POLICY,
    source: FEEDBACK_SOURCE,
    selection,
    reasons: isGood ? [] : reasons.slice(0, 2),
    note: isGood ? "" : note.trim().slice(0, 280),
  };
}

export async function submitFeedback(
  set: RecommendationSet,
  selection: FeedbackSelection,
  reasons: FeedbackReason[],
  note: string
): Promise<boolean> {
  const payload = buildFeedbackPayload(set, selection, reasons, note, {
    install: await installNonce(),
    session: sessionNonce,
  });
  try {
    const response = await fetch(FEEDBACK_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return response.ok;
  } catch {
    return false;
  }
}

export const REASON_LABELS: { id: FeedbackReason; label: string }[] = [
  { id: "style", label: "Wrong style" },
  { id: "mood_energy", label: "Wrong mood" },
  { id: "tempo", label: "Wrong tempo" },
  { id: "vocals_language", label: "Vocals or language" },
  { id: "instruments_timbre", label: "Instruments or texture" },
];
