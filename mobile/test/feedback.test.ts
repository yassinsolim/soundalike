import { describe, expect, test } from "vitest";

import { buildFeedbackPayload, randomNonce } from "../src/lib/feedback";
import type { RecommendationSet } from "../src/lib/types";

const nonces = { install: "a".repeat(32), session: "b".repeat(32) };

function set(overrides: Partial<RecommendationSet> = {}): RecommendationSet {
  return {
    seed: { title: "Blinding Lights", artist: "The Weeknd" },
    vibe: {},
    results: [
      { position: 1, title: "Jealous", artist: "Nick Jonas" },
      { position: 2, title: "Thinkin", artist: "Mario" },
    ],
    method: "dual_sonic64_guardrail",
    indexVersion: "2026.07.11-dual-sonic64",
    librarySize: 272853,
    ...overrides,
  };
}

describe("randomNonce", () => {
  test("produces the 32 hex characters the server requires", () => {
    expect(randomNonce()).toMatch(/^[a-f0-9]{32}$/);
  });
});

describe("buildFeedbackPayload", () => {
  test("declares the mobile source with its own policies", () => {
    const payload = buildFeedbackPayload(set(), "mixed", ["tempo"], " slow ", nonces);
    expect(payload.source).toBe("mobile");
    expect(payload.language_policy).toBe("none");
    expect(payload.selection_policy).toBe("mobile-top-20-model-quality-v1");
    expect(payload.api_version).toBe("4");
    expect(payload.note).toBe("slow");
    expect(payload.displayed_results).toEqual([
      { position: 1, title: "Jealous", artist: "Nick Jonas" },
      { position: 2, title: "Thinkin", artist: "Mario" },
    ]);
  });

  test("clears reasons and notes for a good rating", () => {
    const payload = buildFeedbackPayload(set(), "good", ["tempo"], "ignored", nonces);
    expect(payload.reasons).toEqual([]);
    expect(payload.note).toBe("");
  });

  test("caps reasons at two and notes at 280 characters", () => {
    const payload = buildFeedbackPayload(
      set(),
      "off",
      ["style", "tempo", "mood_energy"],
      "x".repeat(400),
      nonces
    );
    expect(payload.reasons).toHaveLength(2);
    expect(payload.note).toHaveLength(280);
  });

  test("falls back to known enum values for unexpected server metadata", () => {
    const payload = buildFeedbackPayload(
      set({ method: "something_new", indexVersion: "not a version!" }),
      "mixed",
      [],
      "",
      nonces
    );
    expect(payload.method).toBe("unknown");
    expect(payload.index_version).toBe("unknown");
  });

  test("never sends more than the twenty displayed rows", () => {
    const many = Array.from({ length: 25 }, (_, index) => ({
      position: index + 1,
      title: `Track ${index}`,
      artist: "Someone",
    }));
    const payload = buildFeedbackPayload(set({ results: many }), "off", [], "", nonces);
    expect(payload.displayed_results).toHaveLength(20);
    expect(payload.displayed_results[19].position).toBe(20);
  });
});
