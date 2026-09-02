import { useEffect, useState } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";

import { REASON_LABELS, submitFeedback } from "../lib/feedback";
import type {
  FeedbackReason,
  FeedbackSelection,
  RecommendationSet,
} from "../lib/types";
import { theme } from "../theme";

const CHOICES: { id: FeedbackSelection; label: string }[] = [
  { id: "good", label: "Good" },
  { id: "mixed", label: "Mixed" },
  { id: "off", label: "Off" },
];

export function FeedbackBar({ set }: { set: RecommendationSet }) {
  const [selection, setSelection] = useState<FeedbackSelection | null>(null);
  const [reasons, setReasons] = useState<FeedbackReason[]>([]);
  const [state, setState] = useState<"idle" | "sending" | "sent" | "failed">("idle");

  useEffect(() => {
    setSelection(null);
    setReasons([]);
    setState("idle");
  }, [set]);

  const toggleReason = (id: FeedbackReason) => {
    setReasons((current) => {
      if (current.includes(id)) return current.filter((value) => value !== id);
      return current.length >= 2 ? current : [...current, id];
    });
  };

  const send = async (choice: FeedbackSelection, chosenReasons: FeedbackReason[]) => {
    setState("sending");
    const ok = await submitFeedback(set, choice, chosenReasons, "");
    setState(ok ? "sent" : "failed");
  };

  if (state === "sent") {
    return (
      <View style={styles.wrap}>
        <Text style={styles.thanks}>Thanks, that helps.</Text>
      </View>
    );
  }

  return (
    <View style={styles.wrap}>
      <Text style={styles.prompt}>How close were these?</Text>
      <View style={styles.choices}>
        {CHOICES.map((choice) => {
          const active = selection === choice.id;
          return (
            <Pressable
              key={choice.id}
              style={[styles.chip, active && styles.chipActive]}
              accessibilityRole="button"
              accessibilityState={{ selected: active }}
              onPress={() => {
                setSelection(choice.id);
                setReasons([]);
                if (choice.id === "good") void send("good", []);
              }}
            >
              <Text style={[styles.chipText, active && styles.chipTextActive]}>
                {choice.label}
              </Text>
            </Pressable>
          );
        })}
      </View>

      {selection && selection !== "good" ? (
        <View style={styles.detail}>
          <Text style={styles.hint}>What was off? Pick up to two.</Text>
          <View style={styles.choices}>
            {REASON_LABELS.map((reason) => {
              const active = reasons.includes(reason.id);
              return (
                <Pressable
                  key={reason.id}
                  style={[styles.chip, active && styles.chipActive]}
                  accessibilityRole="button"
                  accessibilityState={{ selected: active }}
                  onPress={() => toggleReason(reason.id)}
                >
                  <Text style={[styles.chipText, active && styles.chipTextActive]}>
                    {reason.label}
                  </Text>
                </Pressable>
              );
            })}
          </View>
          <Pressable
            style={[styles.send, state === "sending" && styles.sendBusy]}
            accessibilityRole="button"
            disabled={state === "sending"}
            onPress={() => void send(selection, reasons)}
          >
            <Text style={styles.sendText}>
              {state === "sending" ? "Sending" : "Send feedback"}
            </Text>
          </Pressable>
        </View>
      ) : null}

      {state === "failed" ? (
        <Text style={styles.failed}>That did not send. Try again in a moment.</Text>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    paddingHorizontal: theme.space,
    paddingTop: 20,
    paddingBottom: 32,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: theme.border,
    marginTop: 16,
    gap: 12,
  },
  prompt: { color: theme.text, fontSize: 15, fontWeight: "600" },
  hint: { color: theme.muted, fontSize: 13 },
  choices: { flexDirection: "row", flexWrap: "wrap", gap: 8 },
  chip: {
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 999,
    backgroundColor: theme.surfaceHigh,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: theme.border,
  },
  chipActive: { backgroundColor: theme.accent, borderColor: theme.accent },
  chipText: { color: theme.text, fontSize: 13, fontWeight: "500" },
  chipTextActive: { color: theme.accentText },
  detail: { gap: 12 },
  send: {
    alignSelf: "flex-start",
    paddingVertical: 10,
    paddingHorizontal: 18,
    borderRadius: 999,
    backgroundColor: theme.accent,
  },
  sendBusy: { opacity: 0.6 },
  sendText: { color: theme.accentText, fontWeight: "700", fontSize: 14 },
  thanks: { color: theme.muted, fontSize: 14 },
  failed: { color: theme.danger, fontSize: 13 },
});
