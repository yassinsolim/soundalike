import { Image } from "expo-image";
import { Pressable, StyleSheet, Text, View } from "react-native";

import { theme } from "../theme";
import type { Recommendation } from "../lib/types";

type Props = {
  result: Recommendation;
  coverUrl?: string | null;
  onPress: () => void;
};

export function ResultRow({ result, coverUrl, onPress }: Props) {
  return (
    <Pressable
      style={({ pressed }) => [styles.row, pressed && styles.rowPressed]}
      onPress={onPress}
      accessibilityRole="button"
      accessibilityLabel={`${result.title} by ${result.artist}. Open in Spotify.`}
    >
      <Text style={styles.rank}>{result.position}</Text>
      <View style={styles.artWrap}>
        {coverUrl ? (
          <Image
            source={{ uri: coverUrl }}
            style={styles.art}
            contentFit="cover"
            transition={160}
          />
        ) : (
          <View style={[styles.art, styles.artPlaceholder]} />
        )}
      </View>
      <View style={styles.labels}>
        <Text style={styles.title} numberOfLines={2}>
          {result.title}
        </Text>
        <Text style={styles.artist} numberOfLines={1}>
          {result.artist}
        </Text>
      </View>
      {result.bpm ? (
        <View style={styles.tempo}>
          <Text style={styles.bpm}>{result.bpm}</Text>
          <Text style={styles.bpmUnit}>BPM</Text>
        </View>
      ) : null}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 8,
    paddingHorizontal: theme.space,
    gap: 12,
  },
  rowPressed: { backgroundColor: theme.surfaceHigh },
  rank: {
    width: 22,
    textAlign: "right",
    color: theme.faint,
    fontVariant: ["tabular-nums"],
    fontSize: 13,
  },
  artWrap: { width: 48, height: 48, borderRadius: 6, overflow: "hidden" },
  art: { width: 48, height: 48 },
  artPlaceholder: { backgroundColor: theme.surfaceHigh },
  labels: { flex: 1, minWidth: 0 },
  title: { color: theme.text, fontSize: 15, fontWeight: "600", lineHeight: 20 },
  artist: { color: theme.muted, fontSize: 13, marginTop: 2 },
  tempo: { minWidth: 34, alignItems: "flex-end" },
  bpm: {
    color: theme.muted,
    fontSize: 13,
    fontVariant: ["tabular-nums"],
  },
  bpmUnit: {
    color: theme.faint,
    fontSize: 9,
    letterSpacing: 0.5,
    marginTop: 1,
  },
});
