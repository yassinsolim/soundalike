import { Image } from "expo-image";
import { Pressable, StyleSheet, Text, View } from "react-native";

import type { CatalogTrack, SharedTrack } from "../lib/types";
import { theme } from "../theme";

type Props = {
  shared: SharedTrack;
  matches: CatalogTrack[];
  onPick: (track: CatalogTrack) => void;
  onCancel: () => void;
};

export function ChooseScreen({ shared, matches, onPick, onCancel }: Props) {
  return (
    <View style={styles.wrap}>
      <Pressable
        onPress={onCancel}
        style={styles.back}
        accessibilityRole="button"
        accessibilityLabel="Back to search"
        hitSlop={12}
      >
        <Text style={styles.backText}>Back</Text>
      </Pressable>

      <View style={styles.shared}>
        {shared.artworkUrl ? (
          <Image
            source={{ uri: shared.artworkUrl }}
            style={styles.art}
            contentFit="cover"
            transition={180}
          />
        ) : (
          <View style={[styles.art, styles.artPlaceholder]} />
        )}
        <View style={styles.labels}>
          <Text style={styles.kicker}>You shared</Text>
          <Text style={styles.title} numberOfLines={2}>
            {shared.title ?? "That track"}
          </Text>
          {shared.artist ? (
            <Text style={styles.artist} numberOfLines={1}>
              {shared.artist}
            </Text>
          ) : null}
        </View>
      </View>

      <Text style={styles.prompt}>Which one is it?</Text>

      {matches.map((track, index) => (
        <Pressable
          key={`${track.row}-${index}`}
          style={({ pressed }) => [styles.hit, pressed && styles.hitPressed]}
          onPress={() => onPick(track)}
          accessibilityRole="button"
          accessibilityLabel={`${track.title} by ${track.artist}`}
        >
          <Text style={styles.hitTitle} numberOfLines={1}>
            {track.title}
          </Text>
          <Text style={styles.hitArtist} numberOfLines={1}>
            {track.artist}
          </Text>
        </Pressable>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { flex: 1 },
  back: { paddingHorizontal: theme.space, paddingVertical: 10 },
  backText: { color: theme.accent, fontSize: 15, fontWeight: "600" },
  shared: {
    flexDirection: "row",
    gap: 14,
    paddingHorizontal: theme.space,
    paddingTop: 4,
    paddingBottom: 20,
  },
  art: { width: 72, height: 72, borderRadius: 8 },
  artPlaceholder: { backgroundColor: theme.surfaceHigh },
  labels: { flex: 1, minWidth: 0, justifyContent: "center" },
  kicker: {
    color: theme.faint,
    fontSize: 11,
    textTransform: "uppercase",
    letterSpacing: 1,
  },
  title: { color: theme.text, fontSize: 18, fontWeight: "700", marginTop: 3 },
  artist: { color: theme.muted, fontSize: 14, marginTop: 2 },
  prompt: {
    color: theme.text,
    fontSize: 15,
    fontWeight: "600",
    paddingHorizontal: theme.space,
    paddingBottom: 8,
  },
  hit: {
    paddingHorizontal: theme.space,
    paddingVertical: 12,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: theme.border,
  },
  hitPressed: { backgroundColor: theme.surfaceHigh },
  hitTitle: { color: theme.text, fontSize: 15, fontWeight: "600" },
  hitArtist: { color: theme.muted, fontSize: 13, marginTop: 2 },
});
