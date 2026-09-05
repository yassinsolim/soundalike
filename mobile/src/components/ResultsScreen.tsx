import { Image } from "expo-image";
import { useEffect, useRef, useState } from "react";
import { FlatList, Pressable, StyleSheet, Text, View } from "react-native";

import { fetchCoverByName, fetchCovers } from "../lib/artwork";
import { openInSpotify } from "../lib/open";
import type { RecommendationSet } from "../lib/types";
import { theme } from "../theme";
import { FeedbackBar } from "./FeedbackBar";
import { ResultRow } from "./ResultRow";

type Props = {
  set: RecommendationSet;
  seedArtworkUrl?: string;
  onBack: () => void;
};

export function ResultsScreen({ set, seedArtworkUrl, onBack }: Props) {
  const [covers, setCovers] = useState<Record<number, string | null>>({});
  const [seedCover, setSeedCover] = useState<string | undefined>(seedArtworkUrl);
  const list = useRef<FlatList<RecommendationSet["results"][number]>>(null);
  const scrollToBottomNext = useRef(false);

  useEffect(() => {
    let active = true;
    setCovers({});
    setSeedCover(seedArtworkUrl);
    const ids = set.results
      .map((result) => result.deezerId)
      .filter((id): id is number => typeof id === "number");
    void fetchCovers(ids, (id, url) => {
      if (active) setCovers((current) => ({ ...current, [id]: url }));
    });
    if (!seedArtworkUrl && set.seed.title && set.seed.artist) {
      void fetchCoverByName(set.seed.title, set.seed.artist).then((url) => {
        if (active && url) setSeedCover(url);
      });
    }
    return () => {
      active = false;
    };
  }, [set, seedArtworkUrl]);

  const vibe = [set.vibe.tempo, set.vibe.tone, set.vibe.dynamics]
    .filter(Boolean)
    .join("  ·  ");

  return (
    <View style={styles.screen}>
      <View style={styles.top}>
        <Pressable
          onPress={onBack}
          style={styles.back}
          accessibilityRole="button"
          accessibilityLabel="Back to search"
          hitSlop={12}
        >
          <Text style={styles.backText}>Back</Text>
        </Pressable>
        <View style={styles.seed}>
          {seedCover ? (
            <Image
              source={{ uri: seedCover }}
              style={styles.seedArt}
              contentFit="cover"
              transition={180}
            />
          ) : (
            <View style={[styles.seedArt, styles.seedArtPlaceholder]} />
          )}
          <View style={styles.seedLabels}>
            <Text style={styles.seedKicker}>Sounds like</Text>
            <Text style={styles.seedTitle} numberOfLines={2}>
              {set.seed.title}
            </Text>
            <Text style={styles.seedArtist} numberOfLines={1}>
              {set.seed.artist}
            </Text>
            {vibe ? (
              <Text style={styles.vibe} numberOfLines={1}>
                {vibe}
              </Text>
            ) : null}
          </View>
        </View>
      </View>

      <FlatList
        ref={list}
        data={set.results}
        keyExtractor={(item) => `${item.position}`}
        contentContainerStyle={styles.listContent}
        ListHeaderComponent={
          <Text style={styles.tapHint}>Tap any track to open it in Spotify.</Text>
        }
        renderItem={({ item }) => (
          <ResultRow
            result={item}
            coverUrl={item.deezerId ? covers[item.deezerId] : null}
            onPress={() => void openInSpotify(item.title, item.artist)}
          />
        )}
        onContentSizeChange={() => {
          if (!scrollToBottomNext.current) return;
          scrollToBottomNext.current = false;
          list.current?.scrollToEnd({ animated: true });
        }}
        ListFooterComponent={
          <View>
            <FeedbackBar
              set={set}
              onExpand={() => {
                scrollToBottomNext.current = true;
              }}
            />
            <Text style={styles.credit}>
              Track names and cover art come from Spotify and Deezer.
            </Text>
          </View>
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1 },
  top: {
    paddingBottom: 12,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: theme.border,
  },
  back: { paddingHorizontal: theme.space, paddingVertical: 10 },
  backText: { color: theme.accent, fontSize: 15, fontWeight: "600" },
  seed: {
    flexDirection: "row",
    gap: 14,
    paddingHorizontal: theme.space,
    paddingTop: 2,
  },
  seedArt: { width: 64, height: 64, borderRadius: 8 },
  seedArtPlaceholder: { backgroundColor: theme.surfaceHigh },
  seedLabels: { flex: 1, minWidth: 0, justifyContent: "center" },
  seedKicker: {
    color: theme.faint,
    fontSize: 11,
    textTransform: "uppercase",
    letterSpacing: 1,
  },
  seedTitle: { color: theme.text, fontSize: 19, fontWeight: "700", marginTop: 3 },
  seedArtist: { color: theme.muted, fontSize: 14, marginTop: 2 },
  vibe: { color: theme.faint, fontSize: 12, marginTop: 5 },
  listContent: { paddingBottom: 12 },
  tapHint: {
    color: theme.faint,
    fontSize: 12,
    paddingHorizontal: theme.space,
    paddingTop: 12,
    paddingBottom: 6,
  },
  credit: {
    color: theme.faint,
    fontSize: 11,
    textAlign: "center",
    paddingHorizontal: theme.space,
    paddingBottom: 28,
  },
});
