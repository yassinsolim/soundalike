import { useCallback, useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  FlatList,
  Keyboard,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";

import { searchCatalog } from "../lib/api";
import { CATALOG_SIZE_LABEL } from "../lib/config";
import type { CatalogTrack } from "../lib/types";
import { theme } from "../theme";

type Props = {
  onPick: (track: CatalogTrack) => void;
};

export function SearchScreen({ onPick }: Props) {
  const [query, setQuery] = useState("");
  const [tracks, setTracks] = useState<CatalogTrack[]>([]);
  const [searching, setSearching] = useState(false);
  const controller = useRef<AbortController | null>(null);

  const run = useCallback(async (value: string) => {
    controller.current?.abort();
    const next = new AbortController();
    controller.current = next;
    if (value.trim().length < 2) {
      setTracks([]);
      setSearching(false);
      return;
    }
    setSearching(true);
    try {
      const hits = await searchCatalog(value, 15, next.signal);
      if (!next.signal.aborted) setTracks(hits);
    } catch {
      if (!next.signal.aborted) setTracks([]);
    } finally {
      if (!next.signal.aborted) setSearching(false);
    }
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => void run(query), 220);
    return () => clearTimeout(timer);
  }, [query, run]);

  useEffect(() => () => controller.current?.abort(), []);

  return (
    <View style={styles.wrap}>
      <View style={styles.header}>
        <Text style={styles.title}>Soundalike</Text>
        <Text style={styles.subtitle}>
          Search a song, or share one to this app straight from Spotify.
        </Text>
      </View>

      <TextInput
        style={styles.input}
        value={query}
        onChangeText={setQuery}
        placeholder="Song or artist"
        placeholderTextColor={theme.faint}
        autoCorrect={false}
        autoCapitalize="none"
        returnKeyType="search"
        clearButtonMode="while-editing"
        accessibilityLabel="Search the Soundalike library"
      />

      <FlatList
        data={tracks}
        keyExtractor={(item, index) => `${item.row}-${index}`}
        keyboardShouldPersistTaps="handled"
        onScrollBeginDrag={Keyboard.dismiss}
        contentContainerStyle={tracks.length ? undefined : styles.emptyWrap}
        ListEmptyComponent={
          searching ? (
            <ActivityIndicator color={theme.muted} style={styles.spinner} />
          ) : (
            <View style={styles.empty}>
              <Text style={styles.emptyTitle}>
                {query.trim().length >= 2
                  ? "Nothing matched that."
                  : "Find songs that sound alike"}
              </Text>
              <Text style={styles.emptyBody}>
                {query.trim().length >= 2
                  ? `The library holds ${CATALOG_SIZE_LABEL} tracks, so newer or very niche songs may be missing.`
                  : `Type a song above, or open a track in Spotify, tap Share, and pick Soundalike. Matching runs against ${CATALOG_SIZE_LABEL} tracks.`}
              </Text>
            </View>
          )
        }
        renderItem={({ item }) => (
          <Pressable
            style={({ pressed }) => [styles.hit, pressed && styles.hitPressed]}
            onPress={() => {
              Keyboard.dismiss();
              onPick(item);
            }}
            accessibilityRole="button"
            accessibilityLabel={`${item.title} by ${item.artist}`}
          >
            <Text style={styles.hitTitle} numberOfLines={1}>
              {item.title}
            </Text>
            <Text style={styles.hitArtist} numberOfLines={1}>
              {item.artist}
            </Text>
          </Pressable>
        )}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { flex: 1 },
  header: { paddingHorizontal: theme.space, paddingTop: 8, paddingBottom: 16 },
  title: { color: theme.text, fontSize: 30, fontWeight: "800", letterSpacing: -0.5 },
  subtitle: { color: theme.muted, fontSize: 14, marginTop: 6, lineHeight: 20 },
  input: {
    marginHorizontal: theme.space,
    marginBottom: 12,
    paddingHorizontal: 14,
    paddingVertical: 12,
    borderRadius: theme.radius,
    backgroundColor: theme.surface,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: theme.border,
    color: theme.text,
    fontSize: 16,
  },
  spinner: { marginTop: 32 },
  emptyWrap: { flexGrow: 1 },
  empty: { paddingHorizontal: theme.space, paddingTop: 40, gap: 8 },
  emptyTitle: { color: theme.text, fontSize: 17, fontWeight: "600" },
  emptyBody: { color: theme.muted, fontSize: 14, lineHeight: 21 },
  hit: {
    paddingHorizontal: theme.space,
    paddingVertical: 11,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: theme.border,
  },
  hitPressed: { backgroundColor: theme.surfaceHigh },
  hitTitle: { color: theme.text, fontSize: 15, fontWeight: "600" },
  hitArtist: { color: theme.muted, fontSize: 13, marginTop: 2 },
});
