import Constants, { ExecutionEnvironment } from "expo-constants";
import { StatusBar } from "expo-status-bar";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  SafeAreaView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { ShareIntentProvider, useShareIntentContext } from "expo-share-intent";

import { ApiError, fetchRecommendations } from "./src/lib/api";
import { resolveSharedText } from "./src/lib/resolve";
import type {
  CatalogTrack,
  RecommendationSet,
  SharedTrack,
} from "./src/lib/types";
import { ChooseScreen } from "./src/components/ChooseScreen";
import { ResultsScreen } from "./src/components/ResultsScreen";
import { SearchScreen } from "./src/components/SearchScreen";
import { theme } from "./src/theme";

const SHARE_INTENT_DISABLED =
  Platform.OS === "web" ||
  Constants.executionEnvironment === ExecutionEnvironment.StoreClient;

type Screen =
  | { name: "search" }
  | { name: "choose"; shared: SharedTrack; matches: CatalogTrack[] }
  | { name: "results"; set: RecommendationSet; artworkUrl?: string };

function Soundalike() {
  const { hasShareIntent, shareIntent, resetShareIntent } = useShareIntentContext();
  const [screen, setScreen] = useState<Screen>({ name: "search" });
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const work = useRef<AbortController | null>(null);

  useEffect(() => () => work.current?.abort(), []);

  const recommend = useCallback(
    async (track: CatalogTrack, artworkUrl?: string) => {
      work.current?.abort();
      const next = new AbortController();
      work.current = next;
      setError(null);
      setBusy("Finding soundalikes");
      try {
        const set = await fetchRecommendations(track.title, track.artist, next.signal);
        if (!next.signal.aborted) setScreen({ name: "results", set, artworkUrl });
      } catch (caught) {
        if (next.signal.aborted) return;
        setError(
          caught instanceof ApiError
            ? caught.message
            : "Something went wrong reaching Soundalike."
        );
      } finally {
        if (!next.signal.aborted) setBusy(null);
      }
    },
    []
  );

  const handleShared = useCallback(
    async (text: string) => {
      work.current?.abort();
      const next = new AbortController();
      work.current = next;
      setError(null);
      setBusy("Reading that track");
      try {
        const resolution = await resolveSharedText(text, next.signal);
        if (next.signal.aborted) return;
        if (resolution.kind === "resolved") {
          setBusy(null);
          await recommend(resolution.track, resolution.shared.artworkUrl);
          return;
        }
        if (resolution.kind === "choose") {
          setScreen({
            name: "choose",
            shared: resolution.shared,
            matches: resolution.matches,
          });
          return;
        }
        setScreen({ name: "search" });
        setError(
          resolution.kind === "missing"
            ? `"${resolution.shared.title ?? "That song"}" is not in the Soundalike library yet.`
            : "That did not look like a Spotify song link."
        );
      } catch {
        if (!next.signal.aborted) setError("Could not read that shared link.");
      } finally {
        if (!next.signal.aborted) setBusy(null);
      }
    },
    [recommend]
  );

  useEffect(() => {
    if (!hasShareIntent) return;
    const text = shareIntent?.webUrl || shareIntent?.text || "";
    resetShareIntent();
    if (text) void handleShared(text);
  }, [hasShareIntent, shareIntent, resetShareIntent, handleShared]);

  const goBack = useCallback(() => {
    work.current?.abort();
    setBusy(null);
    setError(null);
    setScreen({ name: "search" });
  }, []);

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar style="light" />
      <KeyboardAvoidingView
        style={styles.flex}
        behavior={Platform.OS === "ios" ? "padding" : undefined}
      >
        {error ? (
          <Pressable
            style={styles.error}
            onPress={() => setError(null)}
            accessibilityRole="button"
            accessibilityLabel={`${error}. Tap to dismiss.`}
          >
            <Text style={styles.errorText}>{error}</Text>
          </Pressable>
        ) : null}

        {screen.name === "search" ? (
          <SearchScreen onPick={(track) => void recommend(track)} />
        ) : null}

        {screen.name === "choose" ? (
          <ChooseScreen
            shared={screen.shared}
            matches={screen.matches}
            onPick={(track) => void recommend(track, screen.shared.artworkUrl)}
            onCancel={goBack}
          />
        ) : null}

        {screen.name === "results" ? (
          <ResultsScreen
            set={screen.set}
            seedArtworkUrl={screen.artworkUrl}
            onBack={goBack}
          />
        ) : null}

        {busy ? (
          <View style={styles.busy} pointerEvents="none">
            <ActivityIndicator color={theme.accent} />
            <Text style={styles.busyText}>{busy}</Text>
          </View>
        ) : null}
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

export default function App() {
  return (
    <ShareIntentProvider options={{ disabled: SHARE_INTENT_DISABLED }}>
      <Soundalike />
    </ShareIntentProvider>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: theme.bg },
  flex: { flex: 1 },
  error: {
    marginHorizontal: theme.space,
    marginTop: 8,
    padding: 12,
    borderRadius: theme.radius,
    backgroundColor: theme.surface,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: theme.border,
  },
  errorText: { color: theme.danger, fontSize: 13, lineHeight: 19 },
  busy: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(11,11,13,0.82)",
    gap: 12,
  },
  busyText: { color: theme.muted, fontSize: 14 },
});
