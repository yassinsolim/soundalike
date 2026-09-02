import { Linking } from "react-native";

import { searchUris } from "./spotify";

/** Opens the Spotify app when it is installed, otherwise the web player. */
export async function openInSpotify(title: string, artist: string): Promise<void> {
  for (const uri of searchUris(title, artist)) {
    try {
      if (await Linking.canOpenURL(uri)) {
        await Linking.openURL(uri);
        return;
      }
    } catch {
      // Fall through to the next candidate.
    }
  }
  await Linking.openURL(searchUris(title, artist)[1]);
}
