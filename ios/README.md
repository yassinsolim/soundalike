# Soundalike for iPhone

Native Swift/SwiftUI app (iOS 18+) with a Share Extension and Control Center controls.
It replaces the Expo app in `../mobile` and keeps its bundle id (`app.yassin.soundalike`),
App Store Connect record and app group.

| Target | Bundle id | What it does |
|---|---|---|
| `Soundalike` | `app.yassin.soundalike` | Search, recognise a song (ShazamKit), current Apple Music song, results with previews, Apple Music Play Next / Play all / Save playlist, App Intents and Siri phrases |
| `ShareExtension` | `app.yassin.soundalike.share` | "Soundalike" in the share sheet. Takes a Spotify or Apple Music song link and shows results over the music app |
| `Controls` | `app.yassin.soundalike.controls` | Control Center, Lock Screen and Action Button controls: Recognise Song, Soundalikes for Now Playing |
| `SoundalikeKit` (package) | – | Link parsing, link-to-song lookup, API client, Deezer enrichment, open-in-app links. Unit tested on macOS |
| `SoundalikeUI` (package) | – | Results list, preview player, shared settings, on-device event log |

## How a request flows

1. The input is a shared link, a search pick, the current Music song or a ShazamKit match.
2. `SoundalikeClient` calls `GET /api/mobile/similar` first. If that route fails or is missing,
   it falls back to `/api/anysong/similar`, which today it always does because the route returns 404.
   - Spotify links are named from the public embed page (`open.spotify.com/embed/track/<id>`),
     without sign-in.
   - Apple Music links are named from the iTunes lookup API.
   - Results are filled in from Deezer: cover, ISRC and a 30-second preview.
3. "Open" sends the user back to the app they came from:
   - Spotify: `https://open.spotify.com/search/isrc:<ISRC>`, which returns exactly that recording.
   - Apple Music: a MusicKit catalog match by ISRC, or the iTunes Search API when MusicKit isn't authorised.

Every entry point (intents, controls, Siri, links) goes through one URL scheme handled in `App/SoundalikeApp.swift`:

| URL | Action |
|---|---|
| `soundalike://find?url=<link>` | Find soundalikes for a shared link |
| `soundalike://now-playing` | Use the song currently playing in Music |
| `soundalike://listen` | Recognise the song playing nearby |
| `soundalike://search?q=<text>` | Open search with this text |

`https://soundalike.yassin.app/find?url=…` opens the app once the site serves an
`apple-app-site-association` file for `FJH4UWVC4K.app.yassin.soundalike` with the `/find*` path.

## Build

```sh
brew install xcodegen
cd ios
xcodegen generate                               # Soundalike.xcodeproj is generated, edit project.yml
(cd SoundalikeKit && swift test)                # parser and client tests
xcodebuild -project Soundalike.xcodeproj -scheme Soundalike \
  -destination 'id=<device udid>' -allowProvisioningUpdates build
```

Signing is automatic for team `FJH4UWVC4K`. The app ID needs the **MusicKit** and **ShazamKit** App Services
enabled in the developer portal (enabled 2026-11-20).

TestFlight upload:

```sh
xcodebuild archive -project Soundalike.xcodeproj -scheme Soundalike -destination 'generic/platform=iOS' \
  -archivePath /tmp/Soundalike.xcarchive -allowProvisioningUpdates
xcodebuild -exportArchive -archivePath /tmp/Soundalike.xcarchive -exportPath /tmp/export \
  -exportOptionsPlist ExportOptions.plist -allowProvisioningUpdates
```

## Diagnostics

The app and extension append one JSON line per notable event: share payload, request timing, open attempts,
recognition time and Apple Music actions. They write to `Library/events.jsonl` in the app group container,
nothing leaves the phone, and the file is capped at 512 KB. To read it from a connected iPhone:

```sh
xcrun devicectl device copy from --device <udid> --domain-type appGroupDataContainer \
  --domain-identifier group.app.yassin.soundalike --source Library/events.jsonl --destination events.jsonl
```

## Shortcut (no app needed)

`Shortcut/build_shortcut.py` builds "Find Soundalikes", a share-sheet Shortcut that opens
`https://soundalike.yassin.app/find?url=<link>`. When run with no input, it reads the clipboard.
Sign it with `shortcuts sign --mode anyone`.
