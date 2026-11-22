# Soundalike for iPhone

Native Swift/SwiftUI app (iOS 26+) with a Share Extension and Control Center controls.
It replaces the Expo app in `../mobile` and keeps its bundle id (`app.yassin.soundalike`),
App Store Connect record and app group.

| Target | Bundle id | What it does |
|---|---|---|
| `Soundalike` | `app.yassin.soundalike` | Search, recognise a song (ShazamKit), current Apple Music song, results with previews, Apple Music Play Next / Play all / Save playlist, App Intents and Siri phrases |
| `ShareExtension` | `app.yassin.soundalike.share` | "Soundalike" in the share sheet. Takes a Spotify or Apple Music song link and shows results over the music app |
| `Controls` | `app.yassin.soundalike.controls` | The *Soundalike* control for Control Center, the Lock Screen and the Action Button |
| `SoundalikeKit` (package) | – | Link parsing, link-to-song lookup, API client, Deezer enrichment, open-in-app links. Unit tested on macOS |
| `SoundalikeUI` (package) | – | Results list, preview player, shared settings, on-device event log |

## How a request flows

1. The input is one of:
   - a shared link or a search pick;
   - the song being played, read without the microphone by `App/NowPlaying.swift`:
     - the Music app through MusicKit, which needs no subscription;
     - Spotify, once the user taps **Connect Spotify** (one Agree tap in the Spotify app, no password). It tries the Web API first. When that fails, because Spotify's development mode only serves 5 allow-listed accounts, it asks the Spotify app through the App Remote SDK (`App/SpotifyRemote.swift`), which worked for a non-allow-listed account in 2–150 ms;
   - a ShazamKit match, used only when nothing can be read.
2. `SoundalikeClient` calls `GET /api/mobile/similar` first and falls back to `/api/anysong/similar`
   if that route fails or is missing.
   - Spotify links are named from the public embed page (`open.spotify.com/embed/track/<id>`),
     without sign-in.
   - Apple Music links are named from the iTunes lookup API.
   - Results are filled in from Deezer: cover, ISRC and a 30-second preview.
3. "Open" sends the user back to the app they came from:
   - Spotify: `https://open.spotify.com/search/isrc:<ISRC>`, which returns exactly that recording.
   - Apple Music: a MusicKit catalog match by ISRC, or the iTunes Search API when MusicKit isn't authorised.

**Control Center, Action Button, Siri and Back Tap** all run `FindSoundalikesIntent`. It opens the app straight onto soundalikes for the playing song (`soundalike://auto`); the app listens only when nothing can be read.

Not possible on iOS 26.6, all tried on the device:
- **A result card over the current app.** Controls never show snippets or dialogs: the intent returned a snippet every time and nothing appeared.
- **A Live Activity.** Apple caps it at 160 points tall and collapses it after a few seconds, which was too small to use.

Queueing and playing the full song in Spotify use the Web API, so for now they only work for allow-listed accounts. App Remote could do both for everyone once results carry Spotify track ids.

Links and in-app routes go through one URL scheme handled in `App/SoundalikeApp.swift`:

| URL | Action |
|---|---|
| `soundalike://auto` | Read the playing song, or listen when it can't be read |
| `soundalike://find?url=<link>` | Find soundalikes for a shared link |
| `soundalike://now-playing` | Use the song currently playing in Music |
| `soundalike://listen` | Recognise the song playing nearby |
| `soundalike://search?q=<text>` | Open search with this text |

`https://soundalike.yassin.app/find?url=…` opens the app once the site serves an
`apple-app-site-association` file for `FJH4UWVC4K.app.yassin.soundalike` with the `/find*` path.

## History

Every song that returns results is saved in `Library/history.json` in the app group (`SoundalikeUI/History.swift`), newest first. Each song is kept once and the list is capped at 200. The share extension writes to the same file, so its lookups appear in the app too. The home screen shows the last four under **Recent**; **See All** lists them by day, with swipe to delete and **Clear**. Nothing leaves the phone.

## Player bar

The bar at the bottom of every screen (`App/PlayerBar.swift`, `App/PlayerController.swift`) is a remote for the song playing in Spotify or the Music app, like the one in Control Center. It shows the cover, title, previous / play-pause / next and a scrubber; tapping the song finds its soundalikes. Soundalike plays nothing through it.

- **Music app:** MusicKit's `SystemMusicPlayer`, live.
- **Spotify:** App Remote when the Spotify app accepts the connection. Otherwise the Web API: it polls the player every 4 s and sends commands with the device id, because a backgrounded Spotify app doesn't count as the active device. The Web API part works only for allow-listed accounts and Premium.
- When both apps have a song, it shows the one that is playing; when neither plays, it keeps the last one shown.

Auto-detect runs at launch and after the app comes back from the background. Pulling down Control Center or a notification only makes the app inactive, so they don't re-run it.

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
