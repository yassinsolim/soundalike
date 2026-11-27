# Soundalike for iPhone

Native Swift/SwiftUI app (iOS 26+) with a Share Extension and Control Center controls.
It replaces the Expo app in `../mobile` and keeps its bundle id (`app.yassin.soundalike`),
App Store Connect record and app group.

| Target | Bundle id | What it does |
|---|---|---|
| `Soundalike` | `app.yassin.soundalike` | Search, recognise a song (ShazamKit), current Apple Music song, results with 30-second Deezer previews and Match a part, Apple Music Play Next / Play all / Save playlist, App Intents and Siri phrases |
| `ShareExtension` | `app.yassin.soundalike.share` | "Soundalike" in the share sheet. Takes a Spotify or Apple Music song link and shows results over the music app |
| `Controls` | `app.yassin.soundalike.controls` | The *Soundalike* control for Control Center, the Lock Screen and the Action Button |
| `SoundalikeKit` (package) | – | Link parsing, link-to-song lookup, API client, Deezer enrichment, open-in-app links, WAV encoding of clips. Unit tested on macOS |
| `SoundalikeUI` (package) | – | Results list, preview player, Match a part, shared settings, on-device event log |

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
   - Results are filled in from Deezer: cover and ISRC.
   - Previews are Deezer's 30-second MP3s (played with Deezer's permission), from each result's `preview_url` in the
     mobile answer, streamed by `SoundalikeUI/PreviewPlayer.swift` and never saved. Their addresses expire about
     15 minutes after the answer, so on a playback error the player gets a new one from
     `GET /api/preview?id=<deezer_id>` (`{"ok": true, "preview": "<mp3>", "cover": "<jpg>"}`) and tries once more.
     A preview pauses Spotify or Music and, when it ends or stops, releases the audio session with
     `notifyOthersOnDeactivation` so they can resume. When the app can't play the full song (Spotify not connected
     or not on the allow list, no Apple Music subscription), tapping a cover plays or stops the preview with a
     progress ring, and tapping the song opens it in the music app. When it can, tapping a result plays the full
     song in the music app as before, and the context menu offers **Play 30-second preview**. The share extension
     plays previews the same way.
3. "Open" sends the user back to the app they came from:
   - Spotify: `https://open.spotify.com/search/isrc:<ISRC>`, which returns exactly that recording.
   - Apple Music: a MusicKit catalog match by ISRC, or the iTunes Search API when MusicKit isn't authorised.

**Control Center, Action Button, Siri and Back Tap** all run `FindSoundalikesIntent`. It opens the app straight onto soundalikes for the playing song (`soundalike://auto`); the app listens only when nothing can be read.

Not possible on iOS 26.6, all tried on the device:
- **A result card over the current app.** Controls never show snippets or dialogs: the intent returned a snippet every time and nothing appeared.
- **A Live Activity.** Apple caps it at 160 points tall and collapses it after a few seconds, which was too small to use.

Queueing and playing the full song in Spotify use the Web API, so they only work for allow-listed accounts. Spotify answers everyone else with 403 "User not registered in the Developer Dashboard" (or "not approved"); the app remembers that (`SpotifyAccount.webAPIRefused`, checked right after Connect Spotify), and tapping their results opens the song in Spotify instead. Reading and controlling Spotify still work for them through App Remote. App Remote could do both for everyone once results carry Spotify track ids.

Links and in-app routes go through one URL scheme handled in `App/SoundalikeApp.swift`:

| URL | Action |
|---|---|
| `soundalike://auto` | Read the playing song, or listen when it can't be read |
| `soundalike://find?url=<link>` | Find soundalikes for a shared link |
| `soundalike://now-playing` | Use the song currently playing in Music |
| `soundalike://listen` | Recognise the song playing nearby |
| `soundalike://search?q=<text>` | Open search with this text |

`https://soundalike.yassin.app/find?url=…` opens the app once the site serves
`/.well-known/apple-app-site-association` for `FJH4UWVC4K.app.yassin.soundalike` with the `/find*` path
(`webapp/well-known/apple-app-site-association.json`, routed in `webapp/site.json`).

## Match a part

"Match a part" (beta) sits under the song on the results screen, in the app and in the share extension, and only
when the answer on screen has `query.parts` ([docs/ANYSONG.md › Match a part in the clients](../docs/ANYSONG.md#match-a-part-in-the-clients)),
so it costs no request of its own. It opens a sheet (`SoundalikeUI/MatchAPart.swift`):

- **The 30-second preview.** A bar of the preview with the part on it: drag the part, drag an edge, tap elsewhere to
  move it there, or use −5 s / +5 s and the 10 / 15 / 20 s buttons (clamped to `parts.min_s`–`max_s`); it stays on a
  5-second grid. Times are song times when the answer has `song_offset_s`, else preview times; the app doesn't
  know the song's length, so it never assumes 0:48. **Play this part** plays just the span from the seed's preview
  (`/api/preview`), and **Apply** reloads the lists with `part=<start>-<end>`. A chip ("Matching 0:58–1:13 of the
  song", or "of the preview") with **Whole preview** leads back. An answer to `part=` without `query.part` is for
  the whole preview (the server ignored the part), so no chip shows; a 400 or 404 to `part=` is asked once more
  without it (`SoundalikeClient.mobile`). Each new song starts from its whole preview.
- **Any part of the song (app only).** Only when the answer also has `query.parts.excerpt` (an engine with
  `--excerpts`), the sheet gets two more sources. Both send a WAV to `POST /api/anysong/excerpt?id=<the seed's
  Deezer id>` (`SoundalikeClient.similar(toClip:seed:)`), and the chip reads "Matching your clip (0:00–0:15 of your
  recording)" or "… of your file":
  - **Listen**: the listener plays the song out loud in their music app at the part they want; the microphone
    records 10, 15 or 20 seconds (within `excerpt.min_s`–`max_s` and `max_bytes`) with a countdown and a level
    meter, and can cancel. The audio session mixes with the music app (`.playAndRecord`, `.mixWithOthers`,
    `.measurement`), so the music keeps playing. It refuses before recording when the output is headphones,
    Bluetooth, AirPlay or a car (the microphone can't hear those), and after recording when it heard almost nothing.
  - **File**: an audio file picked with the document picker, decoded on the phone with `AVAudioFile`; a bar of
    the whole file with the same span control (on a 1-second grid) and **Play this part**. Apply decodes just the
    span.
  - The clip is mixed to one channel at 22,050 Hz, brought to full scale and encoded as 16-bit PCM WAV
    (`SoundalikeKit/AudioClip.swift`; 20 s is 882 KB). 429 and 503 are shown with their `Retry-After`, and 404,
    408, 413, 415 and 422 with what they mean. The clip lives in memory only and is sent once.
  - The share extension offers the preview only: it has no microphone permission and little memory.
- Lists for a part or a clip replace the list on screen; history keeps the song alone, never a part or a clip.
  The event log records each request (`part`: kind, span or clip length and bytes, route, ms, ok or error) and
  never any audio.

## History

Every song that returns results is saved in `Library/history.json` in the app group (`SoundalikeUI/History.swift`), newest first. Each song is kept once and the list is capped at 200. The share extension writes to the same file, so its lookups appear in the app too. The home screen shows the last four under **Recent**; **See All** lists them by day, with swipe to delete and **Clear**. Nothing leaves the phone.

## Player bar

The bar at the bottom of every screen (`App/PlayerBar.swift`, `App/PlayerController.swift`) is a remote for the song playing in Spotify or the Music app, like the one in Control Center. It shows the cover, title, previous / play-pause / next and a scrubber; tapping the song finds its soundalikes. Soundalike plays nothing through it.

- **Music app:** MusicKit's `SystemMusicPlayer`, live.
- **Spotify:** App Remote when the Spotify app accepts the connection. Otherwise the Web API: it polls the player every 4 s and sends commands with the device id, because a backgrounded Spotify app doesn't count as the active device. The Web API part works only for allow-listed accounts and Premium.
- When both apps have a song, it shows the one that is playing; when neither plays, the one that changed most recently.
- **A played result brings its whole list.** Tapping a result hands the music app the whole soundalike list, starting at that song: the Web API's `uris` + `offset` for Spotify (`SpotifyAccount.play(trackIDs:startingAt:)`), a `SystemMusicPlayer` queue for Music (`AppleMusicService.play(_:in:)`). The list becomes what the app is playing, so its own skips (lock screen, Control Center, headphones, CarPlay, this bar) walk it, and when the user picks something else in Spotify or Music it replaces the list; nothing is put in the user's queue to linger. The list's Spotify ids or Music songs are looked up side by side as soon as results load, so a tap starts at once. The bar shows the tapped song straight away and ignores reports of the previous song for up to 4 s, until the app reports the new one. After 3 seconds Previous restarts the song, like Control Center.
- **The playing result is green.** The results list marks the song in the bar like Spotify does: a green title and a moving waveform on the cover (still while paused). It's the result just tapped, or else the first result with the same title and artist (`LiveSong` in `SoundalikeUI/ResultsView.swift`).
- **No trip to Spotify on the first tap.** Right after Connect Spotify, the Spotify app is running but isn't the "active device", and the Web API answers 404. Play and queue requests then go to the phone's Spotify by device id (`me/player/devices`, type `Smartphone`; never a computer). Spotify is opened only when it isn't running on the phone at all.
- **Show Spotify controls.** After a restart the Spotify app refuses App Remote connections until it handles an authorization. When App Remote refuses and the Web API can't see the player (accounts not on the allow list), the bar offers this row. It runs the Spotify app's authorization again: Spotify opens for a moment, the user confirms if asked, and Spotify returns to Soundalike, after which App Remote connects. Spotify's SDK call `authorizeAndPlayURI` reported "Spotify not installed" on iOS 26.6, so the app opens the authorize URL itself.

Auto-detect runs at launch and after the app comes back from the background. Pulling down Control Center or a notification only makes the app inactive, so they don't re-run it. When the app comes back with results open (from any route: auto-detect, Recent, search, the player bar) and a different song is playing, it shows that song's results; a song from the open list (one the user played from it) keeps the list (`ResultsOnScreen` in `App/HomeView.swift`). Each results screen gets its request's id, so the Control Center button or a link replacing open results always loads the new song; before, SwiftUI reused the old screen and its state, and the old list stayed.

## Which app results open in

Once Spotify is connected, results open in Spotify whatever app the song came from; an Apple Music subscriber without Spotify gets Apple Music. The Spotify / Apple Music switch shows only to people who can play in both apps or in neither (`ResultsModel.preferred`, `ResultsView(showsTargetPicker:)`). It's applied again whenever Spotify connects or disconnects, so results already on screen move to Spotify when Connect Spotify finishes late. Before, a song found in the Music app switched the results to Apple Music and saved that as the preferred app.

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
