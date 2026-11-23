# Soundalike for Android

Native Kotlin / Jetpack Compose app (Android 8.0+, `minSdk 26`, `targetSdk 36`). It replaces the Expo app in
`../mobile` and keeps its application id, `app.yassin.soundalike`. It mirrors the iPhone app in `../ios`: the same
request flow, the same fallbacks, the same look.

| Module | What it is |
|---|---|
| `core` | Pure Kotlin (JVM, no Android): link parsing, link-to-song lookup, API client, Deezer enrichment, open-in-app links. Unit tested. The counterpart of `ios/SoundalikeKit` |
| `app` | The Android app: results sheet, now-playing reader, tile, widget, notification, floating button, search, history, settings, guide |

## How a request flows

1. The input is one of:
   - a link shared from a music app (`ACTION_SEND` `text/plain`), a `soundalike://` link or
     `https://soundalike.yassin.app/find?url=…`;
   - the song a music app is playing, read from its media session (`NowPlaying.kt`), with no login;
   - a search pick or a pasted link inside the app.
2. `SoundalikeClient` calls `GET /api/mobile/similar` first, naming the song in exactly one way (`url`, `spotify_id`,
   `apple_id`, `isrc`, or `title` + `artist`). The route is live since 2026-11-22 and answers every request the
   app made on a real device. If it is missing or fails, the app falls back to `/api/anysong/similar`:
   - Spotify links are named from the public embed page (`open.spotify.com/embed/track/<id>`), without sign-in;
     `spotify.link` short links are expanded first.
   - Apple Music links are named from the iTunes lookup API.
   - YouTube Music and YouTube links are named from YouTube's oEmbed endpoint. The server does not take these
     links, so they are always named on the phone.
   - Results are filled in from Deezer: cover, ISRC and a 30-second preview.
   - A 429 is shown as "Too many requests" with the server's `Retry-After` and is not retried on the other route.
3. Tapping a result opens it in the app the request came from (the picker above the list switches app):
   - Spotify: `spotify:track:<id>` when the server gave the song's own page, else
     `https://open.spotify.com/search/isrc:<ISRC>` (lands on exactly that recording), else a title and artist search.
   - Apple Music: the server's link, else the best iTunes Search match, else a search link.
   - YouTube Music: `https://music.youtube.com/search?q=<title artist>`.
   - If the app is not installed, the same address opens in the browser, with one exception: Spotify's web
     player drops a search link on phones (it lands on an empty search page), so without the Spotify app a
     search link is not opened. A dialog offers "Get Spotify" (Play Store) or picking another app.

## Surfaces

| Surface | What it does | Taps from a playing song | Needs |
|---|---|---|---|
| **Share target** "Find soundalikes" | Results as a bottom sheet over the music app. Also registered as a Sharing Shortcut, so it sits in the direct-share row of Android's share sheet | Spotify: Share › More › Find soundalikes = 3 from Now Playing, 4 from a song's ⋮ menu. 1 tap once Android's share sheet is open | Nothing |
| **Quick Settings tile** "Soundalikes" | Opens the sheet for the playing song | Swipe down + 1 tap | Notification access; tile added once (on Android 13+ the guide offers it in a system dialog: 2 taps) |
| **Home-screen widget** | Shows the playing song; tap opens the sheet | 1 tap on the home screen | Notification access |
| **Notification while music plays** (off by default) | Quiet ongoing notification; tap opens the sheet | Swipe down + 1 tap | Notification access, notification permission |
| **App shortcuts** | Long-press the icon: Find soundalikes (playing song), Search a song | Long-press + 1 tap | Notification access for the first |
| **Floating button** (off by default) | Small draggable button over the music app while it plays; tap opens the sheet, hold hides it until the next song, it remembers where it was dragged | 1 tap, without leaving the music app | Notification access, "Display over other apps"; with "Usage access" it shows only while the playing app is in front |
| **The app** | Playing-now card, search (Deezer), pasted links, history, settings, guide | Open app + 1 tap | Notification access for the card |

Measured in Spotify 9.2.0 on a real device, from "listening" to the list and on to hearing a soundalike:

| Route | Taps to the list | Then to hear one |
|---|---|---|
| Floating button | 1 | +1 preview in the sheet, or +2 in Spotify (row, then the one search hit); +1 when the server sends the track's own link |
| Quick Settings tile | swipe + 1 | same |
| Share from Now Playing (share icon › More › Find soundalikes) | 3 | same |
| Share from a song's ⋮ menu (⋮ › Share › More › Find soundalikes) | 4 | same |

Recommended default: the first-run guide sets up **notification access and the tile** (three taps on Android 13+),
which gives swipe + tap from inside any music app with nothing on screen otherwise. **Share** needs no setup and is
the route for a song that is not the one playing. The **floating button** is the closest thing to a button inside
Spotify and is offered in Settings, not switched on by default: it needs "Display over other apps", Android shows
its own "displaying over other apps" notification, and without the extra "Usage access" it shows over every app
while music plays.

While the list is open and the player moves to another song, a line at the top offers "Show soundalikes" for the
new song.

### Match a part

The matches normally follow the song's whole 30-second preview. When the answer on screen says the server can
match a part of it (`query.parts`), a **Match a part** button (beta) sits under the song, in the sheet and in the
app; with a server that can't, nothing shows and nothing is asked.

- Tap it: a bar of the preview opens with the part on it, 10 to 20 seconds long on a 5-second grid like the
  website's. Drag the part to move it, drag near an edge to resize it, tap elsewhere on the bar to bring it there,
  or use −5 s / +5 s and the 10 s / 15 s / 20 s buttons, which are easier than dragging on a small screen.
- **Play this part** plays just that span of the song's own preview (fetched from Deezer by the id the server
  matched) with the same audio focus as the row previews, so the music app pauses and resumes; a line moves across
  the bar while it plays.
- **Apply** reloads the list for the part (`part=<start>-<end>`); the button becomes a chip, "Matching 0:58–1:08 of
  the song", with **Whole preview** next to it. Tapping the chip reopens the picker at that part.
- Times are song times when the server gives the preview's place in the song, or when the song is known to be 90
  seconds or longer (from the player or the search result; previews usually start 0:48 in, and the picker says it
  is an assumption). Otherwise they are preview times, "0:10–0:20 of the preview".
- The part belongs to the song on screen. It stays while the music app moves on (the "Now playing … Show
  soundalikes" line still offers the new song), through a rotation, and through a failed request (Try again asks
  for the same part). A new song, from that line, a share, the tile or a search, starts from its whole preview.
- A server that refuses `part=` (400 or 404) is asked once more without it, and an answer without `query.part` is
  shown as the whole preview. From the phone to the first list for a part: two taps (Match a part, Apply) for the
  default 0:10–0:20.

There is no waveform: drawing one means downloading and decoding the preview before the picker can open.

What is playing comes from two sources:
- `MediaSessionManager.getActiveSessions`, which Android allows for apps with notification access.
  `NowPlayingListener` is an empty `NotificationListenerService`; no notification is read. It gives title, artist
  and the player's package for any player. **Spotify puts the track URI in the session** (`METADATA_KEY_MEDIA_ID` =
  `spotify:track:<id>`, measured with Spotify 9.2.0, no setting and no Spotify API involved), so for Spotify the
  server gets the exact `spotify_id`. Spotify also publishes the cover's https address
  (`com.spotify.music.extra.ART_HTTPS_URI`, shown while the list loads) and marks adverts (`ADVERTISEMENT` = 1,
  treated as nothing playing).
- The `com.android.music.metadatachanged` / `metachanged` / `playstatechanged` broadcasts (extras `track`,
  `artist`, `playing`). The AOSP Music app and players built on it keep a media session but put no song in it;
  they name the song only in these broadcasts. Used when no session names a song, and only for a session that
  belongs to such a player. A title that is a file name loses its audio extension.
- Spotify's `com.spotify.music.metadatachanged` broadcast (extra `id` = track URI), which Spotify only sends when
  the user turns on **Device Broadcast Status** in Spotify's settings. Read as a sticky broadcast and through a
  runtime receiver. It is used when its track name matches the Spotify session's title.

Settings › "Show what my player reports" prints every text field of the active sessions, the last broadcasts, the
song the app would use, and which server route answered the last request and how fast. The same two lines go to
logcat under the tag `Soundalike` (`adb logcat -s Soundalike:I`), in release builds too.

The sheet closes when the user leaves it, so a stale list never comes back; it stays through a rotation, through
the screen turning off and on, and while the user is on the notification-access screen. Held sideways, the song
and the app picker share one row, and the sheet stops below the status bar.

What Spotify for Android does with each way of opening a result (measured, Premium account):

| Sent to `com.spotify.music` | Result |
|---|---|
| `spotify:track:<id>` (VIEW) | Opens the album at that track and **starts playing it** |
| `https://open.spotify.com/search/isrc%3A<ISRC>` (VIEW) | Search page with exactly that recording as the only song: one more tap plays it. Used when no track id is known |
| `spotify:search:isrc:<ISRC>` (VIEW) | Worse: searches the bare code as text and lists unrelated things |
| `android.media.action.MEDIA_PLAY_FROM_SEARCH` with title and artist | Opens a search page for the words; plays nothing |
| `playFromSearch` / `playFromUri` on Spotify's media session (through notification access) | Ignored |
| Connecting to Spotify's `MediaBrowserService` | Refused ("No root for client") |

So a tap plays at once only with a track id. The phone has no credential-free way to turn an ISRC into an id; the
server does it for the rows it can (`spotify_kind: "track"`, `spotify_id`). In the list those rows carry a play
mark and the others an open mark:

- **A row with a track id**: the tap starts the song in Spotify. Spotify comes to the front, and as soon as the
  player reports the new song the sheet comes back in front of it: the list stays, the row reads "Playing in
  Spotify", and the next tap switches song. On a real device (Spotify 9.2.0, 12 taps in a row) the song started
  0.6 to 0.9 s after the tap seven times and 1.4 to 9.7 s the other five; the sheet was back every time. When no
  song change is seen within 12 s (no notification access, the same song tapped again, or Spotify had to start
  first) the sheet closes and Spotify stays in front, where the song is. Coming back on a timer instead covered
  Spotify before it had read the link about one tap in ten, and the song did not start.
- **A row without one**: the tap opens the ISRC search in Spotify, one more tap there.

There is no login-free way to queue a song or play the whole list in Spotify.

**Spotify App Remote** (the SDK the iPhone app uses to play and queue without leaving) was tried on the device with
the site's client id (`6aae8635d21e471cabdeacd2f226640e`, redirect `soundalike://spotify-callback`): the connection
fails with `UserNotAuthorizedException` before any consent screen, because Spotify only talks to Android apps whose
package and signing fingerprint are registered for the client id. It is not in the app. To enable it, the owner of
the Spotify developer app adds under Android packages: package `app.yassin.soundalike` with SHA-1
`7B:31:86:2B:7E:60:E7:AC:BA:9F:B6:83:04:B9:6E:7D:F8:54:CB:7A` (release key) and, for debug builds from the
development PC, `70:73:78:31:1E:0D:04:9B:14:51:58:17:FC:2A:A1:CE:FC:08:76:DD`. The SDK is an Apache-2.0 AAR on
Spotify's GitHub (`spotify-app-remote-release-0.8.0.aar`, last release July 2023), used under Spotify's Developer
Terms; it would be fetched at build time, not committed. Whether Spotify's five-user development limit applies to
App Remote on Android is not known until a second account tries.

Not built:
- **AccessibilityService**: not used anywhere, by decision.
- **Bubbles API**: Android only shows bubbles for conversation notifications (a messaging-style notification tied
  to a person shortcut). A music shortcut is not a conversation, so the floating button uses a plain overlay window.
- **Assistant / App Actions**: a `shortcuts.xml` capability (for example `actions.intent.OPEN_APP_FEATURE`) only
  works for an app published on Play and reviewed for Assistant, and needs a Google account to try. Not added; the
  deep link it would call exists (`soundalike://auto`).

Links handled:

| Link | Action |
|---|---|
| `soundalike://auto`, `soundalike://now-playing` | Sheet for the playing song |
| `soundalike://find?url=<link>` | Sheet for a shared link |
| `soundalike://search?q=<text>` | The app's search with this text |
| `https://soundalike.yassin.app/find?url=…` | Sheet for the link. The site serves `/.well-known/assetlinks.json` with the release key's SHA-256 (`1E:CE:72:59:6B:BF:02:D1:61:FE:8E:6F:83:EA:31:70:F5:69:C4:65:A7:79:21:60:80:D3:45:BC:34:48:EF:3B`); with the release build Android 13 reports the domain `verified` and opens the app without asking. A debug build is signed with another key, so it is not expected to verify (not measured) |

## Privacy, release, licences

[`PRIVACY.md`](PRIVACY.md) is the privacy policy; [`RELEASE.md`](RELEASE.md) is how to cut a release and what Google
Play asks for. The app's dependencies are all Apache-2.0: Kotlin and kotlinx (coroutines, serialization), AndroidX
and Jetpack Compose, Coil with OkHttp and Okio, and the Gradle wrapper; the icons are Material Symbols paths
(Apache-2.0). No binaries other than the Gradle wrapper jar are in the tree.

## Build

Needs JDK 17 and the Android SDK with `platforms;android-37.2` and build-tools 36 (Compose 1.13 must compile
against API 37.1 or later). On the development PC everything lives under `D:\android`; `D:\android\SDK_READY.txt`
lists it.

```powershell
. D:\android\env.ps1          # JAVA_HOME, ANDROID_HOME, GRADLE_USER_HOME, PATH for this shell
cd android
.\gradlew.bat :core:test                 # unit tests, plain JVM
.\gradlew.bat :app:assembleDebug         # app/build/outputs/apk/debug/app-debug.apk
.\gradlew.bat :app:assembleRelease       # app/build/outputs/apk/release/app-release.apk (R8, 1.7 MB)
.\gradlew.bat :app:lintDebug
```

Release signing stays outside the repository. Set the Gradle property `soundalike.keystoreProperties` (for example
in `GRADLE_USER_HOME/gradle.properties`) to a properties file with `storeFile`, `storePassword`, `keyAlias` and
`keyPassword`; without it the release APK is unsigned. On the development PC: keystore
`D:\android\keys\soundalike-release.jks`, alias `soundalike`, passwords in
`D:\android\keys\soundalike-release.properties`. An install signed with another key (the old Expo build, or a debug
build) has to be uninstalled before this one installs.

Debug builds add three things: Settings › Debug build sets the engine address (from the emulator the PC is
`http://10.0.2.2:8792`; plain http is allowed in debug only), and two stand-ins for a music app:

```sh
# a media session, as a player publishes it (add --ez paused true, or --ez stop true to end it)
adb shell am start -n app.yassin.soundalike/.debug.FakePlayerActivity --es title "Du hast" --es artist "Rammstein" \
    [--es media_id spotify:track:5awDvzxWfd53SSrsRZ8pXO]
# Android's share sheet with a song link, as Spotify's Share > More opens it
adb shell am start -n app.yassin.soundalike/.debug.FakeShareActivity --es text "https://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b?si=x"
# a share straight to the sheet, and Spotify's broadcast
adb shell am start -a android.intent.action.SEND -t text/plain -n app.yassin.soundalike/.SheetActivity \
    --es android.intent.extra.TEXT "https://music.youtube.com/watch?v=fHI8X4OXluQ"
adb shell am broadcast -a com.spotify.music.metadatachanged --es id spotify:track:0VjIjW4GlUZAMYd2vXMi3b \
    --es track "Blinding Lights" --es artist "The Weeknd"
adb shell cmd notification allow_listener app.yassin.soundalike/app.yassin.soundalike.NowPlayingListener
```

`tools/fake_engine.py 429|503` is an engine that always fails, for the error states (port 8799).

## What was verified, and where

Run on 2026-11-22 in the emulator (Android 16, API 36, Google APIs x86_64), debug build against the local engine
unless noted:

| Verified in the emulator | Result |
|---|---|
| Share intents with Spotify, Apple Music and YouTube Music links | Sheet with seed, cover art, match %, list |
| Spotify album link | "That isn't a song link" |
| Android's share sheet (via `FakeShareActivity`) | "Find soundalikes" is in the direct-share row; one tap opens the sheet |
| Preview | Plays (MediaPlayer started, transient audio focus), stops when a result is opened |
| Opening a result | YouTube Music app receives the search link; with the official Spotify app installed, `com.spotify.music` receives the `open.spotify.com/search/isrc…` link (it then shows its login screen); without it Chrome does |
| Now playing from a media session (`FakePlayerActivity`) | No access, nothing playing and playing states; tile, widget (live title), notification, app shortcuts and floating button (drag, tap) all open the sheet |
| Spotify broadcast sent with `am broadcast` | Received by the runtime receiver on Android 16, shown in diagnostics |
| Rotation | List kept, no second request, landscape layout |
| Airplane mode, engine port closed, 429 with `Retry-After`, 503 | Offline message, offline message, "Try again in 42 seconds", "problem (503)"; Try again works |
| In-app search, results, history, settings, guide | Work |
| Release APK (minified, signed) against the production engine | Share, list, preview, app screens work |

Run on 2026-11-22 on a real device, a Retroid Pocket Flip 2 (Android 13, arm64, landscape handheld, Wi-Fi), against
the production engine:

| Verified on the Flip 2 | Result |
|---|---|
| Install, cold start | Debug: first start 729 ms. Release: 158 to 165 ms (`am start -W`, three cold starts) |
| Share intents (Spotify, Apple Music, YouTube Music links) | Sheet with results; all from `/api/mobile/similar`: 649 to 878 ms for known songs, 5.1 s for a song the engine had not seen, about 100 ms when repeated |
| Android 13 share sheet (debug `FakeShareActivity`) | "Find soundalikes" is in the direct-share row and Soundalike is first in the app row; one tap opens the sheet |
| Preview | Plays (MediaPlayer started, transient audio focus held), stops when a row is opened |
| Row tapped without Spotify | Was: Chrome on an empty Spotify search page. Now: "Spotify isn't installed" dialog; Get Spotify opens the Play Store page |
| Notification access | The real system screen and its Allow dialog walked once; back in the sheet the request is retried by itself |
| Playing song from a real player (AOSP Music playing a tagged silent file) | Its session carries no song; found through the `com.android.music` broadcasts after the fix. Tile (shows the title), notification (posted with the title), floating button (over the launcher and over Spotify) and the app's playing card all work |
| Screen off and on, force-stop of the app | The sheet survives screen off/on. After a force-stop Android restarts the listener by itself and the notification comes back |
| Official Spotify 9.2.0.2053 (arm64 splits copied from another of the owner's devices) | Installs and starts to its login screen. A tapped result is handed to `com.spotify.music` (`SpotifyMainActivity`), which shows the login screen. An idle, logged-out Spotify session is no longer mistaken for the player |
| `https://soundalike.yassin.app/find?url=…` with the release build | Domain `verified`; the link opens the sheet directly, 183 ms to the activity, list in 242 ms |
| In-app search and results | Work |

Not possible on that device: its launcher hosts no widgets and shows no app shortcuts (both are registered), the
screen stays in landscape (portrait was only run in the emulator), the notification was checked in
`dumpsys notification` but not tapped, and the device was not rebooted.

With Spotify logged in on the same device (2026-11-22, release build unless noted):

| Verified with a logged-in Spotify | Result |
|---|---|
| Playing song | Read from the session with its track URI; sent as `spotify_id`; list in 117 to 592 ms for songs the engine knew, about 4 s for new ones |
| Device Broadcast Status (off by default, turned on for the test, then off again) | `com.spotify.music.metadatachanged` is delivered and sticky on Android 13 with the track URI. Not needed, since the session already has it |
| Share from Now Playing | Share icon › More › Android's sheet with "Find soundalikes" first in the direct-share row. Spotify's own sheet lists only its friends, Copy link and More. Payload: `EXTRA_TEXT` = `https://open.spotify.com/track/<id>?si=…&utm_source=native-share-menu&sci=…`, referrer `android-app://com.spotify.music`, no subject. The sheet opens half height over Spotify (portrait there), in its own task |
| A song's ⋮ menu | Share is its first entry |
| Row tapped | Spotify shows the ISRC search with that one song; the sheet is gone |
| Preview | Spotify pauses for it and resumes afterwards |
| Tile, notification (follows the song, goes when paused), floating button, playing card | All work while Spotify plays |
| Floating button (debug build) | Hidden over Soundalike's own screens; one tap inside Spotify opens the sheet; drag position kept; hold hides it until the next song; with usage access it is hidden over other apps |
| Song change with the sheet open (debug build) | "Now playing: … Show soundalikes" appears; tapping it reloads |
| Add-tile dialog (debug build) | Guide button › system dialog › Add tile |

Tap = plays, release build against production (2026-11-22, Spotify 9.2.0 logged in): the live route marked 12 to
13 of 20 rows with a track id. Tile, then a row: the song plays and the list is back in front, 12 taps out of 12;
a row without an id opens the ISRC search. From Spotify's own share sheet (portrait there): share icon, More, Find
soundalikes, a row: the song started 0.8 s after the last tap. With Spotify swiped out of recents the song starts
only after Spotify has loaded (over ten seconds once), and the sheet closes instead of coming back. When Spotify
moves on by itself the green mark goes and the "Now playing" line appears. Listening to hearing a soundalike: two
taps with the floating button, swipe and two taps with the tile, four taps by sharing; about one second after the
last tap in the usual case.

Re-run on the release build the same day: the add-tile dialog from the guide, the playing card following the
player, the floating button (only over Spotify with usage access, hidden over Soundalike's own screens, one tap
opens the sheet, hold hides it), and the song-change line.

Match a part, release build 2.1.0 against production (2026-11-23):

| Checked | Emulator (Android 16) | Flip 2 (Android 13, Spotify playing) |
|---|---|---|
| Button shown, picker opens and scrolls into view | yes | yes, landscape and portrait |
| Drag the part, drag each edge, tap the bar, ±5 s, length buttons, limits at both ends | yes, every result read back | drags and buttons, read from screenshots |
| Play this part | starts at the part, stops by itself after its length | audible 1.0 s after the tap, ran 15.3 s for a 15 s part; Spotify paused during it and resumed |
| Apply, chip, reopen at the part, Whole preview | yes | yes; 0.8 s for parts the server had ranked, 3.4 to 5.1 s for new ones |
| Rotation with the picker open | part and picker kept, no new request | not rotated by hand; the device was in both orientations during the pass |
| Player moves to another song | not run | part kept; tapping a row still plays it and returns; "Show soundalikes" loads the new song whole |
| Offline on Apply, then Try again | error with the song shown; the same part is asked again | not run |
| In the app (search pick), YouTube link | yes | not run |
| Song times ("0:58–1:08 of the song") | not run | yes, for a song Spotify reports as longer than 90 s |

Needs a real phone (not tested):
- A Spotify Free account: whether `spotify:track:` still starts playback there.
- Apple Music for Android, and YouTube Music signed in (the search link was only seen to reach the app).
- Other Android versions and manufacturer skins; the minimum, Android 8.0, was not run.

Play Store notes: the share target, tile, widget and shortcuts need no declaration. Notification access needs a
prominent in-app disclosure (the guide and the sheet explain it before sending the user to the system screen) and
a privacy policy that says title and artist of the playing song are sent to the Soundalike server.
`SYSTEM_ALERT_WINDOW` and the optional `PACKAGE_USAGE_STATS` (usage access) are allowed but draw review attention;
both serve only the floating button and can go with it. No
Spotify name or logo is used in the app's name or icon. These points are a reading of the policy research in the
project notes, not a confirmed review outcome.
