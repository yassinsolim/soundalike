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
   `apple_id`, `isrc`, or `title` + `artist`). If that route is missing (404 today) or fails, it falls back to
   `/api/anysong/similar`:
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
   - If the app is not installed, the same address opens in the browser.

## Surfaces

| Surface | What it does | Taps from a playing song | Needs |
|---|---|---|---|
| **Share target** "Find soundalikes" | Results as a bottom sheet over the music app. Also registered as a Sharing Shortcut, so it sits in the direct-share row of Android's share sheet | Spotify: Share › More › Find soundalikes = 3 from Now Playing, 4 from a song's ⋮ menu. 1 tap once Android's share sheet is open | Nothing |
| **Quick Settings tile** "Soundalikes" | Opens the sheet for the playing song | Swipe down + 1 tap | Notification access; tile added once |
| **Home-screen widget** | Shows the playing song; tap opens the sheet | 1 tap on the home screen | Notification access |
| **Notification while music plays** (off by default) | Quiet ongoing notification; tap opens the sheet | Swipe down + 1 tap | Notification access, notification permission |
| **App shortcuts** | Long-press the icon: Find soundalikes (playing song), Search a song | Long-press + 1 tap | Notification access for the first |
| **Floating button** (experimental, off by default) | Draggable button over other apps while music plays; tap opens the sheet | 1 tap, without leaving the music app | Notification access, "Display over other apps" |
| **The app** | Playing-now card, search (Deezer), pasted links, history, settings, guide | Open app + 1 tap | Notification access for the card |

Recommended default: the **share target** is what the first-run guide teaches first, because it needs no permission
and works for any song on screen, not only the playing one. For the playing song, the **Quick Settings tile** is the
recommended one-tap route: always one swipe away, nothing on screen otherwise. The floating button is the only
route with a single tap inside the music app, but it stays opt-in: Android cannot tell an app which app is in front
without accessibility or usage access, so it shows over every app while music plays, and Android adds its own
"displaying over other apps" notification.

What is playing comes from two sources:
- `MediaSessionManager.getActiveSessions`, which Android allows for apps with notification access.
  `NowPlayingListener` is an empty `NotificationListenerService`; no notification is read. It gives title, artist
  and the player's package for any player. If a session's media id or media URI is a `spotify:track:` URI it is used.
- Spotify's `com.spotify.music.metadatachanged` broadcast (extra `id` = track URI), which Spotify only sends when
  the user turns on **Device Broadcast Status** in Spotify's settings. Read as a sticky broadcast and through a
  runtime receiver. It is used when its track name matches the Spotify session's title.

Settings › "Show what my player reports" prints every text field of the active sessions and the last Spotify
broadcast; use it to see what a real player exposes.

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
| `https://soundalike.yassin.app/find?url=…` | Sheet for the link. Opens the app without asking only once the site serves `/.well-known/assetlinks.json` for `app.yassin.soundalike` with the release key's SHA-256 (`1E:CE:72:59:6B:BF:02:D1:61:FE:8E:6F:83:EA:31:70:F5:69:C4:65:A7:79:21:60:80:D3:45:BC:34:48:EF:3B`) |

## Build

Needs JDK 17 and the Android SDK with `platforms;android-37.2` and build-tools 36 (Compose 1.13 must compile
against API 37.1 or later). On the development PC everything lives under `D:\android`; `D:\android\SDK_READY.txt`
lists it.

```powershell
. D:\android\env.ps1          # JAVA_HOME, ANDROID_HOME, GRADLE_USER_HOME, PATH for this shell
cd android
.\gradlew.bat :core:test                 # 23 unit tests, plain JVM
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

Needs a real phone or a Spotify login (not tested):
- Spotify's own share flow: where "Find soundalikes" appears, the exact shared text, the tap counts above.
- That the sheet stays over Spotify when shared from it, and what Spotify does with it after a result is opened.
- What Spotify puts in its media session (media id / URI), and whether its broadcast is still sticky.
- `spotify:track:` and `open.spotify.com/search/isrc:` opening the right song in Spotify for Android (the second
  is verified on iOS only).
- Apple Music for Android, and YouTube Music signed in (the search link was only seen to reach the app).
- Other Android versions and manufacturer skins; the minimum, Android 8.0, was not run.
- `/api/mobile/similar`: covered by unit tests only, the route is not deployed.

Play Store notes: the share target, tile, widget and shortcuts need no declaration. Notification access needs a
prominent in-app disclosure (the guide and the sheet explain it before sending the user to the system screen) and
a privacy policy that says title and artist of the playing song are sent to the Soundalike server.
`SYSTEM_ALERT_WINDOW` is allowed but draws review attention; drop the floating button if it causes trouble. No
Spotify name or logo is used in the app's name or icon. These points are a reading of the policy research in the
project notes, not a confirmed review outcome.
