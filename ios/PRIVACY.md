# Soundalike for iPhone: privacy policy

Last updated: 27 November 2026

Soundalike has no accounts, no ads and no analytics. Nothing about you is sold or shared for tracking.

## What leaves your phone

- **The song you look up.** To find soundalikes, the app sends the song's title, artist and, when known, its ISRC or a shared song link to `soundalike.yassin.app`. The request carries no identifier for you or your device.
- **Song details and previews.** Covers and song details come from Deezer's public API (`api.deezer.com`) and Spotify's public track pages, and Apple Music links from Apple's iTunes Search API. 30-second previews are streamed from Deezer when you tap to play them, and nothing of them is saved. These requests contain song ids or names only.
- **A part of a song you match.** When you use Match a part with the microphone, Soundalike records only after you tap Record, and only for the 10 to 20 seconds shown. When you pick a file, it is read on your phone and only the part you choose is used. That clip is sent once to `soundalike.yassin.app` to find soundalikes and is not stored, on your phone or on the server: the server keeps no file, cache entry or log of it.
- **Audio recognition.** When you ask Soundalike to listen, ShazamKit identifies the song. Apple receives an audio signature, not the recording, under [Apple's privacy policy](https://www.apple.com/legal/privacy/). Audio is never saved.
- **Spotify, if you connect it.** Connecting gives Soundalike a Spotify token to read the song you're playing and, at your request, play or queue songs. The token is kept in the iOS Keychain on your phone and is sent only to Spotify. Disconnect in the app or at spotify.com/account/apps.
- **Apple Music, if you allow it.** MusicKit reads the playing song and, when you ask, adds songs to your queue or a playlist. This stays between the app and Apple.

## What stays on your phone

Your settings, the songs you looked up (your history), your last results and a small diagnostic log of what the app did and how long it took. None of it is uploaded. Deleting the app deletes it.

## Children

Soundalike is not directed at children and collects no personal information from anyone.

## Contact

Questions: open an issue at https://github.com/yassinsolim/soundalike/issues.
