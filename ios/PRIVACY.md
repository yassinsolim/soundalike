# Soundalike for iPhone: privacy policy

Last updated: 27 November 2026

Soundalike has no accounts, no ads and no analytics. Nothing about you is sold or shared for tracking.

## What leaves your phone

- **The song you look up.** To find soundalikes, the app sends the song's title, artist and, when known, its ISRC or a shared song link to `soundalike.yassin.app`. The request carries no identifier for you or your device.
- **Song details.** Covers and song details come from Deezer's public API (`api.deezer.com`) and Spotify's public track pages, and Apple Music links from Apple's iTunes Search API. These requests contain song ids or names only.
- **Audio recognition.** When you ask Soundalike to listen, ShazamKit identifies the song. Apple receives an audio signature, not the recording, under [Apple's privacy policy](https://www.apple.com/legal/privacy/). Audio is never saved.
- **Spotify, if you connect it.** Connecting gives Soundalike a Spotify token to read the song you're playing and, at your request, play or queue songs. The token is kept in the iOS Keychain on your phone and is sent only to Spotify. Disconnect in the app or at spotify.com/account/apps.
- **Apple Music, if you allow it.** MusicKit reads the playing song and, when you ask, adds songs to your queue or a playlist. This stays between the app and Apple.

## What stays on your phone

Your settings, the songs you looked up (your history), your last results and a small diagnostic log of what the app did and how long it took. None of it is uploaded. Deleting the app deletes it.

## Children

Soundalike is not directed at children and collects no personal information from anyone.

## Contact

Questions: open an issue at https://github.com/yassinsolim/soundalike/issues.
