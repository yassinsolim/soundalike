# Soundalike

Find tracks that genuinely sound like the song you are playing, directly
inside Spotify. Soundalike compares precomputed audio embeddings instead of
using genres, popularity, or listening-history recommendations.

![Soundalike results inside Spotify](../../docs/spicetify-results.png)

## Use it

1. Install **Soundalike** from Marketplace.
2. Right-click any track in Spotify.
3. Select **Find soundalikes**.
4. Double-click a result or use its left-side play button. Right-click it for
   Spotify's normal track actions.

That is all normal use requires. There is no Soundalike account, Python
installation, terminal command, or local server to configure. The extension
automatically uses the hosted 272,853-track recommendation library.

## Spotify-native results

- Results open on a normal Spotify page, so the Back and Forward buttons work
  and the list stays visible while music plays.
- Results use a playlist-style layout with album artwork, verified artist
  names, album names, and measured BPM.
- Spotify's own lyrics metadata keeps confidently identified languages
  together (English with English, French with French, and so on). Tracks
  without lyrics/language metadata remain eligible as a fallback, so
  instrumentals and sparse-language catalogs do not produce an empty page.
- The play button on a verified result plays that exact Spotify track
  immediately without closing or leaving the page. Double-clicking anywhere
  on the row does the same; a single click does not interrupt playback.
- Right-clicking a result opens Spotify's native menu for playlists, queueing,
  song radio, artist and album navigation, credits, and sharing.
- If Spotify cannot confidently match a recommendation, Soundalike safely
  opens a Spotify search instead of playing the wrong track.
- Successful recommendations and resolved Spotify metadata are cached locally
  for seven days, so reopening the same track avoids repeating the expensive
  recommendation and catalog lookups.

## Privacy

Soundalike sends the selected track's title and artist to
`https://soundalike.yassin.app` to retrieve recommendations. It never sends
your Spotify password, access token, library, listening history, or artwork.
Language labels come directly from Spotify inside the already-authenticated
desktop client and are not sent to Soundalike. No separate Spotify login is
required.

## Good to know

- Soundalike quietly prewarms the hosted recommender after Spotify starts.
  Opening a result while that first warm-up is still running can take up to
  about 30 seconds. Repeated tracks use the local cache, and hosted responses
  can be reused by the CDN.
- The hosted library currently covers 272,853 tracks.
- If **Find soundalikes** does not appear immediately after installation,
  fully quit and reopen Spotify.
- If a Spotify update breaks Spicetify features, update Spicetify and run
  `spicetify backup apply`.

Want private local queries, on-the-fly analysis for tracks outside the hosted
library, or a manual installation? See the
[advanced setup guide](SETUP.md).

Not using Spicetify? Try the [hosted web app](https://soundalike.yassin.app).
