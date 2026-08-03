# Soundalike

Find tracks that genuinely sound like the song you are playing, directly
inside Spotify. Soundalike compares precomputed audio embeddings instead of
using genres, popularity, or listening-history recommendations.

![Soundalike results inside Spotify](../../docs/spicetify-results.png)

## Use it

1. Install **Soundalike** from Marketplace.
2. Right-click any track in Spotify.
3. Select **Find soundalikes**.
4. Use the play button on a result, or right-click it for Spotify's normal
   track actions.

That is all normal use requires. There is no Soundalike account, Python
installation, terminal command, or local server to configure. The extension
automatically uses the hosted 272,853-track recommendation library.

## Spotify-native results

- Results open on a normal Spotify page, so the Back and Forward buttons work
  and the list stays visible while music plays.
- Album artwork and verified artist names load progressively on that page.
- The play button on a verified result plays that exact Spotify track
  immediately without closing or leaving the page.
- Right-clicking a result opens Spotify's native menu for playlists, queueing,
  song radio, artist and album navigation, credits, and sharing.
- If Spotify cannot confidently match a recommendation, Soundalike safely
  opens a Spotify search instead of playing the wrong track.

## Privacy

Soundalike sends the selected track's title and artist to
`https://soundalike.yassin.app` to retrieve recommendations. It never sends
your Spotify password, access token, library, listening history, or artwork.
No separate Spotify login is required.

## Good to know

- The first request after the hosted service has been idle can take about 30
  seconds while the recommendation index warms. Later requests are fast.
- The hosted library currently covers 272,853 tracks.
- If **Find soundalikes** does not appear immediately after installation,
  fully quit and reopen Spotify.
- If a Spotify update breaks Spicetify features, update Spicetify and run
  `spicetify backup apply`.

Want private local queries, on-the-fly analysis for tracks outside the hosted
library, or a manual installation? See the
[advanced setup guide](SETUP.md).

Not using Spicetify? Try the [hosted web app](https://soundalike.yassin.app).
