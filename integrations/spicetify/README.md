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
- For a seed with a known Spotify lyrics language, only candidates with that
  exact known language are shown (English with English, French with French,
  and so on). Different-language candidates and candidates with unavailable
  language metadata are hidden rather than used as fallbacks, so strict
  filtering can return fewer than 20 results.
- If Spotify has no language for the seed, Soundalike preserves the normal
  ranking. Lyrics metadata alone cannot safely distinguish an instrumental
  from a metadata failure.
- The play button on a verified result plays that exact Spotify track
  immediately without closing or leaving the page. Double-clicking anywhere
  on the row does the same; a single click does not interrupt playback.
- Right-clicking a result opens Spotify's native menu with **Find soundalikes**
  again, plus playlists, queueing, song radio, artist and album navigation,
  credits, and sharing.
- If Spotify cannot confidently match a recommendation, Soundalike safely
  searches Spotify's full Songs results before falling back to a Spotify search
  page. It never plays a low-confidence match.
- Successful recommendations and resolved Spotify metadata are cached locally
  for seven days, so reopening the same track avoids repeating the expensive
  recommendation and catalog lookups. Unresolved Spotify searches are not
  cached, so a temporary catalog miss can recover on the next attempt.

## Privacy

Soundalike sends the selected track's title and artist to
`https://soundalike-api.yassin.app` to retrieve recommendations. If that
service is unavailable, it sends the same title and artist to
`https://soundalike.yassin.app` as a fallback. It never sends your Spotify
password, access token, library, listening history, or artwork. Language
labels come directly from Spotify inside the already-authenticated desktop
client and are not sent to Soundalike. No separate Spotify login is required.

## Good to know

- Soundalike normally uses an always-on recommendation service. It quietly
  prewarms the Vercel fallback after Spotify starts; if the primary service is
  unavailable, the first fallback request after idle can take up to about 30
  seconds. Repeated tracks use the local cache, and hosted responses can be
  reused by the CDN.
- The hosted library currently covers 272,853 tracks.
- Production ranking continues to use the independently validated
  `dual_sonic64_guardrail` model. The `v=4` value in an extension request is
  the strict Spotify-lyrics policy contract, not a V4 recommendation model.
  It invalidates permissive cached results. If the always-on service has not
  received that contract yet, the extension reuses its identical v3 ranking
  response while still enforcing the strict gate locally; Vercel remains the
  infrastructure fallback.
- The completed V5 receipt favored the full-track acoustic control by 56/96
  pairwise choices, versus 54/96 for the preference head and 46/96 for fixed
  V4. That research scorer depends on Jamendo full-track section embeddings
  unavailable for arbitrary Spotify songs, so it is not falsely presented as
  the production ranker. V5's exact-language behavior is the deployable policy
  promoted here.
- Spotify availability and Soundalike library coverage are different. The
  extension can now resolve recommendations through Spotify's full Songs
  search, but a Spotify seed that is absent from the 272,853-track embedding
  index still cannot be analyzed by the hosted model.
- If **Find soundalikes** does not appear immediately after installation,
  fully quit and reopen Spotify.
- If a Spotify update breaks Spicetify features, update Spicetify and run
  `spicetify backup apply`.

Want private local queries, on-the-fly analysis for tracks outside the hosted
library, or a manual installation? See the
[advanced setup guide](SETUP.md).

Not using Spicetify? Try the [hosted web app](https://soundalike.yassin.app).
