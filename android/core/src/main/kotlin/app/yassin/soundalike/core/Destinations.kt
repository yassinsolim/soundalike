package app.yassin.soundalike.core

/** Where tapping a result goes, without any Spotify, Apple or Google sign-in. */
object Destinations {
    private val spotifyTrackPath = Regex("""^https?://open\.spotify\.com/(?:intl-[a-z-]+/)?track/([A-Za-z0-9]{22})(?:[/?#].*)?$""")

    /**
     * Addresses to try in order. A known track id opens the song itself (`spotify:track:`,
     * then the same song as a web link for phones without the app). Otherwise a search for
     * `isrc:<code>`, which lands on exactly that recording, then for title and artist.
     */
    fun spotify(match: Match): List<String> {
        val id = match.spotifyId ?: match.spotifyUrl?.let(::spotifyTrackId)
        if (id != null) return listOf("spotify:track:$id", "https://open.spotify.com/track/$id")
        if (!match.isrc.isNullOrEmpty()) return listOf(spotifySearch("isrc:${match.isrc}"))
        return listOf(match.spotifyUrl ?: spotifySearch("${match.title} ${match.artist}"))
    }

    /** The track id in a link to a song's own Spotify page; null for search and other links. */
    fun spotifyTrackId(url: String): String? = spotifyTrackPath.find(url)?.groupValues?.get(1)

    fun spotifySearch(text: String): String = "https://open.spotify.com/search/${percentEncode(text)}"

    fun youTubeMusic(match: Match): String =
        urlWithQuery("https://music.youtube.com/search", mapOf("q" to "${match.title} ${match.artist}"))

    /** The song's own Apple Music page when the server or the iTunes Search API finds it, else a search. */
    suspend fun appleMusic(http: Http, match: Match, storefront: String = "us"): String {
        match.appleMusicUrl?.takeIf { "/search?" !in it }?.let { return it }
        val search = urlWithQuery("https://itunes.apple.com/search", mapOf(
            "term" to "${TextMatch.coreTitle(match.title)} ${TextMatch.primaryArtist(match.artist)}",
            "entity" to "song", "limit" to "10", "country" to storefront,
        ))
        val rows = try {
            parseJsonObject(http.get(search, 8_000).body)?.objects("results").orEmpty()
        } catch (error: SoundalikeError) {
            emptyList()
        }
        val best = rows.mapNotNull { row ->
            val url = row.string("trackViewUrl") ?: return@mapNotNull null
            val score = TextMatch.score(match.title, match.artist, row.string("trackName").orEmpty(), row.string("artistName").orEmpty())
            if (score >= 6) score to url else null
        }.maxByOrNull { it.first }
        if (best != null) return stripTracking(best.second)
        return match.appleMusicUrl
            ?: urlWithQuery("https://music.apple.com/$storefront/search", mapOf("term" to "${match.title} ${match.artist}"))
    }

    internal fun stripTracking(url: String): String =
        url.replace(Regex("""([?&])uo=[^&]*&?"""), "$1").trimEnd('?', '&')
}
