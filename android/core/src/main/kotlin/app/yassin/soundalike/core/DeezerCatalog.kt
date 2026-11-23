package app.yassin.soundalike.core

import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit

/**
 * Deezer's public, key-free API: cover art, ISRC and 30-second previews for the Deezer
 * track ids the recommender returns, plus song search for the search box.
 */
class DeezerCatalog(private val http: Http) {

    class Track(val isrc: String?, val artworkUrl: String?, val previewUrl: String?)

    suspend fun track(id: Long): Track? {
        val response = try {
            http.get("https://api.deezer.com/track/$id", 10_000)
        } catch (error: SoundalikeError) {
            return null
        }
        val track = parseJsonObject(response.body)?.takeIf { "error" !in it } ?: return null
        val album = track.obj("album")
        return Track(
            isrc = track.string("isrc"),
            artworkUrl = album?.string("cover_big") ?: album?.string("cover_medium"),
            previewUrl = track.string("preview"),
        )
    }

    /** Fills missing art, ISRC and preview, a few requests at a time. Order is kept. */
    suspend fun enrich(matches: List<Match>, concurrency: Int = 6): List<Match> = coroutineScope {
        val limit = Semaphore(concurrency)
        matches.map { match ->
            async {
                val id = match.deezerId
                if (id == null || (match.artworkUrl != null && match.previewUrl != null && match.isrc != null)) {
                    return@async match
                }
                val track = limit.withPermit { track(id) } ?: return@async match
                match.copy(
                    isrc = match.isrc ?: track.isrc,
                    artworkUrl = match.artworkUrl ?: track.artworkUrl,
                    previewUrl = match.previewUrl ?: track.previewUrl,
                )
            }
        }.awaitAll()
    }

    /** Song search for the in-app search box. */
    suspend fun search(text: String, limit: Int = 12): List<Seed> {
        val trimmed = text.trim()
        if (trimmed.isEmpty()) return emptyList()
        val url = urlWithQuery("https://api.deezer.com/search", mapOf("q" to trimmed.take(120), "limit" to limit.toString()))
        val rows = parseJsonObject(http.get(url, 10_000).body)?.objects("data").orEmpty()
        val seen = HashSet<String>()
        return rows.mapNotNull { row ->
            val title = row.string("title") ?: return@mapNotNull null
            val artist = row.obj("artist")?.string("name") ?: return@mapNotNull null
            if (!seen.add(TextMatch.normalize(title) + "|" + TextMatch.normalize(artist))) return@mapNotNull null
            Seed(title = title, artist = artist, artworkUrl = row.obj("album")?.string("cover_medium"), source = MusicSource.SEARCH)
        }
    }
}
