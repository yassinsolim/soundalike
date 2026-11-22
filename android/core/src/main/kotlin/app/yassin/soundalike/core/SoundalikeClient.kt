package app.yassin.soundalike.core

/**
 * Talks to soundalike.yassin.app.
 *
 * Prefers `/api/mobile/similar`, which resolves links server-side and returns per-platform
 * URLs. Until that route is live (404) or if it fails, falls back to `/api/anysong/similar`
 * and fills art, ISRC and previews from Deezer's public API.
 */
class SoundalikeClient(
    private val http: Http,
    private val base: String = DEFAULT_BASE,
    /** `same`: matches in the seed's language. `all`: any language. */
    private val language: String = "same",
    private val useMobileEndpoint: Boolean = true,
) {
    private val resolver = SeedResolver(http)
    private val headers = mapOf("Origin" to DEFAULT_BASE, "Accept" to "application/json")

    /** Shared text or URL in, soundalikes out. */
    suspend fun similar(sharedText: String, k: Int = 20): SoundalikeResults =
        similar(SharedLink.find(sharedText) ?: throw SoundalikeError.UnsupportedLink(), k)

    suspend fun similar(link: SharedLink, k: Int = 20): SoundalikeResults {
        // The server resolves Spotify and Apple Music links itself; YouTube links are named here.
        val platform = platform(link.source)
        if (useMobileEndpoint && platform != null) {
            mobile(mapOf("url" to link.url, "platform" to platform), link.source, link.url, k)?.let { return it }
            return anysong(resolver.resolve(link), k)
        }
        return similar(resolver.resolve(link), k)
    }

    /** For a song already known by name (search, the playing song, a resolved link). */
    suspend fun similar(seed: Seed, k: Int = 20): SoundalikeResults {
        if (useMobileEndpoint) {
            // The route takes exactly one way of naming the song.
            val name = when {
                seed.spotifyId != null -> mapOf("spotify_id" to seed.spotifyId)
                seed.appleMusicId != null -> mapOf("apple_id" to seed.appleMusicId, "storefront" to (seed.storefront ?: "us"))
                !seed.isrc.isNullOrEmpty() -> mapOf("isrc" to seed.isrc)
                else -> mapOf("title" to seed.title, "artist" to seed.artist)
            }
            mobile(name + ("platform" to platform(seed.source)), seed.source, seed.sharedUrl, k)?.let { found ->
                // Keep what the phone already knows: the server has no cover for a title-and-artist seed.
                return found.copy(seed = found.seed.copy(
                    title = found.seed.title.ifEmpty { seed.title },
                    artist = found.seed.artist.ifEmpty { seed.artist },
                    artworkUrl = found.seed.artworkUrl ?: seed.artworkUrl,
                    spotifyId = seed.spotifyId, appleMusicId = seed.appleMusicId, storefront = seed.storefront,
                ))
            }
        }
        return anysong(seed, k)
    }

    private fun platform(source: MusicSource): String? = when (source) {
        MusicSource.SPOTIFY -> "spotify"
        MusicSource.APPLE_MUSIC -> "apple"
        else -> null
    }

    /**
     * Null means "use the fallback route": the route is missing (404 until it is deployed)
     * or failed. Being offline or rate limited would fail there too, so those are thrown.
     */
    private suspend fun mobile(query: Map<String, String?>, source: MusicSource, sharedUrl: String?, k: Int): SoundalikeResults? {
        val url = urlWithQuery("$base/api/mobile/similar", query + mapOf("k" to k.toString(), "lang" to language))
        val response = http.get(url, 30_000, headers)
        if (response.status == 429) throw SoundalikeError.RateLimited(response.retryAfterSeconds)
        if (response.status != 200) return null
        return runCatching { parseMobile(response.body, source, sharedUrl) }.getOrNull()
    }

    private suspend fun anysong(seed: Seed, k: Int): SoundalikeResults {
        val url = urlWithQuery("$base/api/anysong/similar", mapOf(
            "title" to seed.title, "artist" to seed.artist, "isrc" to seed.isrc, "k" to k.toString(), "lang" to language,
        ))
        val response = http.get(url, 60_000, headers)
        when (response.status) {
            200 -> Unit
            404 -> throw SoundalikeError.NotInCatalog()
            429 -> throw SoundalikeError.RateLimited(response.retryAfterSeconds)
            else -> throw SoundalikeError.Server(response.status)
        }
        val results = parseAnysong(response.body, seed)
        val deezer = DeezerCatalog(http)
        // The engine names the recording it matched; Deezer has its cover and ISRC.
        val known = results.seed
        val matched = known.deezerId?.takeIf { known.artworkUrl == null || known.isrc == null }?.let { deezer.track(it) }
        return results.copy(
            seed = known.copy(isrc = known.isrc ?: matched?.isrc, artworkUrl = known.artworkUrl ?: matched?.artworkUrl),
            matches = deezer.enrich(results.matches),
        )
    }

    companion object {
        const val DEFAULT_BASE = "https://soundalike.yassin.app"

        fun parseMobile(body: String, source: MusicSource, sharedUrl: String?): SoundalikeResults {
            val root = parseJsonObject(body)?.takeIf { it["ok"]?.toString() == "true" } ?: throw SoundalikeError.Server(200)
            val query = root.obj("query") ?: throw SoundalikeError.Server(200)
            val seed = Seed(
                title = query.string("title").orEmpty(), artist = query.string("artist").orEmpty(),
                isrc = query.string("isrc"), artworkUrl = query.string("artwork"),
                source = source, sharedUrl = sharedUrl, deezerId = query.long("deezer_id"),
            )
            val matches = root.objects("results").mapNotNull { row ->
                Match(
                    title = row.string("title") ?: return@mapNotNull null,
                    artist = row.string("artist") ?: return@mapNotNull null,
                    match = row.long("match")?.toInt(),
                    isrc = row.string("isrc"), deezerId = row.long("deezer_id"),
                    spotifyUrl = row.string("spotify_url"), appleMusicUrl = row.string("apple_music_url"),
                    artworkUrl = row.string("artwork"), previewUrl = row.string("preview_url"),
                )
            }
            if (matches.isEmpty()) throw SoundalikeError.NotInCatalog()
            return SoundalikeResults(seed, matches, route = "mobile")
        }

        fun parseAnysong(body: String, seed: Seed): SoundalikeResults {
            val root = parseJsonObject(body) ?: throw SoundalikeError.Server(200)
            if (root["ok"]?.toString() != "true") throw SoundalikeError.NotInCatalog()
            val matches = root.obj("results")?.objects("main").orEmpty().mapNotNull { row ->
                Match(
                    title = row.string("title") ?: return@mapNotNull null,
                    artist = row.string("artist") ?: return@mapNotNull null,
                    match = row.long("match")?.toInt(), deezerId = row.long("id"),
                )
            }
            if (matches.isEmpty()) throw SoundalikeError.NotInCatalog()
            val query = root.obj("query")
            // Keep the user's own spelling unless we had none.
            val resolved = seed.copy(
                title = seed.title.ifEmpty { query?.string("title").orEmpty() },
                artist = seed.artist.ifEmpty { query?.string("artist").orEmpty() },
                deezerId = query?.long("id"),
            )
            return SoundalikeResults(resolved, matches, route = "anysong")
        }
    }
}
