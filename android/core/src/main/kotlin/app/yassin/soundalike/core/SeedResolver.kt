package app.yassin.soundalike.core

/**
 * Turns a shared link into a [Seed] using public, sign-in-free pages: Spotify's embed page,
 * Apple's iTunes lookup API and YouTube's oEmbed endpoint.
 */
class SeedResolver(private val http: Http) {

    suspend fun resolve(link: SharedLink): Seed = when (link) {
        is SharedLink.SpotifyShort -> spotifySeed(expandShortLink(link.url) ?: throw SoundalikeError.UnsupportedLink())
        is SharedLink.SpotifyTrack -> spotifySeed(link)
        is SharedLink.AppleMusicSong -> appleMusicSeed(link)
        is SharedLink.YouTubeVideo -> youTubeSeed(link)
    }

    internal suspend fun expandShortLink(url: String): SharedLink.SpotifyTrack? {
        val response = http.get(url, 10_000)
        (SharedLink.parse(response.finalUrl) as? SharedLink.SpotifyTrack)?.let { return it }
        // Branch.io pages sometimes stop at an HTML hop that embeds the target.
        return SharedLink.find(response.body) as? SharedLink.SpotifyTrack
    }

    private suspend fun spotifySeed(link: SharedLink.SpotifyTrack): Seed {
        val response = http.get("https://open.spotify.com/embed/track/${link.id}", 10_000)
        val meta = response.takeIf { it.status == 200 }?.let { parseSpotifyEmbed(it.body) }
            ?: throw SoundalikeError.SongNotIdentified()
        return Seed(
            title = meta.title, artist = meta.artist, artworkUrl = meta.artworkUrl,
            source = MusicSource.SPOTIFY, sharedUrl = link.url, spotifyId = link.id,
        )
    }

    private suspend fun appleMusicSeed(link: SharedLink.AppleMusicSong): Seed {
        val lookup = urlWithQuery(
            "https://itunes.apple.com/lookup",
            mapOf("id" to link.id, "country" to link.storefront, "entity" to "song"),
        )
        val song = parseJsonObject(http.get(lookup, 10_000).body)?.objects("results")
            ?.firstOrNull { it.string("wrapperType") == "track" }
        val title = song?.string("trackName")
        val artist = song?.string("artistName")
        if (song == null || title == null || artist == null) throw SoundalikeError.SongNotIdentified()
        return Seed(
            title = title, artist = artist,
            artworkUrl = song.string("artworkUrl100")?.replace("100x100bb", "600x600bb"),
            source = MusicSource.APPLE_MUSIC, sharedUrl = link.url,
            appleMusicId = link.id, storefront = link.storefront,
        )
    }

    private suspend fun youTubeSeed(link: SharedLink.YouTubeVideo): Seed {
        val oembed = urlWithQuery(
            "https://www.youtube.com/oembed",
            mapOf("format" to "json", "url" to "https://www.youtube.com/watch?v=${link.id}"),
        )
        val response = http.get(oembed, 10_000)
        val video = response.takeIf { it.status == 200 }?.let { parseJsonObject(it.body) }
        val name = video?.string("title")?.let { parseYouTubeTitle(it, video.string("author_name").orEmpty()) }
            ?: throw SoundalikeError.SongNotIdentified()
        return Seed(
            title = name.title, artist = name.artist, artworkUrl = video.string("thumbnail_url"),
            source = MusicSource.YOUTUBE_MUSIC, sharedUrl = link.url,
        )
    }

    data class SongName(val title: String, val artist: String, val artworkUrl: String? = null)

    companion object {
        /** Reads `__NEXT_DATA__` from https://open.spotify.com/embed/track/<id>. */
        fun parseSpotifyEmbed(html: String): SongName? {
            val start = html.indexOf("<script id=\"__NEXT_DATA__\"").takeIf { it >= 0 } ?: return null
            val open = html.indexOf('>', start).takeIf { it >= 0 } ?: return null
            val close = html.indexOf("</script>", open).takeIf { it >= 0 } ?: return null
            val entity = parseJsonObject(html.substring(open + 1, close))
                ?.obj("props")?.obj("pageProps")?.obj("state")?.obj("data")?.obj("entity") ?: return null
            val title = entity.string("name") ?: entity.string("title") ?: return null
            val artist = entity.objects("artists").firstOrNull()?.string("name") ?: return null
            val images = entity.obj("visualIdentity")?.objects("image").orEmpty()
                .mapNotNull { image -> image.string("url")?.let { (image.long("maxWidth") ?: 0) to it } }
                .sortedBy { it.first }
            val artwork = (images.firstOrNull { it.first >= 300 } ?: images.lastOrNull())?.second
            return SongName(title, artist, artwork)
        }

        private val topicSuffix = Regex("""\s+-\s+Topic$""")
        private val channelSuffix = Regex("""\s*(VEVO|Official)$""", RegexOption.IGNORE_CASE)
        private val videoNoise = Regex(
            """\s*[(\[][^)\]]*\b(official|video|audio|lyrics?|visuali[sz]er|hd|hq|4k|mv)\b[^)\]]*[)\]]""",
            RegexOption.IGNORE_CASE,
        )
        private val dash = Regex("""\s+[-–—]\s+""")

        /**
         * YouTube names songs three ways: "<Artist> - Topic" channels carry the plain title,
         * music videos are titled "Artist - Title (Official Video)", anything else is the
         * video title on the artist's channel.
         */
        fun parseYouTubeTitle(title: String, author: String): SongName? {
            val cleaned = videoNoise.replace(title, "").trim()
            if (cleaned.isEmpty()) return null
            if (topicSuffix.containsMatchIn(author)) return SongName(cleaned, topicSuffix.replace(author, ""))
            val parts = dash.split(cleaned, limit = 2)
            if (parts.size == 2 && parts.all { it.isNotBlank() }) return SongName(parts[1].trim(), parts[0].trim())
            val channel = channelSuffix.replace(author, "").trim()
            return if (channel.isEmpty()) null else SongName(cleaned, channel)
        }
    }
}
