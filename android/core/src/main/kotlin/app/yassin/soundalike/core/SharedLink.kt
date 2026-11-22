package app.yassin.soundalike.core

import java.net.URI

/** A song link found in whatever a share sheet handed over. */
sealed interface SharedLink {
    val url: String
    val source: MusicSource

    data class SpotifyTrack(val id: String) : SharedLink {
        override val url get() = "https://open.spotify.com/track/$id"
        override val source get() = MusicSource.SPOTIFY
    }

    /** spotify.link short links hide the track id behind a redirect. */
    data class SpotifyShort(override val url: String) : SharedLink {
        override val source get() = MusicSource.SPOTIFY
    }

    data class AppleMusicSong(val id: String, val storefront: String, override val url: String) : SharedLink {
        override val source get() = MusicSource.APPLE_MUSIC
    }

    /** A YouTube Music song or a YouTube video; both are named by the video id. */
    data class YouTubeVideo(val id: String) : SharedLink {
        override val url get() = "https://music.youtube.com/watch?v=$id"
        override val source get() = MusicSource.YOUTUBE_MUSIC
    }

    companion object {
        private val spotifyUri = Regex("""spotify:track:([A-Za-z0-9]{22})(?![A-Za-z0-9])""")
        private val webLink = Regex("""https?://[^\s<>"']+""", RegexOption.IGNORE_CASE)
        private val spotifyId = Regex("""[A-Za-z0-9]{22}""")
        private val youtubeId = Regex("""[A-Za-z0-9_-]{11}""")
        private val spotifyShortHosts = setOf("spotify.link", "link.tospotify.com", "spotify.app.link")
        private val youtubeHosts = setOf("music.youtube.com", "www.youtube.com", "youtube.com", "m.youtube.com")

        /** Finds the first song link in free text, for example "Song by Artist https://open.spotify.com/track/…". */
        fun find(text: String): SharedLink? {
            spotifyUri.find(text)?.let { return SpotifyTrack(it.groupValues[1]) }
            return webLink.findAll(text).firstNotNullOfOrNull { parse(it.value.trimEnd('.', ',', ';', ')', ']', '!', '?')) }
        }

        fun parse(url: String): SharedLink? {
            if (url.startsWith("spotify:", ignoreCase = true)) {
                return spotifyUri.find(url)?.let { SpotifyTrack(it.groupValues[1]) }
            }
            val uri = runCatching { URI(url.trim()) }.getOrNull() ?: return null
            val host = uri.host?.lowercase() ?: return null
            val parts = (uri.path ?: "").split('/').filter { it.isNotEmpty() }
            val query = queryItems(uri.rawQuery)

            if (host == "open.spotify.com" || host == "play.spotify.com") {
                // /track/<id>, /intl-de/track/<id>, /embed/track/<id>
                val id = parts.getOrNull(parts.indexOf("track") + 1)?.takeIf { "track" in parts } ?: return null
                return if (spotifyId.matches(id)) SpotifyTrack(id) else null
            }
            if (host in spotifyShortHosts) {
                return if (parts.isEmpty()) null else SpotifyShort(url.trim())
            }
            if (host.endsWith("music.apple.com") || host == "itunes.apple.com") {
                val storefront = parts.firstOrNull()?.takeIf { it.length == 2 }?.lowercase() ?: "us"
                // Album links name the song with ?i=<songId>.
                query["i"]?.takeIf(::isNumeric)?.let { return AppleMusicSong(it, storefront, url.trim()) }
                if ("song" in parts) {
                    parts.last().takeIf(::isNumeric)?.let { return AppleMusicSong(it, storefront, url.trim()) }
                }
                return null
            }
            if (host in youtubeHosts) {
                // /watch?v=<id>; playlists, channels and albums (/playlist, /browse) are not songs.
                if (parts.firstOrNull() != "watch") return null
                return query["v"]?.takeIf { youtubeId.matches(it) }?.let(::YouTubeVideo)
            }
            if (host == "youtu.be") {
                return parts.singleOrNull()?.takeIf { youtubeId.matches(it) }?.let(::YouTubeVideo)
            }
            return null
        }

        private fun isNumeric(value: String) = value.isNotEmpty() && value.all { it in '0'..'9' }

        private fun queryItems(rawQuery: String?): Map<String, String> =
            rawQuery.orEmpty().split('&').mapNotNull { item ->
                val name = item.substringBefore('=')
                if (name.isEmpty() || '=' !in item) null else name to percentDecode(item.substringAfter('='))
            }.toMap()
    }
}
