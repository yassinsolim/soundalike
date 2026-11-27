package app.yassin.soundalike.core

/** The music app a request came from. Results open back in the same app. */
enum class MusicSource(val wire: String, val displayName: String, val androidPackage: String?) {
    SPOTIFY("spotify", "Spotify", "com.spotify.music"),
    APPLE_MUSIC("apple", "Apple Music", "com.apple.android.music"),
    YOUTUBE_MUSIC("youtube", "YouTube Music", "com.google.android.apps.youtube.music"),
    SEARCH("search", "Search", null);

    companion object {
        /** The apps a result can be opened in. */
        val targets = listOf(SPOTIFY, APPLE_MUSIC, YOUTUBE_MUSIC)

        fun fromWire(wire: String?): MusicSource? = entries.firstOrNull { it.wire == wire }

        fun fromPackage(packageName: String?): MusicSource? =
            packageName?.let { name -> entries.firstOrNull { it.androidPackage == name } }
    }
}

/** The song the user asked about. */
data class Seed(
    val title: String,
    val artist: String,
    val isrc: String? = null,
    val artworkUrl: String? = null,
    val source: MusicSource,
    /** The link the user shared, when there was one. Sent to the server resolver as is. */
    val sharedUrl: String? = null,
    val appleMusicId: String? = null,
    val storefront: String? = null,
    val spotifyId: String? = null,
    /** The recording the engine ranked from, once it has answered. */
    val deezerId: Long? = null,
    /** The song's length, when the player or the search result gave it. */
    val durationSeconds: Int? = null,
)

/** One soundalike. */
data class Match(
    val title: String,
    val artist: String,
    /** 0..100 as the server reports it; null when the engine has no figure. */
    val match: Int?,
    val isrc: String? = null,
    val deezerId: Long? = null,
    val spotifyUrl: String? = null,
    /** The Spotify track id, when the server found the song itself: such a row plays at once. */
    val spotifyId: String? = null,
    val appleMusicUrl: String? = null,
    val artworkUrl: String? = null,
    val previewUrl: String? = null,
) {
    val id: String get() = deezerId?.toString() ?: isrc ?: "$artist|$title"
}

data class SoundalikeResults(
    val seed: Seed,
    val matches: List<Match>,
    /** Which server route answered, for diagnostics. */
    val route: String,
    /** Set when the server can match a part of the preview: what it takes. */
    val parts: PartLimits? = null,
    /** The part of the preview these matches are for; null when they are for the whole preview. */
    val part: Part? = null,
)

sealed class SoundalikeError(message: String) : Exception(message) {
    class UnsupportedLink : SoundalikeError("That isn't a song link. Share a single song from Spotify, Apple Music or YouTube Music.")
    class SongNotIdentified : SoundalikeError("Couldn't work out which song that is.")
    class NotInCatalog : SoundalikeError("Soundalike doesn't know that song yet.")
    class Offline : SoundalikeError("Couldn't reach Soundalike. Check your connection and try again.")
    class RateLimited(val retryAfterSeconds: Int?) : SoundalikeError(
        if (retryAfterSeconds != null && retryAfterSeconds > 0) "Too many requests. Try again in $retryAfterSeconds seconds."
        else "Too many requests. Try again in a minute."
    )
    class Server(val status: Int) : SoundalikeError("Soundalike had a problem ($status). Try again in a moment.")
}
