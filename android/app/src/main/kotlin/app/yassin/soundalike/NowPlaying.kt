package app.yassin.soundalike

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.media.MediaMetadata
import android.media.session.MediaController
import android.media.session.MediaSessionManager
import android.media.session.PlaybackState
import android.os.Build
import android.provider.Settings as SystemSettings
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import app.yassin.soundalike.core.MusicSource
import app.yassin.soundalike.core.Seed

private val audioExtension = Regex("""\.(mp3|m4a|aac|flac|ogg|opus|wav|wma)$""", RegexOption.IGNORE_CASE)

/** Players of local files often report the file name as the title. */
private fun String.withoutAudioExtension() = replace(audioExtension, "").ifBlank { this }

/** The song a music app is playing, as its media session reports it. */
data class PlayingSong(
    val title: String,
    val artist: String,
    val packageName: String,
    val playing: Boolean,
    /** `spotify:track:<id>` when the session or Spotify's broadcast names it. */
    val spotifyUri: String?,
) {
    fun toSeed() = Seed(
        title = title, artist = artist,
        source = MusicSource.fromPackage(packageName) ?: MusicSource.SEARCH,
        spotifyId = spotifyUri?.substringAfterLast(':'),
    )
}

/**
 * Reads what is playing without any login: players publish a media session, and an app with
 * notification access may list the active ones. Spotify can also broadcast the exact track
 * URI when the user turns on Device Broadcast Status in Spotify's settings. Older players
 * (the AOSP Music app and its descendants) keep a session but name the song only in the
 * `com.android.music.*` broadcasts that scrobblers listen to; those are the fallback.
 */
object NowPlaying {
    const val SPOTIFY_METADATA_CHANGED = "com.spotify.music.metadatachanged"
    private const val LEGACY_PACKAGE = "com.android.music"
    private const val LEGACY_PLAY_STATE = "com.android.music.playstatechanged"
    /** Sent with `track`, `artist` and `playing` extras; all three are sticky in the AOSP player. */
    val LEGACY_ACTIONS = listOf("com.android.music.metadatachanged", "com.android.music.metachanged", LEGACY_PLAY_STATE)
    private const val SPOTIFY_PACKAGE = "com.spotify.music"
    private val spotifyTrackUri = Regex("""^spotify:track:[A-Za-z0-9]{22}$""")

    fun hasAccess(context: Context): Boolean =
        context.packageName in NotificationManagerCompat.getEnabledListenerPackages(context)

    /** The system screen where the user turns notification access on for Soundalike. */
    fun accessSettingsIntent(context: Context): Intent =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            Intent(SystemSettings.ACTION_NOTIFICATION_LISTENER_DETAIL_SETTINGS).putExtra(
                SystemSettings.EXTRA_NOTIFICATION_LISTENER_COMPONENT_NAME,
                ComponentName(context, NowPlayingListener::class.java).flattenToString(),
            )
        } else {
            Intent(SystemSettings.ACTION_NOTIFICATION_LISTENER_SETTINGS)
        }.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)

    fun sessions(context: Context): List<MediaController> = try {
        context.getSystemService(MediaSessionManager::class.java)
            .getActiveSessions(ComponentName(context, NowPlayingListener::class.java))
    } catch (denied: SecurityException) {
        emptyList()
    }

    /** The playing session's song, else the most recent paused one; null without access or a named song. */
    fun read(context: Context): PlayingSong? {
        val sessions = sessions(context)
        val session = sessions.firstOrNull { it.isPlaying && it.song() != null } ?: sessions.firstOrNull { it.song() != null }
        if (session == null) {
            // A session that names nothing: the song may be in the player's scrobble broadcast.
            // It belongs to the old-style player, never to a modern app that merely holds an idle session.
            val unnamed = sessions.firstOrNull { it.packageName == LEGACY_PACKAGE }
                ?: sessions.firstOrNull { MusicSource.fromPackage(it.packageName) == null }
                ?: return null
            val legacy = legacyBroadcast(context) ?: return null
            return PlayingSong(legacy.track.withoutAudioExtension(), legacy.artist, unnamed.packageName, legacy.playing ?: unnamed.isPlaying, null)
        }
        val (title, artist) = session.song() ?: return null
        val metadata = session.metadata
        val named = listOf(MediaMetadata.METADATA_KEY_MEDIA_ID, MediaMetadata.METADATA_KEY_MEDIA_URI)
            .firstNotNullOfOrNull { key -> metadata?.getString(key)?.takeIf(spotifyTrackUri::matches) }
        val broadcast = spotifyBroadcast(context)?.takeIf {
            session.packageName == SPOTIFY_PACKAGE && it.track.equals(title, ignoreCase = true)
        }
        return PlayingSong(title.withoutAudioExtension(), artist, session.packageName, session.isPlaying, named ?: broadcast?.uri)
    }

    private val MediaController.isPlaying get() = playbackState?.state == PlaybackState.STATE_PLAYING

    private fun MediaController.song(): Pair<String, String>? {
        val metadata = metadata ?: return null
        val title = metadata.text(MediaMetadata.METADATA_KEY_TITLE, MediaMetadata.METADATA_KEY_DISPLAY_TITLE) ?: return null
        val artist = metadata.text(
            MediaMetadata.METADATA_KEY_ARTIST, MediaMetadata.METADATA_KEY_ALBUM_ARTIST, MediaMetadata.METADATA_KEY_DISPLAY_SUBTITLE,
        ) ?: return null
        return title to artist
    }

    private fun MediaMetadata.text(vararg keys: String): String? =
        keys.firstNotNullOfOrNull { key -> getText(key)?.toString()?.trim()?.takeIf { it.isNotEmpty() } }

    class SpotifyBroadcast(val uri: String, val track: String, val artist: String)

    /** Spotify's last `metadatachanged` broadcast; it is sticky, so it can be read at any time. */
    fun spotifyBroadcast(context: Context): SpotifyBroadcast? {
        return fromBroadcast(sticky(context, SPOTIFY_METADATA_CHANGED)) ?: lastBroadcast
    }

    /** Set by [NowPlayingListener] in case the broadcast is not sticky on this phone. */
    @Volatile var lastBroadcast: SpotifyBroadcast? = null

    fun fromBroadcast(intent: Intent?): SpotifyBroadcast? {
        val uri = intent?.getStringExtra("id")?.takeIf(spotifyTrackUri::matches) ?: return null
        return SpotifyBroadcast(uri, intent.getStringExtra("track").orEmpty(), intent.getStringExtra("artist").orEmpty())
    }

    class LegacyBroadcast(val track: String, val artist: String, val playing: Boolean?)

    /** Set by [NowPlayingListener] with the newest `com.android.music.*` broadcast. */
    @Volatile var lastLegacy: LegacyBroadcast? = null

    fun fromLegacy(intent: Intent?): LegacyBroadcast? {
        val track = intent?.getStringExtra("track")?.trim().orEmpty()
        val artist = intent?.getStringExtra("artist")?.trim().orEmpty()
        if (track.isEmpty() || artist.isEmpty() || artist == "<unknown>") return null
        return LegacyBroadcast(track, artist, if (intent!!.hasExtra("playing")) intent.getBooleanExtra("playing", false) else null)
    }

    private fun sticky(context: Context, action: String): Intent? =
        ContextCompat.registerReceiver(context, null, IntentFilter(action), ContextCompat.RECEIVER_EXPORTED)

    /** The newest broadcast seen while listening, else the sticky ones: the song from any, the play state from its own. */
    private fun legacyBroadcast(context: Context): LegacyBroadcast? {
        lastLegacy?.let { return it }
        val song = LEGACY_ACTIONS.firstNotNullOfOrNull { fromLegacy(sticky(context, it)) } ?: return null
        val state = fromLegacy(sticky(context, LEGACY_PLAY_STATE))?.takeIf { it.track == song.track }
        return LegacyBroadcast(song.track, song.artist, state?.playing ?: song.playing)
    }

    /** Everything the players expose, for the diagnostics box in Settings. */
    fun report(context: Context): String = buildString {
        if (!hasAccess(context)) return "Notification access is off."
        val sessions = sessions(context)
        if (sessions.isEmpty()) appendLine("No active media session.")
        for (session in sessions) {
            appendLine("${session.packageName}  state=${session.playbackState?.state}")
            val metadata = session.metadata
            for (key in metadata?.keySet().orEmpty().sorted()) {
                val value = runCatching { metadata?.getText(key) }.getOrNull() ?: continue
                appendLine("  ${key.removePrefix("android.media.metadata.")} = $value")
            }
            session.extras?.let { extras -> appendLine("  session extras: ${extras.keySet().sorted().joinToString()}") }
        }
        val broadcast = spotifyBroadcast(context)
        appendLine(if (broadcast == null) "Spotify broadcast: none received" else "Spotify broadcast: ${broadcast.uri}  ${broadcast.track} / ${broadcast.artist}")
        legacyBroadcast(context)?.let { appendLine("com.android.music broadcast: ${it.track} / ${it.artist}  playing=${it.playing}") }
        read(context)?.let { appendLine("Used: ${it.title} / ${it.artist}  from ${it.packageName}  playing=${it.playing}  uri=${it.spotifyUri}") }
    }.trimEnd()
}
