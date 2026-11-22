package app.yassin.soundalike

import android.content.BroadcastReceiver
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.media.MediaMetadata
import android.media.session.MediaController
import android.media.session.MediaSession
import android.media.session.MediaSessionManager
import android.media.session.PlaybackState
import android.os.Handler
import android.os.Looper
import android.service.notification.NotificationListenerService
import androidx.core.content.ContextCompat

/**
 * Exists so the system grants access to the active media sessions; it reads no notifications.
 * While access is on, it keeps the tile, widget, notification and floating button in step
 * with the playing song.
 */
class NowPlayingListener : NotificationListenerService() {
    private val main = Handler(Looper.getMainLooper())
    private val watched = HashMap<MediaSession.Token, MediaController>()
    private var connected = false

    private val sessionsChanged = MediaSessionManager.OnActiveSessionsChangedListener { watch(it.orEmpty()) }

    private val songChanged = object : MediaController.Callback() {
        override fun onMetadataChanged(metadata: MediaMetadata?) = Surfaces.update(this@NowPlayingListener)
        override fun onPlaybackStateChanged(state: PlaybackState?) = Surfaces.update(this@NowPlayingListener)
    }

    /** Spotify's track broadcast and the older players' scrobble broadcasts. */
    private val playerBroadcasts = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action == NowPlaying.SPOTIFY_METADATA_CHANGED) {
                NowPlaying.lastBroadcast = NowPlaying.fromBroadcast(intent) ?: return
            } else {
                NowPlaying.lastLegacy = NowPlaying.fromLegacy(intent) ?: return
            }
            Surfaces.update(context)
        }
    }

    override fun onListenerConnected() {
        connected = true
        val manager = getSystemService(MediaSessionManager::class.java)
        val self = ComponentName(this, NowPlayingListener::class.java)
        manager.addOnActiveSessionsChangedListener(sessionsChanged, self, main)
        val actions = IntentFilter(NowPlaying.SPOTIFY_METADATA_CHANGED).apply { NowPlaying.LEGACY_ACTIONS.forEach(::addAction) }
        ContextCompat.registerReceiver(this, playerBroadcasts, actions, ContextCompat.RECEIVER_EXPORTED)
        watch(manager.getActiveSessions(self))
    }

    override fun onListenerDisconnected() {
        if (!connected) return
        connected = false
        getSystemService(MediaSessionManager::class.java).removeOnActiveSessionsChangedListener(sessionsChanged)
        unregisterReceiver(playerBroadcasts)
        watch(emptyList())
    }

    private fun watch(sessions: List<MediaController>) {
        val current = sessions.associateBy { it.sessionToken }
        (watched.keys - current.keys).forEach { watched.remove(it)?.unregisterCallback(songChanged) }
        for ((token, session) in current) {
            if (token !in watched) {
                session.registerCallback(songChanged, main)
                watched[token] = session
            }
        }
        Surfaces.update(this)
    }
}
