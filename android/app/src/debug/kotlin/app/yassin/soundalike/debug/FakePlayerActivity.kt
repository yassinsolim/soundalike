package app.yassin.soundalike.debug

import android.app.Activity
import android.media.MediaMetadata
import android.media.session.MediaSession
import android.media.session.PlaybackState
import android.os.Bundle

/**
 * Debug builds only. Publishes a media session the way a music player does, so the
 * now-playing surfaces can be tested without one:
 *
 *     adb shell am start -n app.yassin.soundalike/.debug.FakePlayerActivity \
 *         --es title "Blinding Lights" --es artist "The Weeknd" [--es media_id spotify:track:<id>] [--ez paused true]
 *     adb shell am start -n app.yassin.soundalike/.debug.FakePlayerActivity --ez stop true
 */
class FakePlayerActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        session?.release()
        session = null
        if (!intent.getBooleanExtra("stop", false)) {
            val state = if (intent.getBooleanExtra("paused", false)) PlaybackState.STATE_PAUSED else PlaybackState.STATE_PLAYING
            session = MediaSession(applicationContext, "FakePlayer").apply {
                setMetadata(
                    MediaMetadata.Builder()
                        .putString(MediaMetadata.METADATA_KEY_TITLE, intent.getStringExtra("title"))
                        .putString(MediaMetadata.METADATA_KEY_ARTIST, intent.getStringExtra("artist"))
                        .putString(MediaMetadata.METADATA_KEY_MEDIA_ID, intent.getStringExtra("media_id"))
                        .putLong(MediaMetadata.METADATA_KEY_DURATION, 200_000)
                        .build()
                )
                setPlaybackState(
                    PlaybackState.Builder()
                        .setActions(PlaybackState.ACTION_PLAY_PAUSE)
                        .setState(state, 12_000, 1f)
                        .build()
                )
                isActive = true
            }
        }
        finish()
    }

    private companion object {
        var session: MediaSession? = null
    }
}
