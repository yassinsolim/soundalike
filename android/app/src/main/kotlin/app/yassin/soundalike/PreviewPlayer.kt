package app.yassin.soundalike

import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioManager
import android.media.MediaPlayer
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import app.yassin.soundalike.core.Match
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/** Plays one 30-second preview at a time. The music app pauses for it and resumes afterwards. */
class PreviewPlayer(private val app: SoundalikeApp, private val scope: CoroutineScope) {
    var playingId: String? by mutableStateOf(null)
        private set
    var loadingId: String? by mutableStateOf(null)
        private set
    var progress: Float by mutableFloatStateOf(0f)
        private set

    private var player: MediaPlayer? = null
    private var job: Job? = null
    private val audio = app.getSystemService(AudioManager::class.java)
    private val attributes = AudioAttributes.Builder()
        .setUsage(AudioAttributes.USAGE_MEDIA).setContentType(AudioAttributes.CONTENT_TYPE_MUSIC).build()
    private val focus = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT)
        .setAudioAttributes(attributes)
        .setOnAudioFocusChangeListener { change -> if (change < 0) stop() }
        .build()

    fun toggle(match: Match) {
        val wasThis = playingId == match.id || loadingId == match.id
        stop()
        if (wasThis) return
        loadingId = match.id
        job = scope.launch {
            // Preview addresses expire after a few minutes: fall back to a fresh one from Deezer.
            val started = match.previewUrl?.let { play(it) } ?: false
            if (!started) {
                val fresh = match.deezerId?.let { app.deezer.track(it)?.previewUrl }
                if (fresh == null || fresh == match.previewUrl || !play(fresh)) {
                    stop()
                    return@launch
                }
            }
            loadingId = null
            playingId = match.id
            val playing = player ?: return@launch
            while (playingId == match.id) {
                progress = playing.currentPosition.toFloat() / playing.duration.coerceAtLeast(1)
                delay(100)
            }
        }
    }

    /** Suspends until the stream starts; false when it can't be played. */
    private suspend fun play(url: String): Boolean {
        val result = kotlinx.coroutines.CompletableDeferred<Boolean>()
        val next = MediaPlayer()
        player = next
        try {
            next.setAudioAttributes(attributes)
            next.setDataSource(url)
            next.setOnPreparedListener {
                audio.requestAudioFocus(focus)
                it.start()
                result.complete(true)
            }
            next.setOnErrorListener { _, _, _ ->
                if (!result.complete(false)) stop()
                true
            }
            next.setOnCompletionListener { stop() }
            next.prepareAsync()
        } catch (error: Exception) {
            result.complete(false)
        }
        val started = result.await()
        if (!started) {
            next.release()
            if (player === next) player = null
        }
        return started
    }

    fun stop() {
        job?.cancel()
        job = null
        player?.release()
        player = null
        audio.abandonAudioFocusRequest(focus)
        playingId = null
        loadingId = null
        progress = 0f
    }
}
