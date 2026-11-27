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
import app.yassin.soundalike.core.Part
import app.yassin.soundalike.core.Seed
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

    /** Seconds into the preview while a part of the seed's preview plays, for the picker's playhead. */
    var partPosition: Float? by mutableStateOf(null)
        private set

    fun toggle(match: Match) = toggle(match.id, match.previewUrl, match.deezerId, part = null)

    /** Plays just that part of the song's own preview, or stops it when it is playing. */
    fun togglePart(seed: Seed, part: Part) = toggle(PART_ID, previewUrl = null, seed.deezerId, part)

    private fun toggle(id: String, previewUrl: String?, deezerId: Long?, part: Part?) {
        val wasThis = playingId == id || loadingId == id
        stop()
        if (wasThis) return
        loadingId = id
        val startMs = (part?.start ?: 0) * 1000
        job = scope.launch {
            // Preview addresses expire after a few minutes: fall back to a fresh one from Deezer.
            val started = previewUrl?.let { play(it, startMs) } ?: false
            if (!started) {
                val fresh = deezerId?.let { app.deezer.track(it)?.previewUrl }
                if (fresh == null || fresh == previewUrl || !play(fresh, startMs)) {
                    stop()
                    return@launch
                }
            }
            loadingId = null
            playingId = id
            val playing = player ?: return@launch
            while (playingId == id) {
                val position = playing.currentPosition
                if (part != null && position >= part.end * 1000) {
                    stop()
                    break
                }
                progress = position.toFloat() / playing.duration.coerceAtLeast(1)
                if (part != null) partPosition = position / 1000f
                delay(if (part != null) 40 else 100)
            }
        }
    }

    /** Suspends until the stream plays from `startMs`; false when it can't be played. */
    private suspend fun play(url: String, startMs: Int): Boolean {
        val result = kotlinx.coroutines.CompletableDeferred<Boolean>()
        val next = MediaPlayer()
        player = next
        fun start() {
            audio.requestAudioFocus(focus)
            next.start()
            result.complete(true)
        }
        try {
            next.setAudioAttributes(attributes)
            next.setDataSource(url)
            next.setOnPreparedListener {
                if (startMs > 0) {
                    next.setOnSeekCompleteListener { start() }
                    next.seekTo(startMs.toLong(), MediaPlayer.SEEK_CLOSEST)
                } else {
                    start()
                }
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
        partPosition = null
    }

    companion object {
        /** What [playingId] and [loadingId] hold while a part of the seed's preview plays. */
        const val PART_ID = "seed-part"
    }
}
