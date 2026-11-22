package app.yassin.soundalike

import android.app.Application
import android.content.Intent
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import android.widget.Toast
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import app.yassin.soundalike.core.Match
import app.yassin.soundalike.core.MusicSource
import app.yassin.soundalike.core.Seed
import app.yassin.soundalike.core.SharedLink
import app.yassin.soundalike.core.SoundalikeError
import app.yassin.soundalike.core.SoundalikeResults
import java.util.Locale
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

/** What the user asked soundalikes for. */
sealed interface Request {
    /** Share-sheet text or a link. `from` is the app it was shared from, when Android tells us. */
    data class Shared(val text: String, val from: MusicSource? = null) : Request
    data class Song(val seed: Seed) : Request
    /** Whatever a music app is playing right now. */
    data object Playing : Request

    companion object {
        fun from(intent: Intent, referrer: Uri?): Request {
            val data = intent.data
            return when {
                intent.action == Intent.ACTION_SEND -> Shared(
                    intent.getStringExtra(Intent.EXTRA_TEXT) ?: intent.getStringExtra(Intent.EXTRA_SUBJECT).orEmpty(),
                    MusicSource.fromPackage(referrer?.takeIf { it.scheme == "android-app" }?.host),
                )
                intent.action == Intent.ACTION_VIEW && data != null -> {
                    val link = data.getQueryParameter("url") ?: data.getQueryParameter("text")
                    val title = data.getQueryParameter("title")
                    val artist = data.getQueryParameter("artist")
                    when {
                        !link.isNullOrBlank() -> Shared(link)
                        !title.isNullOrBlank() && !artist.isNullOrBlank() -> Song(Seed(title, artist, source = MusicSource.SEARCH))
                        else -> Playing
                    }
                }
                else -> Playing
            }
        }
    }
}

sealed interface ResultsState {
    data class Loading(val seed: Seed?) : ResultsState
    data class Loaded(val results: SoundalikeResults) : ResultsState
    /** `seed` is the song that was asked about, when it got that far, so the user can see what was understood. */
    data class Failed(val message: String, val reason: Reason = Reason.OTHER, val seed: Seed? = null) : ResultsState

    enum class Reason { NO_ACCESS, NOTHING_PLAYING, NOT_A_SONG_LINK, OTHER }
}

class ResultsViewModel(application: Application) : AndroidViewModel(application) {
    private val app = application as SoundalikeApp
    private var job: Job? = null

    var request: Request? by mutableStateOf(null)
        private set
    var state: ResultsState by mutableStateOf(ResultsState.Loading(null))
        private set
    /** The app a tapped result opens in: where the request came from, else the user's default. */
    var target: MusicSource by mutableStateOf(app.settings.defaultTarget)

    /** Set when a result was tapped but its music app isn't installed and the web can't show the song. */
    var missingApp: MusicSource? by mutableStateOf(null)

    fun installMissingApp() {
        missingApp?.let { Opener.openStore(app, it) }
        missingApp = null
    }

    val player = PreviewPlayer(app, viewModelScope)

    fun load(request: Request) {
        this.request = request
        job?.cancel()
        player.stop()
        job = viewModelScope.launch {
            val started = SystemClock.elapsedRealtime()
            try {
                val results = when (request) {
                    is Request.Shared -> {
                        val link = SharedLink.find(request.text)
                        target = targetFor(link?.source ?: request.from)
                        state = ResultsState.Loading(null)
                        if (link == null) {
                            state = ResultsState.Failed(SoundalikeError.UnsupportedLink().message!!, ResultsState.Reason.NOT_A_SONG_LINK)
                            return@launch
                        }
                        Shortcuts.publish(app)
                        app.client().similar(link)
                    }
                    is Request.Song -> {
                        target = targetFor(request.seed.source)
                        state = ResultsState.Loading(request.seed)
                        app.client().similar(request.seed)
                    }
                    Request.Playing -> {
                        if (!NowPlaying.hasAccess(app)) {
                            state = ResultsState.Failed(
                                "To see what's playing, allow notification access for Soundalike. It reads the song title from your music app and nothing else.",
                                ResultsState.Reason.NO_ACCESS,
                            )
                            return@launch
                        }
                        val song = NowPlaying.read(app)
                        if (song == null) {
                            state = ResultsState.Failed("Nothing is playing. Start a song in your music app, then try again.", ResultsState.Reason.NOTHING_PLAYING)
                            return@launch
                        }
                        val seed = song.toSeed()
                        Log.i("Soundalike", "playing: ${song.title} / ${song.artist} from ${song.packageName}")
                        target = targetFor(seed.source)
                        state = ResultsState.Loading(seed)
                        app.client().similar(seed)
                    }
                }
                app.history.add(results.seed)
                state = ResultsState.Loaded(results)
                app.lastAnswer = "${results.matches.size} results from the ${results.route} route in ${SystemClock.elapsedRealtime() - started} ms"
                Log.i("Soundalike", app.lastAnswer!!)
            } catch (error: SoundalikeError) {
                state = ResultsState.Failed(error.message!!, seed = (state as? ResultsState.Loading)?.seed)
            }
        }
    }

    fun retry() {
        request?.let(::load)
    }

    private fun targetFor(source: MusicSource?): MusicSource =
        source?.takeIf { it in MusicSource.targets } ?: app.settings.defaultTarget

    /** Opens a result in the chosen music app. */
    fun open(match: Match) {
        player.stop()
        val storefront = (state as? ResultsState.Loaded)?.results?.seed?.storefront
            ?: Locale.getDefault().country.lowercase().takeIf { it.length == 2 } ?: "us"
        viewModelScope.launch {
            val message = when (Opener.open(app, match, target, storefront)) {
                Opener.Outcome.IN_APP -> return@launch
                Opener.Outcome.APP_MISSING -> {
                    missingApp = target
                    return@launch
                }
                Opener.Outcome.IN_BROWSER -> "${target.displayName} isn't installed. Opening it in the browser."
                Opener.Outcome.FAILED -> "Couldn't open ${target.displayName}."
            }
            Toast.makeText(app, message, Toast.LENGTH_LONG).show()
        }
    }

    override fun onCleared() {
        player.stop()
    }
}
