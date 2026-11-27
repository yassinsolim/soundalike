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
import app.yassin.soundalike.core.Part
import app.yassin.soundalike.core.Seed
import app.yassin.soundalike.core.SharedLink
import app.yassin.soundalike.core.SoundalikeError
import app.yassin.soundalike.core.SoundalikeResults
import app.yassin.soundalike.core.TextMatch
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
    /** `part` is set while matches for a part of the song's preview are fetched. */
    data class Loading(val seed: Seed?, val part: Part? = null) : ResultsState
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

    /** The playing song this list was made for; null when it was made for some other song. */
    private var followed: String? = null

    /** Set once the player has moved on to another song than the one listed: the screen offers to follow it. */
    val changedSong: PlayingSong?
        get() = NowPlaying.current?.takeIf { followed != null && it.key != followed && state is ResultsState.Loaded }

    val player = PreviewPlayer(app, viewModelScope)

    /** Asks the server about the song on screen again; kept so a part can be matched for exactly that song. */
    private var ask: (suspend (Part?) -> SoundalikeResults)? = null

    fun load(request: Request) {
        this.request = request
        followed = null
        ask = null
        job?.cancel()
        player.stop()
        when (request) {
            is Request.Shared -> {
                val link = SharedLink.find(request.text)
                target = targetFor(link?.source ?: request.from)
                if (link == null) {
                    state = ResultsState.Failed(SoundalikeError.UnsupportedLink().message!!, ResultsState.Reason.NOT_A_SONG_LINK)
                    return
                }
                Shortcuts.publish(app)
                run(seed = null, part = null) { part -> app.client().similar(link, part = part) }
            }
            is Request.Song -> {
                target = targetFor(request.seed.source)
                run(request.seed, part = null) { part -> app.client().similar(request.seed, part = part) }
            }
            Request.Playing -> {
                if (!NowPlaying.hasAccess(app)) {
                    state = ResultsState.Failed(
                        "To see what's playing, allow notification access for Soundalike. It reads the song title from your music app and nothing else.",
                        ResultsState.Reason.NO_ACCESS,
                    )
                    return
                }
                val song = NowPlaying.read(app)
                NowPlaying.report(app).lines().forEach { Log.i("Soundalike", it) }
                if (song == null) {
                    state = ResultsState.Failed("Nothing is playing. Start a song in your music app, then try again.", ResultsState.Reason.NOTHING_PLAYING)
                    return
                }
                val seed = song.toSeed()
                Log.i("Soundalike", "playing: ${song.title} / ${song.artist} from ${song.packageName}")
                target = targetFor(seed.source)
                run(seed, part = null) { part -> app.client().similar(seed, part = part) }
            }
        }
    }

    /**
     * Matches a part of the preview of the song on screen, or its whole preview again (null).
     * It is the same song whatever the player has moved on to since; a newly loaded song
     * always starts from its whole preview.
     */
    fun matchPart(part: Part?) {
        val again = ask ?: return
        val seed = when (val shown = state) {
            is ResultsState.Loaded -> shown.results.seed
            is ResultsState.Loading -> shown.seed
            is ResultsState.Failed -> shown.seed
        }
        job?.cancel()
        player.stop()
        run(seed, part, again)
    }

    /** The part asked for last, for [retry]. */
    private var asked: Part? = null

    private fun run(seed: Seed?, part: Part?, ask: suspend (Part?) -> SoundalikeResults) {
        val sameSong = this.ask === ask
        this.ask = ask
        asked = part
        state = ResultsState.Loading(seed, part)
        job = viewModelScope.launch {
            val started = SystemClock.elapsedRealtime()
            try {
                val results = ask(part)
                app.history.add(results.seed)
                if (!sameSong) {
                    followed = NowPlaying.read(app)?.takeIf {
                        request == Request.Playing || TextMatch.normalize(it.title) == TextMatch.normalize(results.seed.title)
                    }?.key
                }
                state = ResultsState.Loaded(results)
                val what = results.part?.let { "part ${it.wire}" } ?: if (part != null) "whole preview (part ${part.wire} not taken)" else "whole preview"
                app.lastAnswer = "${results.matches.size} results from the ${results.route} route in ${SystemClock.elapsedRealtime() - started} ms, $what"
                Log.i("Soundalike", app.lastAnswer!!)
            } catch (error: SoundalikeError) {
                state = ResultsState.Failed(error.message!!, seed = seed)
            }
        }
    }

    /** Asks again for what failed: the same song and, when a part was asked for, the same part. */
    fun retry() {
        val again = ask
        if (again != null) run((state as? ResultsState.Failed)?.seed, asked, again) else request?.let(::load)
    }

    private fun targetFor(source: MusicSource?): MusicSource =
        source?.takeIf { it in MusicSource.targets } ?: app.settings.defaultTarget

    /** The Spotify track id of the row tapped last, to mark it before Spotify reports the song itself. */
    private var startedId: String? by mutableStateOf(null)

    /** True for the row whose song Spotify is playing: what the player reports, else the row tapped last. */
    fun isPlayingInSpotify(match: Match): Boolean {
        val id = match.spotifyId ?: return false
        // Without notification access the player can't be seen: the row tapped last is the best guess.
        if (!NowPlaying.hasAccess(app)) return startedId == id
        val song = NowPlaying.current?.takeIf { it.playing }
        val reported = song?.spotifyUri ?: return false
        // Spotify may play another release of the tapped song, under another id: the title settles it.
        return reported == "spotify:track:$id" ||
            (startedId == id && TextMatch.coreTitle(song.title) == TextMatch.coreTitle(match.title))
    }

    /** True when tapping the row starts the song itself in Spotify instead of opening a search for it. */
    fun playsAtOnce(match: Match): Boolean =
        target == MusicSource.SPOTIFY && match.spotifyId != null && app.settings.isInstalled(MusicSource.SPOTIFY)

    /**
     * Opens a result in the chosen music app. `onPlaying` runs when the song itself was started
     * in Spotify, which is the moment the sheet may come back in front of it.
     */
    fun open(match: Match, onPlaying: () -> Unit = {}) {
        player.stop()
        val playsAtOnce = playsAtOnce(match)
        val storefront = (state as? ResultsState.Loaded)?.results?.seed?.storefront
            ?: Locale.getDefault().country.lowercase().takeIf { it.length == 2 } ?: "us"
        viewModelScope.launch {
            val message = when (Opener.open(app, match, target, storefront)) {
                Opener.Outcome.IN_APP -> {
                    if (playsAtOnce) {
                        startedId = match.spotifyId
                        followed = "spotify:track:${match.spotifyId}" // the user chose this song: no "song changed" prompt for it
                        onPlaying()
                    }
                    return@launch
                }
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
