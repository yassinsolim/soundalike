package app.yassin.soundalike

import android.content.Intent
import android.content.res.Configuration
import android.graphics.Color.TRANSPARENT
import android.os.Bundle
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.asPaddingValues
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.key
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import app.yassin.soundalike.ui.BrandMark
import app.yassin.soundalike.ui.FailureActions
import app.yassin.soundalike.ui.ResultsList
import app.yassin.soundalike.ui.SoundalikeTheme

/**
 * Results as a bottom sheet drawn over whatever app is in front: the share target
 * ("Find soundalikes" in a music app's share sheet), `soundalike://` and /find links, and
 * the tile, widget, notification, shortcut and floating button for the playing song.
 */
class SheetActivity : ComponentActivity() {
    private val model: ResultsViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge(SystemBarStyle.dark(TRANSPARENT), SystemBarStyle.dark(TRANSPARENT))
        super.onCreate(savedInstanceState)
        // After a rotation the view model still holds the list; only a fresh start loads.
        if (model.request == null) load(intent)
        setContent {
            SoundalikeTheme { Sheet() }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        if (intent.action == ACTION_STAY) return
        setIntent(intent)
        load(intent)
    }

    /**
     * A song was just started in Spotify, which brings Spotify to the front. Spotify acts on the
     * link only while its screen is up: usually within a second, a few seconds when it has to
     * start first. So the sheet waits until the player reports another song than before the tap
     * and only then asks to be in front again, and keeps asking for a moment longer because
     * Spotify opens its own screen once more. Android allows that from an app that was just in
     * front. The song plays, the list stays, and the next tap can switch song.
     *
     * When no song change is seen (no notification access, the same song tapped again, or Spotify
     * too slow), the sheet closes and Spotify stays in front, where the song is. Coming back on a
     * timer instead looked smoother but covered Spotify before it had read the link about one
     * time in ten, and always when Spotify was not running: measured on a device.
     */
    private fun stayInFront(songBefore: String?) {
        if (!NowPlaying.hasAccess(this)) return
        val started = SystemClock.elapsedRealtime()
        var settleUntil = 0L
        stayUntil = started + WAIT_FOR_SONG_MS
        val view = window.decorView
        view.post(object : Runnable {
            override fun run() {
                val now = SystemClock.elapsedRealtime()
                val song = NowPlaying.current
                if (settleUntil == 0L && song != null && song.playing && song.key != songBefore) {
                    settleUntil = now + SETTLE_MS
                    stayUntil = settleUntil
                }
                if (settleUntil != 0L && !inFront) {
                    startActivity(
                        Intent(this@SheetActivity, SheetActivity::class.java).setAction(ACTION_STAY)
                            .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_NO_ANIMATION)
                    )
                }
                when {
                    now < stayUntil -> view.postDelayed(this, CHECK_EVERY_MS)
                    !inFront -> finish()
                }
            }
        })
    }

    /** Until this moment (elapsed realtime) the sheet must not close itself when another app takes the front. */
    private var stayUntil = 0L
    private val staying get() = SystemClock.elapsedRealtime() < stayUntil

    /** The sheet is the activity in front. */
    private var inFront = false

    override fun onResume() {
        super.onResume()
        inFront = true
    }

    override fun onPause() {
        super.onPause()
        inFront = false
    }

    private fun load(intent: Intent) {
        if (intent.action == Intent.ACTION_SEND) {
            // What a music app really hands over differs per app and version; keep it readable in logcat.
            @Suppress("DEPRECATION")
            val extras = intent.extras?.let { bundle -> bundle.keySet().joinToString { "$it=${bundle.get(it)}" } }
            Log.i("Soundalike", "share from $referrer type=${intent.type} flags=0x${Integer.toHexString(intent.flags)} extras: $extras")
        }
        model.load(Request.from(intent, referrer))
    }

    override fun onStop() {
        super.onStop()
        model.player.stop()
        // The sheet belongs to the moment it was asked for. Left behind in a hidden task, Android
        // could bring it back with an old list instead of starting it for the next request.
        // The screen turning off is not leaving: the list is still there when it comes back on.
        val screenOn = getSystemService(PowerManager::class.java).isInteractive
        if (!isChangingConfigurations && !grantingAccess && !staying && screenOn) finish()
        FloatingButton.ownScreens -= 1
        Surfaces.update(this)
    }

    override fun onStart() {
        super.onStart()
        FloatingButton.ownScreens += 1
        Surfaces.update(this)
        if (grantingAccess) {
            grantingAccess = false
            model.retry()
        }
    }

    /** True while the user is on the system's notification-access screen; the sheet waits for them. */
    private var grantingAccess = false

    private fun allowAccess() {
        grantingAccess = true
        startActivity(NowPlaying.accessSettingsIntent(this))
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    private fun Sheet() {
        val landscape = LocalConfiguration.current.orientation == Configuration.ORIENTATION_LANDSCAPE
        // Half height first so the music app stays visible; in landscape that would be too short.
        // Keyed by orientation: a half-open state saved in portrait must not be restored into
        // a landscape sheet that has no half-open position (it would close itself).
        val sheetState = key(landscape) { rememberModalBottomSheetState(skipPartiallyExpanded = landscape) }
        ModalBottomSheet(
            onDismissRequest = ::finish,
            sheetState = sheetState,
            // Opened fully, the sheet stops below the status bar instead of sliding under its icons.
            modifier = Modifier.statusBarsPadding(),
            sheetMaxWidth = 760.dp,
            shape = RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp),
            containerColor = MaterialTheme.colorScheme.surface,
            scrimColor = Color.Black.copy(alpha = 0.45f),
            contentWindowInsets = { WindowInsets(0, 0, 0, 0) },
        ) {
            Column(Modifier.fillMaxHeight(if (landscape) 1f else 0.92f)) {
                Row(Modifier.fillMaxWidth().padding(start = 20.dp, end = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                    BrandMark(24.dp)
                    Spacer(Modifier.width(10.dp))
                    Text("Soundalike", Modifier.weight(1f), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                    IconButton(onClick = ::finish) { Icon(painterResource(R.drawable.ic_close), "Close") }
                }
                ResultsList(
                    state = model.state,
                    target = model.target,
                    onTarget = { model.target = it },
                    player = model.player,
                    onOpen = { match ->
                        val songBefore = NowPlaying.current?.takeIf { it.playing }?.key
                        model.open(match, onPlaying = { stayInFront(songBefore) })
                    },
                    playsAtOnce = model::playsAtOnce,
                    isPlaying = model::isPlayingInSpotify,
                    onRetry = model::retry,
                    onMatchPart = model::matchPart,
                    missingApp = model.missingApp,
                    onGetApp = model::installMissingApp,
                    onDismissMissingApp = { model.missingApp = null },
                    changedSong = model.changedSong,
                    onFollowSong = { model.load(Request.Playing) },
                    failureActions = FailureActions(
                        allowAccess = ::allowAccess,
                        // Shared words can be searched; a link to an album or playlist can't.
                        searchInstead = (::searchInApp).takeIf { "://" !in sharedText },
                    ),
                    modifier = Modifier.widthIn(max = 760.dp),
                    contentPadding = WindowInsets.navigationBars.asPaddingValues(),
                )
            }
        }
    }

    private val sharedText get() = (model.request as? Request.Shared)?.text.orEmpty()

    private fun searchInApp() {
        startActivity(MainActivity.searchIntent(this, sharedText.lineSequence().firstOrNull().orEmpty().take(80)))
        finish()
    }

    companion object {
        private const val WAIT_FOR_SONG_MS = 12_000L
        private const val SETTLE_MS = 1500L
        private const val CHECK_EVERY_MS = 150L
        private const val ACTION_STAY = "app.yassin.soundalike.action.STAY"
        const val ACTION_NOW_PLAYING = "app.yassin.soundalike.action.NOW_PLAYING"
        const val EXTRA_FROM = "from"
    }
}
