package app.yassin.soundalike

import android.content.Intent
import android.content.res.Configuration
import android.graphics.Color.TRANSPARENT
import android.os.Bundle
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
        if (model.request == null) model.load(Request.from(intent, referrer))
        setContent {
            SoundalikeTheme { Sheet() }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        model.load(Request.from(intent, referrer))
    }

    override fun onStop() {
        super.onStop()
        model.player.stop()
        // The sheet belongs to the moment it was asked for. Left behind in a hidden task, Android
        // could bring it back with an old list instead of starting it for the next request.
        if (!isChangingConfigurations && !grantingAccess) finish()
    }

    override fun onStart() {
        super.onStart()
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
            sheetMaxWidth = 640.dp,
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
                    onOpen = model::open,
                    onRetry = model::retry,
                    failureActions = FailureActions(
                        allowAccess = ::allowAccess,
                        // Shared words can be searched; a link to an album or playlist can't.
                        searchInstead = (::searchInApp).takeIf { "://" !in sharedText },
                    ),
                    modifier = Modifier.widthIn(max = 640.dp),
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
        const val ACTION_NOW_PLAYING = "app.yassin.soundalike.action.NOW_PLAYING"
        const val EXTRA_FROM = "from"
    }
}
