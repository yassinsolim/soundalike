package app.yassin.soundalike

import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import app.yassin.soundalike.ui.FailureActions
import app.yassin.soundalike.ui.GuideScreen
import app.yassin.soundalike.ui.HomeScreen
import app.yassin.soundalike.ui.ResultsList
import app.yassin.soundalike.ui.SettingsScreen
import app.yassin.soundalike.ui.SoundalikeTheme

/** The app itself: search, the playing song, history, settings and the first-run guide. */
class MainActivity : ComponentActivity() {
    private val model: ResultsViewModel by viewModels()
    /** Search text handed in by `soundalike://search?q=`; consumed by the home screen. */
    private var incomingSearch by mutableStateOf<String?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge(SystemBarStyle.dark(Color.TRANSPARENT), SystemBarStyle.dark(Color.TRANSPARENT))
        super.onCreate(savedInstanceState)
        if (savedInstanceState == null) incomingSearch = searchText(intent)
        setContent {
            SoundalikeTheme { Screens() }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        incomingSearch = searchText(intent)
    }

    override fun onStart() {
        super.onStart()
        FloatingButton.ownScreens += 1
        Surfaces.update(this)
    }

    override fun onStop() {
        super.onStop()
        model.player.stop()
        FloatingButton.ownScreens -= 1
        Surfaces.update(this)
    }

    private fun searchText(intent: Intent): String? =
        intent.data?.takeIf { it.scheme == "soundalike" && it.host == "search" }?.let { it.getQueryParameter("q").orEmpty() }

    private enum class Screen { HOME, RESULTS, SETTINGS, GUIDE }

    @Composable
    private fun Screens() {
        var screen by rememberSaveable { mutableStateOf(if (app.settings.guideSeen) Screen.HOME else Screen.GUIDE) }
        if (incomingSearch != null && screen != Screen.HOME) screen = Screen.HOME
        BackHandler(enabled = screen != Screen.HOME) { screen = Screen.HOME }

        fun find(request: Request) {
            model.load(request)
            screen = Screen.RESULTS
        }

        when (screen) {
            Screen.HOME -> HomeScreen(
                incomingSearch = incomingSearch,
                onSearchConsumed = { incomingSearch = null },
                onFind = ::find,
                onSettings = { screen = Screen.SETTINGS },
                onGuide = { screen = Screen.GUIDE },
            )
            Screen.RESULTS -> Titled("Soundalikes", onBack = { screen = Screen.HOME }) { modifier ->
                ResultsList(
                    state = model.state,
                    target = model.target,
                    onTarget = { model.target = it },
                    player = model.player,
                    onOpen = model::open,
                    onRetry = model::retry,
                    missingApp = model.missingApp,
                    onGetApp = model::installMissingApp,
                    onDismissMissingApp = { model.missingApp = null },
                    changedSong = model.changedSong,
                    onFollowSong = { model.load(Request.Playing) },
                    failureActions = FailureActions(allowAccess = { startActivity(NowPlaying.accessSettingsIntent(this)) }),
                    modifier = modifier.fillMaxSize(),
                )
            }
            Screen.SETTINGS -> Titled("Settings", onBack = { screen = Screen.HOME }) { modifier ->
                SettingsScreen(modifier, onGuide = { screen = Screen.GUIDE })
            }
            Screen.GUIDE -> Titled("How to use Soundalike", onBack = null) { modifier ->
                GuideScreen(modifier, onDone = {
                    app.settings.guideSeen = true
                    screen = Screen.HOME
                })
            }
        }
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    private fun Titled(title: String, onBack: (() -> Unit)?, content: @Composable (Modifier) -> Unit) {
        Scaffold(
            containerColor = MaterialTheme.colorScheme.background,
            topBar = {
                TopAppBar(
                    title = { Text(title, fontWeight = FontWeight.Bold) },
                    navigationIcon = {
                        if (onBack != null) IconButton(onClick = onBack) { Icon(painterResource(R.drawable.ic_back), "Back") }
                    },
                    colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.background),
                )
            },
        ) { padding ->
            Box(Modifier.fillMaxSize()) { content(Modifier.padding(padding)) }
        }
    }

    companion object {
        fun searchIntent(context: Context, text: String): Intent =
            Intent(Intent.ACTION_VIEW, Uri.Builder().scheme("soundalike").authority("search").appendQueryParameter("q", text).build())
                .setClass(context, MainActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
    }
}
