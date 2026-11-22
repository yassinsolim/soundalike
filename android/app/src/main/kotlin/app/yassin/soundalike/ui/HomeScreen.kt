package app.yassin.soundalike.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LifecycleEventEffect
import app.yassin.soundalike.NowPlaying
import app.yassin.soundalike.PlayingSong
import app.yassin.soundalike.R
import app.yassin.soundalike.Request
import app.yassin.soundalike.app
import app.yassin.soundalike.core.Seed
import app.yassin.soundalike.core.SharedLink
import app.yassin.soundalike.core.SoundalikeError
import kotlinx.coroutines.delay

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(
    incomingSearch: String?,
    onSearchConsumed: () -> Unit,
    onFind: (Request) -> Unit,
    onSettings: () -> Unit,
    onGuide: () -> Unit,
) {
    val context = LocalContext.current
    val app = context.app
    val keyboard = LocalSoftwareKeyboardController.current
    var query by rememberSaveable { mutableStateOf("") }
    var hits by remember { mutableStateOf<List<Seed>?>(null) }
    var searchFailed by remember { mutableStateOf(false) }
    var access by remember { mutableStateOf(NowPlaying.hasAccess(context)) }
    var playing by remember { mutableStateOf(NowPlaying.read(context)) }

    LaunchedEffect(incomingSearch) {
        if (incomingSearch != null) {
            query = incomingSearch
            onSearchConsumed()
        }
    }
    // Coming back from the music app or from the access screen: look again.
    LifecycleEventEffect(Lifecycle.Event.ON_RESUME) {
        access = NowPlaying.hasAccess(context)
        playing = NowPlaying.read(context)
    }
    val pastedLink = remember(query) { SharedLink.find(query) }
    LaunchedEffect(query) {
        searchFailed = false
        if (query.isBlank() || pastedLink != null) {
            hits = null
            return@LaunchedEffect
        }
        delay(350) // wait for a pause in typing
        try {
            hits = app.deezer.search(query)
        } catch (error: SoundalikeError) {
            hits = null
            searchFailed = true
        }
    }

    Scaffold(
        containerColor = MaterialTheme.colorScheme.background,
        topBar = {
            TopAppBar(
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        BrandMark(28.dp)
                        Spacer(Modifier.width(10.dp))
                        Text("Soundalike", fontWeight = FontWeight.Bold)
                    }
                },
                actions = {
                    IconButton(onClick = onGuide) { Icon(painterResource(R.drawable.ic_help), "How to use Soundalike") }
                    IconButton(onClick = onSettings) { Icon(painterResource(R.drawable.ic_settings), "Settings") }
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.background),
            )
        },
    ) { padding ->
        LazyColumn(Modifier.fillMaxSize().padding(padding).imePadding()) {
            item {
                PlayingCard(
                    access = access, song = playing,
                    onFind = { onFind(Request.Playing) },
                    onAllow = { context.startActivity(NowPlaying.accessSettingsIntent(context)) },
                )
            }
            item {
                OutlinedTextField(
                    value = query,
                    onValueChange = { query = it },
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp),
                    placeholder = { Text("Search a song, or paste a link") },
                    leadingIcon = { Icon(painterResource(R.drawable.ic_search), null) },
                    trailingIcon = {
                        if (query.isNotEmpty()) IconButton(onClick = { query = "" }) { Icon(painterResource(R.drawable.ic_close), "Clear") }
                    },
                    singleLine = true,
                    shape = RoundedCornerShape(14.dp),
                    keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search),
                    keyboardActions = KeyboardActions(onSearch = { keyboard?.hide() }),
                )
            }
            when {
                pastedLink != null -> item {
                    Button(
                        onClick = { keyboard?.hide(); onFind(Request.Shared(query)) },
                        Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp),
                    ) { Text("Find soundalikes for this ${pastedLink.source.displayName} link") }
                }
                searchFailed -> item { Hint("Couldn't search. Check your connection and try again.") }
                query.isNotBlank() -> {
                    val found = hits
                    if (found != null && found.isEmpty()) item { Hint("No songs found for “${query.trim()}”.") }
                    items(found.orEmpty()) { seed ->
                        SongRow(seed) { keyboard?.hide(); onFind(Request.Song(seed)) }
                    }
                }
                app.history.items.isEmpty() -> item {
                    Hint("Search for a song, or share one from your music app: Share › Find soundalikes.")
                }
                else -> {
                    item {
                        Row(Modifier.fillMaxWidth().padding(start = 20.dp, end = 8.dp, top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                            Text(
                                "RECENT", Modifier.weight(1f),
                                style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.SemiBold,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                            TextButton(onClick = app.history::clear) { Text("Clear") }
                        }
                    }
                    items(app.history.items) { seed -> SongRow(seed) { onFind(Request.Song(seed)) } }
                }
            }
        }
    }
}

@Composable
private fun PlayingCard(access: Boolean, song: PlayingSong?, onFind: () -> Unit, onAllow: () -> Unit) {
    Column(
        Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 8.dp)
            .background(MaterialTheme.colorScheme.surface, RoundedCornerShape(18.dp))
            .padding(16.dp),
    ) {
        Text(
            "PLAYING NOW",
            style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.SemiBold,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.size(6.dp))
        when {
            !access -> {
                Text("Let Soundalike see which song your music app is playing, for one-tap soundalikes. No sign-in needed.")
                Spacer(Modifier.size(12.dp))
                Button(onClick = onAllow) { Text("Allow notification access") }
            }
            song == null -> Text("Nothing is playing. Start a song in Spotify, YouTube Music or any other player.", color = MaterialTheme.colorScheme.onSurfaceVariant)
            else -> {
                Text(song.title, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold, maxLines = 2, overflow = TextOverflow.Ellipsis)
                Text(song.artist, color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1, overflow = TextOverflow.Ellipsis)
                Spacer(Modifier.size(12.dp))
                Button(onClick = onFind) { Text("Find soundalikes") }
            }
        }
    }
}

@Composable
private fun SongRow(seed: Seed, onClick: () -> Unit) {
    Row(
        Modifier.fillMaxWidth().clickable(onClick = onClick).padding(horizontal = 20.dp, vertical = 7.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Artwork(seed.artworkUrl, 48.dp)
        Spacer(Modifier.width(12.dp))
        Column(Modifier.weight(1f)) {
            Text(seed.title, fontWeight = FontWeight.SemiBold, maxLines = 1, overflow = TextOverflow.Ellipsis)
            Text(seed.artist, color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodyMedium, maxLines = 1, overflow = TextOverflow.Ellipsis)
        }
    }
}

@Composable
private fun Hint(text: String) {
    Text(text, Modifier.fillMaxWidth().padding(horizontal = 24.dp, vertical = 20.dp), color = MaterialTheme.colorScheme.onSurfaceVariant)
}
