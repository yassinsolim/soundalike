package app.yassin.soundalike.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import app.yassin.soundalike.PreviewPlayer
import app.yassin.soundalike.R
import app.yassin.soundalike.ResultsState
import app.yassin.soundalike.core.Match
import app.yassin.soundalike.core.MusicSource
import app.yassin.soundalike.core.Seed
import coil.compose.AsyncImage

/** Extra buttons a failure can offer, depending on where the list is shown. */
class FailureActions(
    val allowAccess: () -> Unit,
    val searchInstead: (() -> Unit)? = null,
)

/** Seed, target picker and the ranked list; shared by the sheet and the app. */
@Composable
fun ResultsList(
    state: ResultsState,
    target: MusicSource,
    onTarget: (MusicSource) -> Unit,
    player: PreviewPlayer,
    onOpen: (Match) -> Unit,
    onRetry: () -> Unit,
    failureActions: FailureActions,
    modifier: Modifier = Modifier,
    contentPadding: PaddingValues = PaddingValues(0.dp),
) {
    LazyColumn(modifier = modifier, contentPadding = contentPadding) {
        when (state) {
            is ResultsState.Loading -> {
                state.seed?.let { seed -> item { SeedHeader(seed) } }
                item {
                    Row(
                        Modifier.fillMaxWidth().heightIn(min = 140.dp),
                        horizontalArrangement = Arrangement.Center,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        CircularProgressIndicator(Modifier.size(22.dp), strokeWidth = 2.5.dp)
                        Spacer(Modifier.width(12.dp))
                        Text(
                            if (state.seed == null) "Finding the song…" else "Listening for soundalikes…",
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }
            is ResultsState.Failed -> item { Failure(state, onRetry, failureActions) }
            is ResultsState.Loaded -> {
                item { SeedHeader(state.results.seed) }
                item { TargetPicker(target, onTarget) }
                items(state.results.matches, key = { it.id }) { match ->
                    ResultRow(match, player, target, onOpen = { onOpen(match) })
                }
                item {
                    Text(
                        "Tap a cover for a 30-second preview. Tap a song to open it in ${target.displayName}.",
                        Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 16.dp),
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        style = MaterialTheme.typography.bodySmall,
                        textAlign = TextAlign.Center,
                    )
                }
            }
        }
    }
}

@Composable
private fun Failure(state: ResultsState.Failed, onRetry: () -> Unit, actions: FailureActions) {
    Column(
        Modifier.fillMaxWidth().padding(horizontal = 28.dp, vertical = 36.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Icon(painterResource(R.drawable.ic_note), null, Modifier.size(40.dp), tint = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(state.message, textAlign = TextAlign.Center)
        when (state.reason) {
            ResultsState.Reason.NO_ACCESS -> {
                Button(onClick = actions.allowAccess) { Text("Allow notification access") }
                OutlinedButton(onClick = onRetry) { Text("Try again") }
            }
            ResultsState.Reason.NOT_A_SONG_LINK ->
                actions.searchInstead?.let { search -> Button(onClick = search) { Text("Search in Soundalike") } }
            else -> Button(onClick = onRetry) { Text("Try again") }
        }
    }
}

@Composable
fun SeedHeader(seed: Seed) {
    Row(
        Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 10.dp).semantics(mergeDescendants = true) {},
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Artwork(seed.artworkUrl, 72.dp)
        Spacer(Modifier.width(14.dp))
        Column {
            Text(
                "SOUNDS LIKE",
                style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(seed.title, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold, maxLines = 2, overflow = TextOverflow.Ellipsis)
            Text(seed.artist, color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1, overflow = TextOverflow.Ellipsis)
        }
    }
}

@Composable
private fun TargetPicker(target: MusicSource, onTarget: (MusicSource) -> Unit) {
    SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 8.dp)) {
        MusicSource.targets.forEachIndexed { index, source ->
            SegmentedButton(
                selected = source == target,
                onClick = { onTarget(source) },
                shape = SegmentedButtonDefaults.itemShape(index, MusicSource.targets.size),
                icon = {},
                label = { Text(source.displayName, maxLines = 1, overflow = TextOverflow.Ellipsis, style = MaterialTheme.typography.labelMedium) },
            )
        }
    }
}

@Composable
private fun ResultRow(match: Match, player: PreviewPlayer, target: MusicSource, onOpen: () -> Unit) {
    val playing = player.playingId == match.id
    val loading = player.loadingId == match.id
    Row(
        Modifier
            .fillMaxWidth()
            .clickable(onClickLabel = "Open in ${target.displayName}", onClick = onOpen)
            .padding(horizontal = 20.dp, vertical = 7.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            Modifier
                .size(52.dp)
                .clip(RoundedCornerShape(8.dp))
                .clickable(
                    enabled = match.previewUrl != null || match.deezerId != null,
                    role = Role.Button,
                    onClick = { player.toggle(match) },
                )
                .semantics { contentDescription = if (playing) "Stop preview" else "Play preview" },
            contentAlignment = Alignment.Center,
        ) {
            Artwork(match.artworkUrl, 52.dp)
            Box(
                Modifier.size(30.dp).background(Color.Black.copy(alpha = if (playing || loading) 0.55f else 0.35f), CircleShape),
                contentAlignment = Alignment.Center,
            ) {
                if (loading) {
                    CircularProgressIndicator(Modifier.size(16.dp), color = Color.White, strokeWidth = 2.dp)
                } else {
                    Icon(painterResource(if (playing) R.drawable.ic_pause else R.drawable.ic_play), null, Modifier.size(18.dp), tint = Color.White)
                }
            }
            if (playing) {
                Box(Modifier.align(Alignment.BottomStart).padding(4.dp).fillMaxWidth(player.progress.coerceIn(0f, 1f)).height(3.dp).background(Color.White, CircleShape))
            }
        }
        Spacer(Modifier.width(12.dp))
        Column(Modifier.weight(1f)) {
            Text(match.title, fontWeight = FontWeight.SemiBold, maxLines = 1, overflow = TextOverflow.Ellipsis)
            Text(
                match.artist,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                style = MaterialTheme.typography.bodyMedium, maxLines = 1, overflow = TextOverflow.Ellipsis,
            )
        }
        match.match?.let { value ->
            Spacer(Modifier.width(8.dp))
            MatchBadge(value)
        }
        Spacer(Modifier.width(10.dp))
        Icon(painterResource(R.drawable.ic_open), null, Modifier.size(20.dp), tint = MaterialTheme.colorScheme.primary)
    }
}

@Composable
private fun MatchBadge(value: Int) {
    val color = when {
        value >= 75 -> Palette.green
        value >= 50 -> Palette.teal
        else -> Palette.muted
    }
    Text(
        "$value%",
        Modifier
            .background(color.copy(alpha = 0.15f), CircleShape)
            .padding(horizontal = 8.dp, vertical = 4.dp)
            .semantics { contentDescription = "$value percent match" },
        color = color,
        style = MaterialTheme.typography.labelMedium,
        fontWeight = FontWeight.Bold,
    )
}

@Composable
fun Artwork(url: String?, size: Dp) {
    Box(
        Modifier.size(size).clip(RoundedCornerShape(size / 6)).background(MaterialTheme.colorScheme.surfaceVariant),
        contentAlignment = Alignment.Center,
    ) {
        Icon(painterResource(R.drawable.ic_note), null, Modifier.size(size / 2.4f), tint = MaterialTheme.colorScheme.outline)
        if (url != null) {
            AsyncImage(model = url, contentDescription = null, modifier = Modifier.size(size), contentScale = ContentScale.Crop)
        }
    }
}
