package app.yassin.soundalike.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.drag
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import app.yassin.soundalike.PreviewPlayer
import app.yassin.soundalike.core.Part
import app.yassin.soundalike.core.PartLimits
import app.yassin.soundalike.core.Parts
import app.yassin.soundalike.core.SoundalikeResults
import kotlin.math.abs

/**
 * "Match a part": the matches normally follow the song's whole 30-second preview; this picks
 * 10 to 20 seconds of it. Shown only when the answer on screen says the server can do that.
 * Closed it is one button, or the part being matched with a way back to the whole preview.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun PartControl(
    results: SoundalikeResults,
    limits: PartLimits,
    player: PreviewPlayer,
    /** Reload the list for this part; null for the whole preview. */
    onMatch: (Part?) -> Unit,
    /** The picker was opened: the list brings it into view. */
    onOpened: () -> Unit,
) {
    val seed = results.seed
    val matched = results.part
    // Another song or another matched part starts over: closed, from the part that is matched.
    // The name is part of the saved-state key because the list keeps an item's saved state
    // while it is away (during a reload), and would hand the old song's picker to the new one.
    val owner = "${seed.title}|${seed.artist}|${matched?.wire}"
    var open by rememberSaveable(key = "part-open:$owner") { mutableStateOf(false) }
    val initial = (matched ?: Parts.DEFAULT).let { Parts.fit(it.start.toDouble(), it.end.toDouble(), limits) }
    var start by rememberSaveable(key = "part-start:$owner") { mutableIntStateOf(initial.start) }
    var end by rememberSaveable(key = "part-end:$owner") { mutableIntStateOf(initial.end) }
    val draft = Part(start, end)
    LaunchedEffect(open) { if (open) onOpened() }
    val partBusy = player.playingId == PreviewPlayer.PART_ID || player.loadingId == PreviewPlayer.PART_ID

    fun change(part: Part) {
        if (part == draft) return
        if (partBusy) player.stop() // what plays must be what is shown
        start = part.start
        end = part.end
    }

    if (!open) {
        Row(
            Modifier.fillMaxWidth().padding(horizontal = 20.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            if (matched == null) {
                OutlinedButton(onClick = { open = true }) {
                    Text("Match a part")
                    Spacer(Modifier.width(8.dp))
                    Beta()
                }
            } else {
                AssistChip(
                    onClick = { open = true },
                    label = { Text("Matching ${Parts.label(matched, limits)}") },
                )
                TextButton(onClick = { onMatch(null) }) { Text("Whole preview") }
            }
        }
        return
    }

    Column(
        Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp)
            .background(MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(16.dp))
            .padding(horizontal = 14.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Beta()
            Text(
                "Soundalike hears a 30-second preview. Pick the part of it you care about.",
                color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodySmall,
            )
        }
        PartTrack(limits, draft, player.partPosition, ::change)
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            val base = limits.songOffsetSeconds ?: 0.0
            for (second in 0..limits.previewSeconds step 10) {
                Text(Parts.clock(base + second), color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.labelSmall)
            }
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                Parts.label(draft, limits),
                Modifier.weight(1f), fontWeight = FontWeight.SemiBold, maxLines = 1, overflow = TextOverflow.Ellipsis,
            )
            TextButton(
                onClick = { change(Parts.move(draft, (draft.start - Parts.STEP_SECONDS).toDouble(), limits)) },
                enabled = draft.start > 0,
                modifier = Modifier.semantics { contentDescription = "Move the part 5 seconds earlier" },
            ) { Text("−5 s") }
            TextButton(
                onClick = { change(Parts.move(draft, (draft.start + Parts.STEP_SECONDS).toDouble(), limits)) },
                enabled = draft.end < limits.previewSeconds,
                modifier = Modifier.semantics { contentDescription = "Move the part 5 seconds later" },
            ) { Text("+5 s") }
        }
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), itemVerticalAlignment = Alignment.CenterVertically) {
            for (length in limits.lengths) {
                FilterChip(
                    selected = draft.length == length,
                    onClick = { change(Parts.withLength(draft, length, limits)) },
                    label = { Text("$length s") },
                )
            }
            OutlinedButton(onClick = { player.togglePart(seed, draft) }, enabled = seed.deezerId != null) {
                Text(
                    when {
                        player.playingId == PreviewPlayer.PART_ID -> "Stop"
                        player.loadingId == PreviewPlayer.PART_ID -> "Loading…"
                        else -> "Play this part"
                    }
                )
            }
            Button(
                onClick = {
                    open = false // also when the server answers for the whole preview after all
                    onMatch(draft)
                },
                enabled = draft != matched,
            ) { Text("Apply") }
            TextButton(onClick = {
                if (partBusy) player.stop()
                open = false
            }) { Text("Cancel") }
        }
        if (limits.songOffsetSeconds != null) {
            Text(
                "Song times assume the preview sits where previews usually do; in a few songs it is elsewhere.",
                color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodySmall,
            )
        }
    }
}

/**
 * The preview as a bar with the part on it. Dragging the part moves it, dragging near one of
 * its edges resizes it, and a press elsewhere brings the part there. It moves in 5-second steps.
 */
@Composable
private fun PartTrack(limits: PartLimits, part: Part, playhead: Float?, onChange: (Part) -> Unit) {
    val current by rememberUpdatedState(part)
    val change by rememberUpdatedState(onChange)
    val accent = MaterialTheme.colorScheme.primary
    val grid = MaterialTheme.colorScheme.outline
    val shape = RoundedCornerShape(10.dp)
    BoxWithConstraints(
        Modifier
            .fillMaxWidth()
            .height(56.dp)
            .clip(shape)
            .background(MaterialTheme.colorScheme.surface)
            .drawBehind {
                for (second in Parts.STEP_SECONDS until limits.previewSeconds step Parts.STEP_SECONDS) {
                    val x = size.width * second / limits.previewSeconds
                    drawLine(grid, Offset(x, size.height * 0.3f), Offset(x, size.height * 0.7f), strokeWidth = 1.dp.toPx())
                }
            }
            .pointerInput(limits) {
                val edge = 22.dp.toPx() // how near an edge a finger has to land to take hold of it
                awaitEachGesture {
                    val down = awaitFirstDown(requireUnconsumed = false)
                    down.consume()
                    val perSecond = size.width / limits.previewSeconds.toFloat()
                    val held = current
                    val x = down.position.x
                    val at = x / perSecond
                    val inside = at >= held.start && at <= held.end
                    val mode = when {
                        abs(x - held.start * perSecond) <= edge && x <= (held.start + held.length / 2f) * perSecond -> Hold.START
                        abs(x - held.end * perSecond) <= edge -> Hold.END
                        else -> Hold.MOVE
                    }
                    // Where in the part the finger is, so the part doesn't jump under it.
                    val grab = if (inside) at - held.start else held.length / 2f
                    fun follow(x: Float) {
                        val second = (x / perSecond).toDouble()
                        change(
                            when (mode) {
                                Hold.START -> Parts.resizeStart(current, second, limits)
                                Hold.END -> Parts.resizeEnd(current, second, limits)
                                Hold.MOVE -> Parts.move(current, second - grab, limits)
                            }
                        )
                    }
                    if (mode == Hold.MOVE && !inside) follow(x)
                    drag(down.id) { moved ->
                        moved.consume()
                        follow(moved.position.x)
                    }
                }
            }
            .semantics {
                contentDescription = "Part of the preview to match"
                stateDescription = "${Parts.label(part, limits)}, ${part.length} seconds"
            },
    ) {
        val perSecond = maxWidth / limits.previewSeconds
        Box(
            Modifier
                .offset(x = perSecond * part.start)
                .width(perSecond * part.length)
                .fillMaxHeight()
                .background(accent.copy(alpha = 0.32f), shape)
                .border(2.dp, accent, shape),
        ) {
            for (side in listOf(Alignment.CenterStart, Alignment.CenterEnd)) {
                Box(Modifier.align(side).padding(horizontal = 7.dp).size(width = 4.dp, height = 22.dp).background(accent, CircleShape))
            }
        }
        if (playhead != null) {
            Box(Modifier.offset(x = perSecond * playhead).width(2.dp).fillMaxHeight().background(Color.White))
        }
    }
}

private enum class Hold { START, END, MOVE }

@Composable
private fun Beta() {
    Text(
        "BETA",
        Modifier.background(MaterialTheme.colorScheme.primaryContainer, RoundedCornerShape(6.dp)).padding(horizontal = 6.dp, vertical = 2.dp),
        color = MaterialTheme.colorScheme.onPrimaryContainer,
        style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold,
    )
}
