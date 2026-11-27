package app.yassin.soundalike.core

import kotlin.math.roundToInt
import kotlinx.serialization.json.JsonObject

/** A span of a song's 30-second preview to match instead of all of it, in whole seconds of the preview. */
data class Part(val start: Int, val end: Int) {
    val length: Int get() = end - start

    /** As the server takes it: `part=10-20`. */
    val wire: String get() = "$start-$end"
}

/**
 * What the server can match a part of. Present in every answer of a server that can, so
 * the control costs no request of its own and stays hidden with a server that can't.
 */
data class PartLimits(
    /** Length of the preview. */
    val previewSeconds: Int,
    val minSeconds: Int,
    val maxSeconds: Int,
    /** Where the preview starts in the song; null when the server does not know. */
    val songOffsetSeconds: Double?,
) {
    /**
     * With the place of the preview filled in for a song of known length: previews usually start
     * 48 s into songs of 90 s or more (16 of 17 measured), so times can be shown as song times.
     */
    fun forSongOf(seconds: Int?): PartLimits =
        if (songOffsetSeconds == null && seconds != null && seconds >= Parts.LONG_SONG_SECONDS) copy(songOffsetSeconds = Parts.USUAL_OFFSET_SECONDS) else this

    /** The lengths offered: every step from the shortest to the longest. */
    val lengths: List<Int> get() = (minSeconds..maxSeconds step Parts.STEP_SECONDS).toList()
}

/** Parts are kept on a 5-second grid, 10 to 20 seconds long, like the other Soundalike clients. */
object Parts {
    const val STEP_SECONDS = 5
    private const val LONGEST_SECONDS = 20
    internal const val USUAL_OFFSET_SECONDS = 48.0
    internal const val LONG_SONG_SECONDS = 90
    val DEFAULT = Part(10, 20)

    /** `query.parts` of an answer; null when the server offers no parts or reports limits that make no sense. */
    internal fun limits(parts: JsonObject?): PartLimits? {
        if (parts == null) return null
        val total = parts.long("preview_s")?.toInt()?.takeIf { it > 0 } ?: 30
        val min = maxOf(STEP_SECONDS, parts.long("min_s")?.toInt() ?: 10)
        val max = minOf(LONGEST_SECONDS, parts.long("max_s")?.toInt()?.takeIf { it > 0 } ?: LONGEST_SECONDS)
        if (min > max || max > total) return null
        val offset = parts.double("song_offset_s")?.takeIf { it >= 0 }
        return PartLimits(total, min, max, offset)
    }

    /** `query.part` of an answer: the span the lists are for; null when they are for the whole preview. */
    internal fun part(part: JsonObject?): Part? {
        val start = part?.long("start")?.toInt() ?: return null
        val end = part.long("end")?.toInt() ?: return null
        return if (start in 0 until end) Part(start, end) else null
    }

    /** The nearest span the server takes: on the grid, of an allowed length, inside the preview. */
    fun fit(start: Double, end: Double, limits: PartLimits): Part {
        val length = snap(end - start).coerceIn(limits.minSeconds, limits.maxSeconds)
        val from = snap(start).coerceIn(0, limits.previewSeconds - length)
        return Part(from, from + length)
    }

    /** The same length, starting at `start` or as near to it as the preview allows. */
    fun move(part: Part, start: Double, limits: PartLimits): Part = fit(start, start + part.length, limits)

    /** Another length from the same start; a part that would pass the end of the preview grows backwards. */
    fun withLength(part: Part, length: Int, limits: PartLimits): Part = fit(part.start.toDouble(), (part.start + length).toDouble(), limits)

    /** The start edge dragged to second `at`; the end stays. */
    fun resizeStart(part: Part, at: Double, limits: PartLimits): Part {
        val length = snap(part.end - at).coerceIn(limits.minSeconds, limits.maxSeconds)
        return Part(maxOf(0, part.end - length), part.end)
    }

    /** The end edge dragged to second `at`; the start stays. */
    fun resizeEnd(part: Part, at: Double, limits: PartLimits): Part {
        val length = snap(at - part.start).coerceIn(limits.minSeconds, limits.maxSeconds)
        return Part(part.start, minOf(limits.previewSeconds, part.start + length))
    }

    /** "0:58–1:13 of the song", or "0:10–0:25 of the preview" when the preview's place in the song is unknown. */
    fun label(part: Part, limits: PartLimits): String {
        val base = limits.songOffsetSeconds ?: 0.0
        val of = if (limits.songOffsetSeconds == null) "preview" else "song"
        return "${clock(base + part.start)}–${clock(base + part.end)} of the $of"
    }

    /** Seconds as m:ss. */
    fun clock(seconds: Double): String {
        val whole = maxOf(0, seconds.roundToInt())
        return "${whole / 60}:${(whole % 60).toString().padStart(2, '0')}"
    }

    private fun snap(seconds: Double): Int = (seconds / STEP_SECONDS).roundToInt() * STEP_SECONDS
}
