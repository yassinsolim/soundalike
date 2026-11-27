package app.yassin.soundalike.core

import java.text.Normalizer

/** Loose title and artist comparison, so "Song (Remastered 2011)" still matches "Song". */
object TextMatch {
    private val marks = Regex("""\p{M}+""")
    private val notAlphanumeric = Regex("""[^\p{L}\p{N}]+""")
    private val featuring = Regex(
        """\s*[(\[]\s*(feat\.?|ft\.?|featuring|with)\s+.*$|\s+(feat\.?|ft\.?|featuring)\s+.*$""",
        RegexOption.IGNORE_CASE,
    )
    private val trailingExtra = Regex("""\s*[(\[][^)\]]*[)\]]\s*$|\s+-\s+[^-]*$""")
    private val artistSuffix = Regex("""\s+(feat\.?|ft\.?|featuring|x|and)\s+.*$""", RegexOption.IGNORE_CASE)

    fun normalize(value: String): String {
        val folded = marks.replace(Normalizer.normalize(value, Normalizer.Form.NFKD), "").lowercase()
        return notAlphanumeric.replace(folded, " ").trim()
    }

    fun coreTitle(value: String): String {
        var working = featuring.replace(value, "")
        while (true) {
            val shorter = trailingExtra.replace(working, "")
            if (shorter == working) break
            working = shorter
        }
        return normalize(working.ifBlank { value })
    }

    fun primaryArtist(value: String): String {
        val first = value.split(',', '&').first()
        return normalize(artistSuffix.replace(first, ""))
    }

    /** Higher is better; 0 means unrelated. Live and remix versions lose to the plain one. */
    fun score(title: String, artist: String, candidateTitle: String, candidateArtist: String): Int {
        var score = 0
        val wanted = coreTitle(title)
        val got = coreTitle(candidateTitle)
        if (wanted == got) score += 4 else if (wanted.isNotEmpty() && got.isNotEmpty() && (wanted in got || got in wanted)) score += 1
        if (primaryArtist(artist) == primaryArtist(candidateArtist)) score += 3
        if (score == 0) return 0
        val lowered = candidateTitle.lowercase()
        if ("live" !in title.lowercase() && "live" in lowered) score -= 1
        if ("remix" !in title.lowercase() && "remix" in lowered) score -= 1
        return maxOf(score, 0)
    }
}
