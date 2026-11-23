package app.yassin.soundalike.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlinx.coroutines.test.runTest

class PartTest {
    private val limits = PartLimits(previewSeconds = 30, minSeconds = 10, maxSeconds = 20, songOffsetSeconds = null)
    private fun parts(json: String) = Parts.limits(parseJsonObject(json))

    @Test fun limitsFromTheAnswer() {
        assertEquals(PartLimits(30, 10, 20, 48.0), parts("""{"min_s":10.0,"max_s":30.0,"preview_s":30.0,"song_offset_s":48.0}"""))
        // The longest part offered is 20 s, whatever the server would take.
        assertEquals(PartLimits(30, 10, 20, null), parts("""{"min_s":10,"max_s":30,"preview_s":30,"song_offset_s":null}"""))
        assertEquals(PartLimits(30, 10, 20, null), parts("{}"))
        assertEquals(listOf(10, 15, 20), limits.lengths)
        assertNull(Parts.limits(null))
        assertNull(parts("""{"min_s":25,"max_s":30,"preview_s":30}"""))
        assertNull(parts("""{"min_s":10,"max_s":20,"preview_s":15}"""))
    }

    @Test fun spansStayOnTheGridAndInsideThePreview() {
        assertEquals(Part(10, 20), Parts.fit(11.9, 21.0, limits))
        assertEquals(Part(0, 10), Parts.fit(-4.0, 3.0, limits))
        assertEquals(Part(10, 30), Parts.fit(12.0, 60.0, limits))
        assertEquals(Part(20, 30), Parts.move(Part(10, 20), 27.0, limits))
        assertEquals(Part(0, 15), Parts.move(Part(5, 20), -3.0, limits))
        assertEquals(Part(10, 30), Parts.withLength(Part(20, 30), 20, limits))
        assertEquals(Part(5, 20), Parts.withLength(Part(5, 15), 15, limits))
    }

    @Test fun edgesResizeWithinTheLimits() {
        assertEquals(Part(5, 20), Parts.resizeStart(Part(10, 20), 4.0, limits))
        assertEquals(Part(10, 20), Parts.resizeStart(Part(10, 20), 16.0, limits))  // not shorter than 10 s
        assertEquals(Part(0, 20), Parts.resizeStart(Part(10, 20), -8.0, limits))   // not longer than 20 s
        assertEquals(Part(0, 10), Parts.resizeStart(Part(0, 10), -5.0, limits))    // nothing before the preview
        assertEquals(Part(10, 25), Parts.resizeEnd(Part(10, 20), 26.0, limits))
        assertEquals(Part(10, 20), Parts.resizeEnd(Part(10, 20), 12.0, limits))
        assertEquals(Part(20, 30), Parts.resizeEnd(Part(20, 30), 40.0, limits))
    }

    @Test fun labels() {
        assertEquals("0:10–0:25 of the preview", Parts.label(Part(10, 25), limits))
        assertEquals("0:58–1:13 of the song", Parts.label(Part(10, 25), limits.copy(songOffsetSeconds = 48.0)))
        assertEquals("1:09–1:19 of the song", Parts.label(Part(10, 20), limits.copy(songOffsetSeconds = 59.4)))
        assertEquals("10-20", Part(10, 20).wire)
        // A song of known length gets song times from the usual place of previews; a short or unknown one does not.
        assertEquals("0:58–1:08 of the song", Parts.label(Part(10, 20), limits.forSongOf(200)))
        assertEquals(limits, limits.forSongOf(89))
        assertEquals(limits, limits.forSongOf(null))
        assertEquals(59.4, limits.copy(songOffsetSeconds = 59.4).forSongOf(200).songOffsetSeconds)
    }

    private val base = "https://engine.test"
    private val seed = Seed("Hello", "Adele", source = MusicSource.SEARCH)
    private fun answer(part: String?) = """{"ok":true,
        "query":{"title":"Hello","artist":"Adele","part":${part ?: "null"},"parts":{"min_s":10,"max_s":30,"preview_s":30,"song_offset_s":48.0}},
        "results":[{"title":"Take On Me","artist":"a-ha","match":81}]}"""

    @Test fun thePartIsSentAndEchoed() = runTest {
        val requests = mutableListOf<String>()
        val http = Http { url, _, _ ->
            requests += url
            HttpResponse(200, answer(if ("part=" in url) """{"start":10,"end":25}""" else null), url)
        }
        val client = SoundalikeClient(http, base)
        val whole = client.similar(seed)
        assertNull(whole.part)
        assertEquals(PartLimits(30, 10, 20, 48.0), whole.parts)
        val part = client.similar(seed, part = Part(10, 25))
        assertEquals(Part(10, 25), part.part)
        assertEquals("$base/api/mobile/similar?artist=Adele&k=20&lang=same&part=10-25&title=Hello", requests.last())
        client.similar("https://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b", part = Part(0, 10))
        assertEquals(
            "$base/api/mobile/similar?k=20&lang=same&part=0-10&platform=spotify&url=https%3A%2F%2Fopen.spotify.com%2Ftrack%2F0VjIjW4GlUZAMYd2vXMi3b",
            requests.last(),
        )
    }

    @Test fun aServerThatIgnoresThePartAnswersForTheWholePreview() = runTest {
        val old = Http { url, _, _ -> HttpResponse(200, """{"ok":true,"query":{"title":"Hello","artist":"Adele"},"results":[{"title":"x","artist":"y"}]}""", url) }
        val results = SoundalikeClient(old, base).similar(seed, part = Part(10, 20))
        assertNull(results.part)
        assertNull(results.parts)
    }

    @Test fun aServerThatRefusesThePartIsAskedOnceMoreWithoutIt() = runTest {
        for (refusal in listOf(400, 404)) {
            val requests = mutableListOf<String>()
            val http = Http { url, _, _ ->
                requests += url
                if ("part=" in url) HttpResponse(refusal, "", url) else HttpResponse(200, answer(null), url)
            }
            val results = SoundalikeClient(http, base).similar(seed, part = Part(10, 20))
            assertNull(results.part)
            assertEquals(2, requests.size)
            assertEquals("$base/api/mobile/similar?artist=Adele&k=20&lang=same&title=Hello", requests.last())
        }
    }

    @Test fun theFallbackRouteTakesThePartToo() = runTest {
        val requests = mutableListOf<String>()
        val http = Http { url, _, _ ->
            requests += url
            when {
                "/api/mobile/" in url -> HttpResponse(503, "", url)
                "/api/anysong/" in url -> HttpResponse(200, """{"ok":true,
                    "query":{"id":1,"title":"Hello","artist":"Adele","part":{"start":10.0,"end":20.0,"heard":{"sonic":[10.0,20.0]}},
                             "parts":{"min_s":10.0,"max_s":30.0,"preview_s":30.0,"song_offset_s":null}},
                    "results":{"main":[{"id":2,"title":"x","artist":"y","match":50}]}}""", url)
                else -> HttpResponse(404, "", url)
            }
        }
        val results = SoundalikeClient(http, base).similar(seed, part = Part(10, 20))
        assertEquals("anysong", results.route)
        assertEquals(Part(10, 20), results.part)
        assertEquals(PartLimits(30, 10, 20, null), results.parts)
        assertEquals("$base/api/anysong/similar?artist=Adele&k=20&lang=same&part=10-20&title=Hello", requests.first { "/api/anysong/" in it })
    }
}
