package app.yassin.soundalike.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import kotlinx.coroutines.test.runTest

/** Answers by URL prefix and records what was asked. */
private class FakeHttp(private val routes: Map<String, (String) -> HttpResponse>) : Http {
    val requests = mutableListOf<String>()

    override suspend fun get(url: String, timeoutMs: Int, headers: Map<String, String>): HttpResponse {
        requests += url
        val route = routes.entries.firstOrNull { url.startsWith(it.key) } ?: return HttpResponse(404, "", url)
        return route.value(url)
    }
}

private fun ok(body: String): (String) -> HttpResponse = { HttpResponse(200, body, it) }
private fun status(code: Int, retryAfter: Int? = null): (String) -> HttpResponse = { HttpResponse(code, "", it, retryAfter) }

class ClientTest {
    private val base = "https://engine.test"
    private val embed = checkNotNull(javaClass.getResource("/fixtures/spotify-embed.html")).readText()
    private val anysong = checkNotNull(javaClass.getResource("/fixtures/anysong.json")).readText()
    private val mobile = """{"ok":true,"query":{"title":"Blinding Lights","artist":"The Weeknd","isrc":"USUG11904206","artwork":null},
        "results":[{"title":"Take On Me","artist":"a-ha","match":81,"deezer_id":664107,"spotify_url":"https://open.spotify.com/search/a-ha"}]}"""
    private val deezerTrack = """{"id":1,"isrc":"GBAYE8500001","preview":"https://p/x.mp3","album":{"cover_big":"https://c/big.jpg"}}"""
    private val share = "Check this https://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b?si=x"

    @Test fun usesTheMobileRouteWhenItAnswers() = runTest {
        val http = FakeHttp(mapOf("$base/api/mobile/similar" to ok(mobile)))
        val results = SoundalikeClient(http, base).similar(share)
        assertEquals("mobile", results.route)
        assertEquals("Take On Me", results.matches.single().title)
        assertEquals(
            listOf("$base/api/mobile/similar?k=20&lang=same&platform=spotify&url=https%3A%2F%2Fopen.spotify.com%2Ftrack%2F0VjIjW4GlUZAMYd2vXMi3b"),
            http.requests,
        )
    }

    @Test fun fallsBackToAnysongAndDeezerWhileTheMobileRouteIsMissing() = runTest {
        val http = FakeHttp(mapOf(
            "https://open.spotify.com/embed/track/0VjIjW4GlUZAMYd2vXMi3b" to ok(embed),
            "$base/api/anysong/similar" to ok(anysong),
            "https://api.deezer.com/track/" to ok(deezerTrack),
        ))
        val results = SoundalikeClient(http, base, language = "all").similar(share, k = 5)
        assertEquals("anysong", results.route)
        assertEquals("The Weeknd", results.seed.artist)
        assertEquals("0VjIjW4GlUZAMYd2vXMi3b", results.seed.spotifyId)
        assertEquals(3, results.matches.size)
        assertTrue(results.matches.all { it.isrc == "GBAYE8500001" && it.previewUrl == "https://p/x.mp3" && it.artworkUrl == "https://c/big.jpg" })
        assertEquals("GBAYE8500001", results.seed.isrc)
        // The mobile route once, then the embed page for the name, then the engine by name.
        assertTrue(http.requests[0].startsWith("$base/api/mobile/similar?"))
        assertTrue(http.requests[1].startsWith("https://open.spotify.com/embed/"))
        assertEquals("$base/api/anysong/similar?artist=The%20Weeknd&k=5&lang=all&title=Blinding%20Lights", http.requests[2])
        assertEquals(1, http.requests.count { "/api/mobile/" in it })
    }

    @Test fun namesTheSeedInExactlyOneWay() = runTest {
        val http = FakeHttp(mapOf("$base/api/mobile/similar" to ok(mobile)))
        val client = SoundalikeClient(http, base)
        client.similar(Seed("Hello", "Adele", source = MusicSource.SEARCH, artworkUrl = "https://c/a.jpg")).let {
            assertEquals("https://c/a.jpg", it.seed.artworkUrl)
        }
        client.similar(Seed("Hello", "Adele", isrc = "GBBKS1500214", source = MusicSource.SEARCH))
        client.similar(Seed("Hello", "Adele", source = MusicSource.APPLE_MUSIC, appleMusicId = "1", storefront = "gb"))
        assertEquals(listOf(
            "$base/api/mobile/similar?artist=Adele&k=20&lang=same&title=Hello",
            "$base/api/mobile/similar?isrc=GBBKS1500214&k=20&lang=same",
            "$base/api/mobile/similar?apple_id=1&k=20&lang=same&platform=apple&storefront=gb",
        ), http.requests)
    }

    @Test fun youTubeLinksAreNamedOnThePhone() = runTest {
        val http = FakeHttp(mapOf(
            "https://www.youtube.com/oembed" to ok("""{"title":"The Weeknd - Blinding Lights (Official Audio)","author_name":"TheWeekndVEVO","thumbnail_url":"https://i/t.jpg"}"""),
            "$base/api/mobile/similar" to ok(mobile),
        ))
        val results = SoundalikeClient(http, base).similar("https://music.youtube.com/watch?v=fHI8X4OXluQ&si=1")
        assertEquals(MusicSource.YOUTUBE_MUSIC, results.seed.source)
        assertEquals("https://i/t.jpg", results.seed.artworkUrl)
        assertEquals("$base/api/mobile/similar?artist=The%20Weeknd&k=20&lang=same&title=Blinding%20Lights", http.requests.last())
    }

    @Test fun shortLinksAreExpanded() = runTest {
        val http = FakeHttp(mapOf(
            "https://spotify.link/abc" to { HttpResponse(200, "", "https://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b?si=1") },
            "https://open.spotify.com/embed/track/0VjIjW4GlUZAMYd2vXMi3b" to ok(embed),
            "$base/api/anysong/similar" to ok(anysong),
        ))
        val results = SoundalikeClient(http, base, useMobileEndpoint = false).similar(SharedLink.SpotifyShort("https://spotify.link/abc"))
        assertEquals("Blinding Lights", results.seed.title)
        val elsewhere = FakeHttp(mapOf("https://spotify.link/abc" to { HttpResponse(200, "<html>playlist</html>", "https://open.spotify.com/playlist/x") }))
        assertFailsWith<SoundalikeError.UnsupportedLink> {
            SoundalikeClient(elsewhere, base, useMobileEndpoint = false).similar(SharedLink.SpotifyShort("https://spotify.link/abc"))
        }
    }

    @Test fun failuresAreNamed() = runTest {
        val seed = Seed("Hello", "Adele", source = MusicSource.SEARCH)
        assertFailsWith<SoundalikeError.UnsupportedLink> { SoundalikeClient(FakeHttp(emptyMap()), base).similar("no link here") }
        assertFailsWith<SoundalikeError.NotInCatalog> { SoundalikeClient(FakeHttp(emptyMap()), base).similar(seed) }
        assertFailsWith<SoundalikeError.Server> {
            SoundalikeClient(FakeHttp(mapOf("$base/api/" to status(503))), base).similar(seed)
        }
        val limited = FakeHttp(mapOf("$base/api/mobile/" to status(429, retryAfter = 42)))
        assertEquals(42, assertFailsWith<SoundalikeError.RateLimited> { SoundalikeClient(limited, base).similar(seed) }.retryAfterSeconds)
        assertEquals(1, limited.requests.size)
        val limitedOnFallback = FakeHttp(mapOf("$base/api/anysong/" to status(429)))
        assertFailsWith<SoundalikeError.RateLimited> { SoundalikeClient(limitedOnFallback, base).similar(seed) }
        val offline = Http { _, _, _ -> throw SoundalikeError.Offline() }
        assertFailsWith<SoundalikeError.Offline> { SoundalikeClient(offline, base).similar(seed) }
        val unknownSong = FakeHttp(mapOf("https://open.spotify.com/embed/" to status(404)))
        assertFailsWith<SoundalikeError.SongNotIdentified> { SoundalikeClient(unknownSong, base).similar("spotify:track:0VjIjW4GlUZAMYd2vXMi3b") }
    }
}
