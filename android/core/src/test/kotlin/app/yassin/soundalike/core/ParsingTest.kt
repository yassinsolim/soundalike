package app.yassin.soundalike.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class ParsingTest {
    private fun fixture(name: String): String =
        checkNotNull(javaClass.getResource("/fixtures/$name")) { name }.readText()

    @Test fun spotifyEmbedPage() {
        val meta = assertNotNull(SeedResolver.parseSpotifyEmbed(fixture("spotify-embed.html")))
        assertEquals("Blinding Lights", meta.title)
        assertEquals("The Weeknd", meta.artist)
        assertNotNull(meta.artworkUrl)
        assertNull(SeedResolver.parseSpotifyEmbed("<html>no data</html>"))
    }

    @Test fun youTubeTitles() {
        assertEquals(SeedResolver.SongName("Blinding Lights", "The Weeknd"),
            SeedResolver.parseYouTubeTitle("The Weeknd - Blinding Lights (Official Audio)", "TheWeekndVEVO"))
        assertEquals(SeedResolver.SongName("Blinding Lights", "The Weeknd"),
            SeedResolver.parseYouTubeTitle("Blinding Lights", "The Weeknd - Topic"))
        assertEquals(SeedResolver.SongName("Du hast", "Rammstein"),
            SeedResolver.parseYouTubeTitle("Du hast [Official Video]", "Rammstein Official"))
        assertEquals(SeedResolver.SongName("Hello (Live)", "Adele"),
            SeedResolver.parseYouTubeTitle("Adele – Hello (Live)", "Some Channel"))
        assertNull(SeedResolver.parseYouTubeTitle("(Official Video)", "x"))
    }

    @Test fun anysongResponse() {
        val seed = Seed("Blinding Lights", "The Weeknd", source = MusicSource.SPOTIFY)
        val results = SoundalikeClient.parseAnysong(fixture("anysong.json"), seed)
        assertEquals(3, results.matches.size)
        assertNotNull(results.matches[0].deezerId)
        assertTrue(results.matches[0].match!! in 0..100)
        assertEquals("anysong", results.route)
        assertEquals(seed.copy(deezerId = 908604612), results.seed)
    }

    @Test fun anysongKeepsDeezerIdsAboveInt() {
        val body = """{"ok":true,"query":{"title":"T","artist":"A"},"results":{"main":[{"id":3278369711,"title":"x","artist":"y","match":97.4}]}}"""
        val results = SoundalikeClient.parseAnysong(body, Seed("", "", source = MusicSource.SEARCH))
        assertEquals(3278369711L, results.matches.single().deezerId)
        assertEquals(97, results.matches.single().match)
        assertEquals("T", results.seed.title)
    }

    @Test fun anysongFailures() {
        val seed = Seed("x", "y", source = MusicSource.SEARCH)
        assertFailsWith<SoundalikeError.NotInCatalog> { SoundalikeClient.parseAnysong("""{"ok":false,"error":"song not found"}""", seed) }
        assertFailsWith<SoundalikeError.NotInCatalog> { SoundalikeClient.parseAnysong("""{"ok":true,"results":{"main":[]}}""", seed) }
        assertFailsWith<SoundalikeError.Server> { SoundalikeClient.parseAnysong("<html>gateway</html>", seed) }
    }

    @Test fun mobileResponse() {
        val body = """
        {"ok":true,"query":{"title":"Blinding Lights","artist":"The Weeknd","isrc":"USUG11904206","artwork":"https://x/a.jpg"},
         "results":[{"title":"Take On Me","artist":"a-ha","match":81,"isrc":"GBAYE8500001","deezer_id":664107,
           "spotify_url":"https://open.spotify.com/track/2WfaOiMkCvy7F5fcp2zZ8L","apple_music_url":null,"artwork":"","preview_url":"https://p/x.mp3"},
          {"title":"No figure","artist":"x","match":null,"isrc":null,"deezer_id":"12"}]}
        """
        val results = SoundalikeClient.parseMobile(body, MusicSource.SPOTIFY, null)
        assertEquals("USUG11904206", results.seed.isrc)
        val match = results.matches.first()
        assertEquals(81, match.match)
        assertNull(match.appleMusicUrl)
        assertNull(match.artworkUrl)
        assertEquals(listOf("spotify:track:2WfaOiMkCvy7F5fcp2zZ8L", "https://open.spotify.com/track/2WfaOiMkCvy7F5fcp2zZ8L"), Destinations.spotify(match))
        assertEquals("2WfaOiMkCvy7F5fcp2zZ8L", match.spotifyId)
        assertNull(results.matches[1].spotifyId)
        assertNull(results.matches[1].match)
        assertEquals(12L, results.matches[1].deezerId)
    }
}
