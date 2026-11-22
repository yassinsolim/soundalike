package app.yassin.soundalike.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlinx.coroutines.test.runTest

class MatchingTest {
    @Test fun variantsStillMatch() {
        assertEquals("blinding lights", TextMatch.coreTitle("Blinding Lights (Remastered 2020) - Radio Edit"))
        assertEquals("stay", TextMatch.coreTitle("Stay (feat. Justin Bieber)"))
        assertEquals("stay with me", TextMatch.coreTitle("Stay With Me"))
        assertEquals("ete", TextMatch.coreTitle("Été"))
        assertEquals("the kid laroi", TextMatch.primaryArtist("The Kid LAROI, Justin Bieber"))
    }

    @Test fun liveVersionLosesToStudio() {
        val studio = TextMatch.score("Blinding Lights", "The Weeknd", "Blinding Lights", "The Weeknd")
        val live = TextMatch.score("Blinding Lights", "The Weeknd", "Blinding Lights (Live)", "The Weeknd")
        assertTrue(studio > live)
        assertEquals(0, TextMatch.score("Blinding Lights", "The Weeknd", "Hello", "Adele"))
    }

    @Test fun spotifyDestinations() {
        assertEquals(listOf("https://open.spotify.com/search/Take%20On%20Me%20a-ha"),
            Destinations.spotify(Match("Take On Me", "a-ha", 80)))
        assertEquals(listOf("https://open.spotify.com/search/isrc%3AGBAYE8500001"),
            Destinations.spotify(Match("Été", "x", 80, isrc = "GBAYE8500001")))
        // A server search link is not the song's page: the ISRC search is more exact.
        assertEquals(listOf("https://open.spotify.com/search/isrc%3AGBAYE8500001"),
            Destinations.spotify(Match("a", "b", 80, isrc = "GBAYE8500001", spotifyUrl = "https://open.spotify.com/search/b%20a")))
        assertEquals("https://open.spotify.com/search/%C3%89t%C3%A9%20x", Destinations.spotifySearch("Été x"))
    }

    @Test fun otherDestinations() = runTest {
        assertEquals("https://music.youtube.com/search?q=Take%20On%20Me%20a-ha", Destinations.youTubeMusic(Match("Take On Me", "a-ha", 80)))
        val itunes = Http { url, _, _ ->
            assertTrue(url.startsWith("https://itunes.apple.com/search?"), url)
            HttpResponse(200, """{"results":[
                {"trackName":"Take On Me (Live)","artistName":"a-ha","trackViewUrl":"https://music.apple.com/us/album/live/1?i=2&uo=4"},
                {"trackName":"Take On Me","artistName":"a-ha","trackViewUrl":"https://music.apple.com/us/album/hunting/3?i=4&uo=4"}]}""", url)
        }
        assertEquals("https://music.apple.com/us/album/hunting/3?i=4", Destinations.appleMusic(itunes, Match("Take On Me", "a-ha", 80)))
        val nothing = Http { url, _, _ -> HttpResponse(200, """{"results":[]}""", url) }
        assertEquals("https://music.apple.com/de/search?term=Take%20On%20Me%20a-ha", Destinations.appleMusic(nothing, Match("Take On Me", "a-ha", 80), "de"))
        val offline = Http { _, _, _ -> throw SoundalikeError.Offline() }
        assertEquals("https://music.apple.com/us/album/x/1?i=2",
            Destinations.appleMusic(offline, Match("a", "b", 1, appleMusicUrl = "https://music.apple.com/us/album/x/1?i=2")))
    }

    @Test fun queryEncoding() {
        assertEquals("https://h/p?a=1%2B1&b=x%20%26%20y", urlWithQuery("https://h/p", mapOf("b" to "x & y", "a" to "1+1", "c" to null, "d" to "")))
        assertEquals("https://h/p", urlWithQuery("https://h/p", mapOf("c" to null)))
    }
}
