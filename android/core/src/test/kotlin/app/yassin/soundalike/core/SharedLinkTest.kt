package app.yassin.soundalike.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class SharedLinkTest {
    private val weeknd = SharedLink.SpotifyTrack("0VjIjW4GlUZAMYd2vXMi3b")

    @Test fun spotifyShareTextWithLabel() {
        val text = "Blinding Lights by The Weeknd\nhttps://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b?si=abc123"
        assertEquals(weeknd, SharedLink.find(text))
        assertEquals("https://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b", weeknd.url)
    }

    @Test fun spotifyLinkShapes() {
        for (link in listOf(
            "https://open.spotify.com/intl-de/track/0VjIjW4GlUZAMYd2vXMi3b",
            "https://open.spotify.com/embed/track/0VjIjW4GlUZAMYd2vXMi3b?utm_source=generator",
            "http://play.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b",
            "spotify:track:0VjIjW4GlUZAMYd2vXMi3b",
            "Listen: https://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b.",
            "https://open.spotify.com/album/4yP0hdKOZPNshxUOjY0cZj?highlight=spotify:track:0VjIjW4GlUZAMYd2vXMi3b",
        )) assertEquals(weeknd, SharedLink.find(link), link)
    }

    @Test fun spotifyShortLinksNeedExpansion() {
        assertEquals(SharedLink.SpotifyShort("https://spotify.link/AbCdEf123"), SharedLink.find("https://spotify.link/AbCdEf123"))
        assertEquals(MusicSource.SPOTIFY, SharedLink.find("https://spotify.app.link/xyz?_p=1")?.source)
        assertNull(SharedLink.find("https://spotify.link/"))
    }

    @Test fun thingsThatAreNotSongs() {
        for (link in listOf(
            "https://open.spotify.com/album/4yP0hdKOZPNshxUOjY0cZj",
            "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M",
            "https://open.spotify.com/track/tooShort",
            "spotify:track:0VjIjW4GlUZAMYd2vXMi3bXX",
            "https://music.apple.com/us/album/after-hours/1499378108",
            "https://music.youtube.com/playlist?list=OLAK5uy_abc",
            "https://www.youtube.com/@TheWeeknd",
            "https://example.com/track/0VjIjW4GlUZAMYd2vXMi3b",
            "just some words",
            "",
        )) assertNull(SharedLink.find(link), link)
    }

    @Test fun appleMusicLinks() {
        val album = "https://music.apple.com/gb/album/after-hours/1499378108?i=1499378615"
        assertEquals(SharedLink.AppleMusicSong("1499378615", "gb", album), SharedLink.parse(album))
        val song = "https://music.apple.com/us/song/blinding-lights/1499378615"
        assertEquals(SharedLink.AppleMusicSong("1499378615", "us", song), SharedLink.parse(song))
        val noStorefront = "https://geo.music.apple.com/album/after-hours/1499378108?i=1499378615&app=music"
        assertEquals(SharedLink.AppleMusicSong("1499378615", "us", noStorefront), SharedLink.parse(noStorefront))
        assertEquals("1499378615", (SharedLink.parse("https://itunes.apple.com/de/album/x/1?i=1499378615&uo=4") as SharedLink.AppleMusicSong).id)
    }

    @Test fun youTubeLinks() {
        val video = SharedLink.YouTubeVideo("fHI8X4OXluQ")
        for (link in listOf(
            "https://music.youtube.com/watch?v=fHI8X4OXluQ&si=AbC-dEf_123",
            "https://music.youtube.com/watch?v=fHI8X4OXluQ&list=RDAMVMfHI8X4OXluQ",
            "https://www.youtube.com/watch?v=fHI8X4OXluQ",
            "https://m.youtube.com/watch?feature=share&v=fHI8X4OXluQ",
            "https://youtu.be/fHI8X4OXluQ?si=xyz",
        )) assertEquals(video, SharedLink.find(link), link)
        assertEquals(MusicSource.YOUTUBE_MUSIC, video.source)
    }
}
