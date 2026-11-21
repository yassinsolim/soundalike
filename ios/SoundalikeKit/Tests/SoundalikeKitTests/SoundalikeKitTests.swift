import Foundation
import Testing
@testable import SoundalikeKit

@Suite struct SharedLinkTests {
    @Test func spotifyShareTextWithLabel() throws {
        let text = "Blinding Lights by The Weeknd\nhttps://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b?si=abc123"
        #expect(SharedLink.find(in: text) == .spotifyTrack(id: "0VjIjW4GlUZAMYd2vXMi3b", url: URL(string: "https://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b")!))
    }

    @Test func spotifyIntlAndUri() {
        #expect(SharedLink.parse(URL(string: "https://open.spotify.com/intl-de/track/0VjIjW4GlUZAMYd2vXMi3b")!)?.source == .spotify)
        if case .spotifyTrack(let id, _)? = SharedLink.find(in: "spotify:track:0VjIjW4GlUZAMYd2vXMi3b") {
            #expect(id == "0VjIjW4GlUZAMYd2vXMi3b")
        } else { Issue.record("spotify: URI not parsed") }
    }

    @Test func spotifyShortLinkNeedsExpansion() {
        #expect(SharedLink.find(in: "https://spotify.link/AbCdEf123") == .spotifyShort(URL(string: "https://spotify.link/AbCdEf123")!))
    }

    @Test func spotifyAlbumAndPlaylistAreNotSongs() {
        #expect(SharedLink.find(in: "https://open.spotify.com/album/4yP0hdKOZPNshxUOjY0cZj") == nil)
        #expect(SharedLink.find(in: "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M") == nil)
    }

    @Test func appleMusicAlbumLinkWithSongParameter() {
        let url = URL(string: "https://music.apple.com/gb/album/after-hours/1499378108?i=1499378615")!
        #expect(SharedLink.parse(url) == .appleMusicSong(id: "1499378615", storefront: "gb", url: url))
    }

    @Test func appleMusicSongLink() {
        let url = URL(string: "https://music.apple.com/us/song/blinding-lights/1499378615")!
        #expect(SharedLink.parse(url) == .appleMusicSong(id: "1499378615", storefront: "us", url: url))
        #expect(SharedLink.parse(URL(string: "https://music.apple.com/us/album/after-hours/1499378108")!) == nil)
    }
}

@Suite struct ParsingTests {
    func fixture(_ name: String) throws -> Data {
        try Data(contentsOf: #require(Bundle.module.url(forResource: name, withExtension: nil, subdirectory: "Fixtures")))
    }

    @Test func spotifyEmbedPage() throws {
        let html = String(decoding: try fixture("spotify-embed.html"), as: UTF8.self)
        let meta = try #require(SeedResolver.parseSpotifyEmbed(html))
        #expect(meta.title == "Blinding Lights")
        #expect(meta.artist == "The Weeknd")
        #expect(meta.artwork != nil)
    }

    @Test func anysongResponse() throws {
        let seed = Seed(title: "Blinding Lights", artist: "The Weeknd", source: .spotify)
        let results = try SoundalikeClient.parseAnysong(try fixture("anysong.json"), seed: seed)
        #expect(results.matches.count == 3)
        #expect(results.matches[0].deezerID != nil)
        #expect((0...100).contains(results.matches[0].match))
        #expect(results.route == "anysong")
    }

    @Test func anysongNotFound() {
        let seed = Seed(title: "x", artist: "y", source: .search)
        #expect(throws: SoundalikeError.notInCatalog) {
            try SoundalikeClient.parseAnysong(Data(#"{"ok":false,"error":"song not found"}"#.utf8), seed: seed)
        }
    }

    @Test func mobileResponse() throws {
        let body = #"""
        {"ok":true,"query":{"title":"Blinding Lights","artist":"The Weeknd","isrc":"USUG11904206","artwork":"https://x/a.jpg"},
         "results":[{"title":"Take On Me","artist":"a-ha","match":81,"isrc":"GBAYE8500001","deezer_id":664107,
           "spotify_url":"https://open.spotify.com/track/2WfaOiMkCvy7F5fcp2zZ8L","apple_music_url":null,"artwork":"","preview_url":"https://p/x.mp3"}]}
        """#
        let results = try SoundalikeClient.parseMobile(Data(body.utf8), source: .spotify, sharedURL: nil)
        #expect(results.seed.isrc == "USUG11904206")
        let match = try #require(results.matches.first)
        #expect(match.match == 81)
        #expect(match.spotifyURL?.absoluteString == "https://open.spotify.com/track/2WfaOiMkCvy7F5fcp2zZ8L")
        #expect(match.appleMusicURL == nil)
        #expect(match.artworkURL == nil)
        #expect(Destinations.spotify(match) == match.spotifyURL)
    }
}

@Suite struct MatchingTests {
    @Test func variantsStillMatch() {
        #expect(TextMatch.coreTitle("Blinding Lights (Remastered 2020) - Radio Edit") == "blinding lights")
        #expect(TextMatch.coreTitle("Stay (feat. Justin Bieber)") == "stay")
        #expect(TextMatch.primaryArtist("The Kid LAROI, Justin Bieber") == "the kid laroi")
    }

    @Test func liveVersionLosesToStudio() {
        let studio = TextMatch.score(title: "Blinding Lights", artist: "The Weeknd", candidateTitle: "Blinding Lights", candidateArtist: "The Weeknd")
        let live = TextMatch.score(title: "Blinding Lights", artist: "The Weeknd", candidateTitle: "Blinding Lights (Live)", candidateArtist: "The Weeknd")
        #expect(studio > live)
        #expect(TextMatch.score(title: "Blinding Lights", artist: "The Weeknd", candidateTitle: "Hello", candidateArtist: "Adele") == 0)
    }

    @Test func spotifyFallbacksAreUniversalSearchLinks() {
        #expect(Destinations.spotify(Match(title: "Take On Me", artist: "a-ha", match: 80)).absoluteString
            == "https://open.spotify.com/search/Take%20On%20Me%20a-ha")
        #expect(Destinations.spotify(Match(title: "Été", artist: "x", match: 80, isrc: "GBAYE8500001")).absoluteString
            == "https://open.spotify.com/search/isrc%3AGBAYE8500001")
        #expect(Destinations.spotifySearch("Été x").absoluteString == "https://open.spotify.com/search/%C3%89t%C3%A9%20x")
    }
}
