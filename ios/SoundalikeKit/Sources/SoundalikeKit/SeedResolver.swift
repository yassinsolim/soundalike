import Foundation

/// Turns a shared link into a `Seed` using public, sign-in-free pages:
/// Spotify's embed page and Apple's iTunes lookup API.
public struct SeedResolver: Sendable {
    public var http: HTTP

    public init(http: HTTP = HTTP()) {
        self.http = http
    }

    public func resolve(_ link: SharedLink) async throws -> Seed {
        switch link {
        case .spotifyShort(let url):
            guard let expanded = try await expandShortLink(url) else { throw SoundalikeError.unsupportedLink }
            return try await resolve(expanded)
        case .spotifyTrack(let id, let url):
            return try await spotifySeed(id: id, url: url)
        case .appleMusicSong(let id, let storefront, let url):
            return try await appleMusicSeed(id: id, storefront: storefront, url: url)
        }
    }

    func expandShortLink(_ url: URL) async throws -> SharedLink? {
        let (data, response) = try await http.get(url, timeout: 10)
        if let finalURL = response.url, let link = SharedLink.parse(finalURL), case .spotifyTrack = link {
            return link
        }
        // Branch.io pages sometimes stop at an HTML hop that embeds the target.
        let html = String(decoding: data, as: UTF8.self)
        if case .spotifyTrack(let id, let target)? = SharedLink.find(in: html) {
            return .spotifyTrack(id: id, url: target)
        }
        return nil
    }

    func spotifySeed(id: String, url: URL) async throws -> Seed {
        let embed = URL(string: "https://open.spotify.com/embed/track/\(id)")!
        let (data, response) = try await http.get(embed, timeout: 10)
        guard response.statusCode == 200, let meta = Self.parseSpotifyEmbed(String(decoding: data, as: UTF8.self)) else {
            throw SoundalikeError.songNotIdentified
        }
        return Seed(
            title: meta.title, artist: meta.artist, artworkURL: meta.artwork,
            source: .spotify, sharedURL: url, spotifyID: id
        )
    }

    /// Reads `__NEXT_DATA__` from https://open.spotify.com/embed/track/<id>.
    public static func parseSpotifyEmbed(_ html: String) -> (title: String, artist: String, artwork: URL?)? {
        guard let start = html.range(of: #"<script id="__NEXT_DATA__""#),
              let open = html.range(of: ">", range: start.upperBound..<html.endIndex),
              let close = html.range(of: "</script>", range: open.upperBound..<html.endIndex),
              let object = try? JSONSerialization.jsonObject(with: Data(html[open.upperBound..<close.lowerBound].utf8)),
              let entity = ((((object as? [String: Any])?["props"] as? [String: Any])?["pageProps"] as? [String: Any])?["state"] as? [String: Any])
                .flatMap({ ($0["data"] as? [String: Any])?["entity"] as? [String: Any] }),
              let title = entity["name"] as? String ?? entity["title"] as? String,
              let artist = (entity["artists"] as? [[String: Any]])?.first?["name"] as? String
        else { return nil }
        let images = ((entity["visualIdentity"] as? [String: Any])?["image"] as? [[String: Any]]) ?? []
        let sorted = images.compactMap { image -> (Int, URL)? in
            guard let raw = image["url"] as? String, let url = URL(string: raw) else { return nil }
            return (image["maxWidth"] as? Int ?? 0, url)
        }.sorted { $0.0 < $1.0 }
        let artwork = (sorted.first { $0.0 >= 300 } ?? sorted.last)?.1
        return (title, artist, artwork)
    }

    func appleMusicSeed(id: String, storefront: String, url: URL) async throws -> Seed {
        let lookup = URL(string: "https://itunes.apple.com/lookup")!.appending(query: ["id": id, "country": storefront, "entity": "song"])
        let object = try await http.json(lookup, timeout: 10)
        guard let song = ((object as? [String: Any])?["results"] as? [[String: Any]])?.first(where: { $0["wrapperType"] as? String == "track" }),
              let title = song["trackName"] as? String, let artist = song["artistName"] as? String
        else { throw SoundalikeError.songNotIdentified }
        return Seed(
            title: title, artist: artist,
            artworkURL: (song["artworkUrl100"] as? String).flatMap { URL(string: $0.replacingOccurrences(of: "100x100bb", with: "600x600bb")) },
            source: .appleMusic, sharedURL: url, appleMusicID: id, storefront: storefront
        )
    }
}
