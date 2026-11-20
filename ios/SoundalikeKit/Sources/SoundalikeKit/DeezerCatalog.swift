import Foundation

/// Deezer's public, key-free API: cover art, ISRC and 30-second previews for the
/// Deezer track ids the recommender returns, plus song search for the search box.
public struct DeezerCatalog: Sendable {
    public var http: HTTP

    public init(http: HTTP = HTTP()) {
        self.http = http
    }

    public struct Track: Sendable {
        public var isrc: String?
        public var artwork: URL?
        public var preview: URL?
    }

    public func track(_ id: Int) async -> Track? {
        guard let object = try? await http.json(URL(string: "https://api.deezer.com/track/\(id)")!, timeout: 10) as? [String: Any],
              object["error"] == nil else { return nil }
        let album = object["album"] as? [String: Any]
        return Track(
            isrc: object["isrc"] as? String,
            artwork: SoundalikeClient.url(album?["cover_big"] ?? album?["cover_medium"]),
            preview: SoundalikeClient.url(object["preview"])
        )
    }

    /// Fills missing art, ISRC and preview, a few requests at a time. Order is kept.
    public func enrich(_ matches: [Match], concurrency: Int = 6) async -> [Match] {
        var enriched = matches
        await withTaskGroup(of: (Int, Track?).self) { group in
            var next = 0
            func schedule() {
                while next < matches.count {
                    let index = next
                    next += 1
                    let match = matches[index]
                    guard let id = match.deezerID, match.artworkURL == nil || match.previewURL == nil || match.isrc == nil else { continue }
                    group.addTask { (index, await track(id)) }
                    return
                }
            }
            for _ in 0..<concurrency { schedule() }
            while let (index, track) = await group.next() {
                if let track {
                    enriched[index].isrc = enriched[index].isrc ?? track.isrc
                    enriched[index].artworkURL = enriched[index].artworkURL ?? track.artwork
                    enriched[index].previewURL = enriched[index].previewURL ?? track.preview
                }
                schedule()
            }
        }
        return enriched
    }

    /// Song search for the in-app search box.
    public func search(_ text: String, limit: Int = 12) async throws -> [Seed] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        let url = URL(string: "https://api.deezer.com/search")!.appending(query: ["q": String(trimmed.prefix(120)), "limit": String(limit)])
        let rows = (try await http.json(url, timeout: 10) as? [String: Any])?["data"] as? [[String: Any]] ?? []
        var seen = Set<String>()
        return rows.compactMap { row in
            guard let title = row["title"] as? String, let artist = (row["artist"] as? [String: Any])?["name"] as? String else { return nil }
            let key = TextMatch.normalize(title) + "|" + TextMatch.normalize(artist)
            guard seen.insert(key).inserted else { return nil }
            let album = row["album"] as? [String: Any]
            return Seed(title: title, artist: artist, artworkURL: SoundalikeClient.url(album?["cover_medium"]), source: .search)
        }
    }
}
