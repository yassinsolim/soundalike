import Foundation

/// Where tapping "open" on a result goes, without any Spotify or Apple sign-in.
public struct Destinations: Sendable {
    public var http: HTTP

    public init(http: HTTP = HTTP()) {
        self.http = http
    }

    /// Exact track link when the server supplied one; otherwise Spotify search as a
    /// universal link (opens the app when installed, the web player when not).
    /// Searching `isrc:<code>` returns exactly that recording (verified on iOS 26.6,
    /// Spotify 2026-11); title and artist are the fallback when there is no ISRC.
    public static func spotify(_ match: Match) -> URL {
        if let url = match.spotifyURL { return url }
        if let isrc = match.isrc, !isrc.isEmpty { return spotifySearch("isrc:\(isrc)") }
        return spotifySearch("\(match.title) \(match.artist)")
    }

    public static func spotifySearch(_ text: String) -> URL {
        let allowed = CharacterSet(charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
        let path = text.addingPercentEncoding(withAllowedCharacters: allowed) ?? ""
        return URL(string: "https://open.spotify.com/search/\(path)")!
    }

    /// Finds the song in Apple Music with the public iTunes Search API.
    /// The app prefers an on-device MusicKit ISRC lookup; this is the no-MusicKit path.
    public func appleMusic(_ match: Match, storefront: String = "us") async -> URL {
        if let url = match.appleMusicURL { return url }
        if let url = await appleSongURL(match, storefront: storefront) { return url }
        let term = "\(match.title) \(match.artist)".addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? ""
        return URL(string: "https://music.apple.com/\(storefront)/search?term=\(term)")!
    }

    /// The best iTunes Search match for a result, cached per result: the API allows only
    /// about 20 requests a minute.
    func appleSongURL(_ match: Match, storefront: String) async -> URL? {
        let key = "\(storefront)|\(match.id)"
        if let cached = AppleSongCache.shared.song(for: key) { return cached }
        let search = URL(string: "https://itunes.apple.com/search")!.appending(query: [
            "term": "\(TextMatch.coreTitle(match.title)) \(TextMatch.primaryArtist(match.artist))",
            "entity": "song", "limit": "10", "country": storefront,
        ])
        guard let object = (try? await http.json(search, timeout: 8)) as? [String: Any] else { return nil }
        let rows = object["results"] as? [[String: Any]] ?? []
        let best = Self.best(rows, for: match)
        AppleSongCache.shared.store(best, for: key)
        return best
    }

    /// The closest iTunes Search row by title and artist, or nil when none is close enough.
    static func best(_ rows: [[String: Any]], for match: Match) -> URL? {
        rows.compactMap { row -> (Int, URL)? in
            guard let title = row["trackName"] as? String, let artist = row["artistName"] as? String,
                  let raw = row["trackViewUrl"] as? String, let url = URL(string: raw) else { return nil }
            let score = TextMatch.score(title: match.title, artist: match.artist, candidateTitle: title, candidateArtist: artist)
            return score >= 6 ? (score, stripTracking(url)) : nil
        }.max { $0.0 < $1.0 }?.1
    }

    static func stripTracking(_ url: URL) -> URL {
        guard var components = URLComponents(url: url, resolvingAgainstBaseURL: false) else { return url }
        components.queryItems = components.queryItems?.filter { $0.name != "uo" }
        return components.url ?? url
    }
}

/// iTunes Search answers, kept for the app's lifetime; a miss is remembered too.
final class AppleSongCache: @unchecked Sendable {
    static let shared = AppleSongCache()
    private let lock = NSLock()
    private var songs: [String: URL?] = [:]

    func song(for key: String) -> URL?? {
        lock.withLock { songs[key] }
    }

    func store(_ song: URL?, for key: String) {
        lock.withLock { songs[key] = .some(song) }
    }
}
