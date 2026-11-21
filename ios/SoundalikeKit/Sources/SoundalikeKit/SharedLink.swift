import Foundation

/// A song link found in whatever a share sheet handed over.
public enum SharedLink: Hashable, Sendable {
    case spotifyTrack(id: String, url: URL)
    /// spotify.link short links hide the track id behind a redirect.
    case spotifyShort(URL)
    case appleMusicSong(id: String, storefront: String, url: URL)

    public var url: URL {
        switch self {
        case .spotifyTrack(_, let url), .spotifyShort(let url), .appleMusicSong(_, _, let url): url
        }
    }

    public var source: MusicSource {
        switch self {
        case .spotifyTrack, .spotifyShort: .spotify
        case .appleMusicSong: .appleMusic
        }
    }

    /// Finds the first song link in free text, for example "Song by Artist https://open.spotify.com/track/…".
    public static func find(in text: String) -> SharedLink? {
        if let id = firstMatch(#"spotify:track:([A-Za-z0-9]{22})"#, in: text) {
            return .spotifyTrack(id: id, url: URL(string: "https://open.spotify.com/track/\(id)")!)
        }
        let detector = try? NSDataDetector(types: NSTextCheckingResult.CheckingType.link.rawValue)
        let range = NSRange(text.startIndex..., in: text)
        for result in detector?.matches(in: text, range: range) ?? [] {
            if let url = result.url, let link = parse(url) { return link }
        }
        return nil
    }

    public static func parse(_ url: URL) -> SharedLink? {
        if url.scheme == "spotify" {
            return find(in: url.absoluteString)
        }
        guard let host = url.host()?.lowercased() else { return nil }
        let parts = url.pathComponents.filter { $0 != "/" }

        if host == "open.spotify.com" || host == "play.spotify.com" {
            // /track/<id>, /intl-de/track/<id>, /embed/track/<id>
            guard let index = parts.firstIndex(of: "track"), index + 1 < parts.count else { return nil }
            let id = parts[index + 1]
            guard id.count == 22, id.allSatisfy({ $0.isASCII && ($0.isLetter || $0.isNumber) }) else { return nil }
            return .spotifyTrack(id: id, url: URL(string: "https://open.spotify.com/track/\(id)")!)
        }
        if host == "spotify.link" || host == "link.tospotify.com" || host == "spotify.app.link" {
            return parts.isEmpty ? nil : .spotifyShort(url)
        }
        if host.hasSuffix("music.apple.com") || host == "itunes.apple.com" {
            let storefront = parts.first.flatMap { $0.count == 2 ? $0.lowercased() : nil } ?? "us"
            let query = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? []
            // Album links name the song with ?i=<songId>.
            if let id = query.first(where: { $0.name == "i" })?.value, isNumeric(id) {
                return .appleMusicSong(id: id, storefront: storefront, url: url)
            }
            if parts.contains("song"), let id = parts.last, isNumeric(id) {
                return .appleMusicSong(id: id, storefront: storefront, url: url)
            }
            return nil
        }
        return nil
    }

    private static func isNumeric(_ value: String) -> Bool {
        !value.isEmpty && value.allSatisfy(\.isASCII) && value.allSatisfy(\.isNumber)
    }

    private static func firstMatch(_ pattern: String, in text: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let range = NSRange(text.startIndex..., in: text)
        guard let match = regex.firstMatch(in: text, range: range), match.numberOfRanges > 1,
              let capture = Range(match.range(at: 1), in: text) else { return nil }
        return String(text[capture])
    }
}
