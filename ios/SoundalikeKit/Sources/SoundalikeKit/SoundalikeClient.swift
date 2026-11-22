import Foundation

/// Talks to soundalike.yassin.app.
///
/// Prefers `/api/mobile/similar`, which resolves links server-side and returns
/// per-platform URLs. Until that route is live (404) or if it fails, falls back to
/// `/api/anysong/similar` and fills art, ISRC and previews from Deezer's public API.
public struct SoundalikeClient: Sendable {
    public static let defaultBase = URL(string: "https://soundalike.yassin.app")!

    public var base: URL
    public var useMobileEndpoint: Bool
    public var http: HTTP
    public var resolver: SeedResolver

    public init(base: URL = SoundalikeClient.defaultBase, useMobileEndpoint: Bool = true, http: HTTP = HTTP()) {
        self.base = base
        self.useMobileEndpoint = useMobileEndpoint
        self.http = http
        self.resolver = SeedResolver(http: http)
    }

    private var headers: [String: String] {
        ["Origin": "https://soundalike.yassin.app", "Accept": "application/json"]
    }

    // MARK: Entry points

    /// Shared text or URL in, soundalikes out.
    public func similar(toShared text: String, k: Int = 20) async throws -> SoundalikeResults {
        guard let link = SharedLink.find(in: text) else { throw SoundalikeError.unsupportedLink }
        return try await similar(to: link, k: k)
    }

    public func similar(to link: SharedLink, k: Int = 20) async throws -> SoundalikeResults {
        if useMobileEndpoint, let results = try? await mobile(["url": link.url.absoluteString, "platform": platform(link.source)], source: link.source, sharedURL: link.url, k: k) {
            return results
        }
        let seed = try await resolver.resolve(link)
        return try await anysong(seed, k: k)
    }

    /// For a song already known by name (search, current Apple Music song, Shazam).
    public func similar(to seed: Seed, k: Int = 20) async throws -> SoundalikeResults {
        if useMobileEndpoint {
            var query: [String: String?] = ["platform": platform(seed.source)]
            if let id = seed.appleMusicID {
                query["apple_id"] = id
                query["storefront"] = seed.storefront ?? "us"
            } else if let id = seed.spotifyID {
                query["spotify_id"] = id
            }
            query["isrc"] = seed.isrc
            query["title"] = seed.title
            query["artist"] = seed.artist
            if let results = try? await mobile(query, source: seed.source, sharedURL: seed.sharedURL, k: k) {
                return results
            }
        }
        return try await anysong(seed, k: k)
    }

    private func platform(_ source: MusicSource) -> String {
        source == .spotify ? "spotify" : "apple"
    }

    // MARK: /api/mobile/similar

    func mobile(_ query: [String: String?], source: MusicSource, sharedURL: URL?, k: Int) async throws -> SoundalikeResults {
        var query = query
        query["k"] = String(k)
        let url = base.appendingPathComponent("api/mobile/similar").appending(query: query)
        let (data, response) = try await http.get(url, timeout: 30, headers: headers)
        // Any failure here (including 404 while the route is not deployed) makes callers fall back to anysong.
        guard response.statusCode == 200 else { throw SoundalikeError.server(response.statusCode) }
        return try Self.parseMobile(data, source: source, sharedURL: sharedURL)
    }

    public static func parseMobile(_ data: Data, source: MusicSource, sharedURL: URL?) throws -> SoundalikeResults {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any], object["ok"] as? Bool == true,
              let query = object["query"] as? [String: Any], let rows = object["results"] as? [[String: Any]]
        else { throw SoundalikeError.server(200) }
        let seed = Seed(
            title: query["title"] as? String ?? "", artist: query["artist"] as? String ?? "",
            isrc: query["isrc"] as? String, artworkURL: url(query["artwork"]),
            source: source, sharedURL: sharedURL
        )
        let matches = rows.compactMap { row -> Match? in
            guard let title = row["title"] as? String, let artist = row["artist"] as? String else { return nil }
            // Without an exact match the server sends an Apple Music search link (`apple_exact` false).
            // Dropping it lets the on-device MusicKit or iTunes lookup find the song itself.
            let appleExact = row["apple_exact"] as? Bool ?? true
            return Match(
                title: title, artist: artist, match: int(row["match"]) ?? 0,
                isrc: row["isrc"] as? String, deezerID: int(row["deezer_id"]),
                spotifyURL: url(row["spotify_url"]), appleMusicURL: appleExact ? url(row["apple_music_url"]) : nil,
                artworkURL: url(row["artwork"]), previewURL: url(row["preview_url"])
            )
        }
        guard !matches.isEmpty else { throw SoundalikeError.notInCatalog }
        return SoundalikeResults(seed: seed, matches: matches, route: "mobile")
    }

    // MARK: /api/anysong/similar

    func anysong(_ seed: Seed, k: Int) async throws -> SoundalikeResults {
        let url = base.appendingPathComponent("api/anysong/similar").appending(query: [
            "title": seed.title, "artist": seed.artist, "isrc": seed.isrc, "k": String(k), "lang": "same",
        ])
        let (data, response) = try await http.get(url, timeout: 60, headers: headers)
        if response.statusCode == 404 { throw SoundalikeError.notInCatalog }
        guard response.statusCode == 200 else { throw SoundalikeError.server(response.statusCode) }
        var results = try Self.parseAnysong(data, seed: seed)
        results.matches = await DeezerCatalog(http: http).enrich(results.matches)
        return results
    }

    public static func parseAnysong(_ data: Data, seed: Seed) throws -> SoundalikeResults {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw SoundalikeError.server(200) }
        guard object["ok"] as? Bool == true else { throw SoundalikeError.notInCatalog }
        let rows = (object["results"] as? [String: Any])?["main"] as? [[String: Any]] ?? []
        let matches = rows.compactMap { row -> Match? in
            guard let title = row["title"] as? String, let artist = row["artist"] as? String else { return nil }
            return Match(title: title, artist: artist, match: int(row["match"]) ?? 0, deezerID: int(row["id"]))
        }
        guard !matches.isEmpty else { throw SoundalikeError.notInCatalog }
        var resolved = seed
        if let query = object["query"] as? [String: Any] {
            // Keep the user's own spelling unless we had none.
            if resolved.title.isEmpty { resolved.title = query["title"] as? String ?? "" }
            if resolved.artist.isEmpty { resolved.artist = query["artist"] as? String ?? "" }
        }
        return SoundalikeResults(seed: resolved, matches: matches, route: "anysong")
    }

    static func int(_ value: Any?) -> Int? {
        if let value = value as? Int { return value }
        if let value = value as? Double { return Int(value.rounded()) }
        if let value = value as? String { return Int(value) }
        return nil
    }

    static func url(_ value: Any?) -> URL? {
        (value as? String).flatMap { $0.isEmpty ? nil : URL(string: $0) }
    }
}
