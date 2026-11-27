import Foundation

/// Talks to soundalike.yassin.app.
///
/// Prefers `/api/mobile/similar`, which resolves links server-side and returns
/// per-platform URLs. Until that route is live (404) or if it fails, falls back to
/// `/api/anysong/similar` and fills art and ISRC from Deezer's public API.
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

    /// `part`: match that part of the song's 30-second preview instead of all of it. The answer's `part` is nil
    /// when the server matched the whole preview anyway (a server without parts).
    public func similar(to link: SharedLink, k: Int = 20, part: PartSpan? = nil) async throws -> SoundalikeResults {
        if useMobileEndpoint, let results = try? await mobile(["url": link.url.absoluteString, "platform": platform(link.source)], source: link.source, sharedURL: link.url, k: k, part: part) {
            return results
        }
        let seed = try await resolver.resolve(link)
        return try await anysong(seed, k: k, part: part)
    }

    /// For a song already known by name (search, current Apple Music song, Shazam).
    public func similar(to seed: Seed, k: Int = 20, part: PartSpan? = nil) async throws -> SoundalikeResults {
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
            if let results = try? await mobile(query, source: seed.source, sharedURL: seed.sharedURL, k: k, part: part) {
                return results
            }
        }
        return try await anysong(seed, k: k, part: part)
    }

    /// Soundalikes for a clip of the listener's own audio (`POST /api/anysong/excerpt`): a WAV of 10 to 30 s from
    /// any part of `seed`. The song keeps the seed's identity (its language, artist and links); the audio is the
    /// clip's. Only offered when an answer carried `parts.excerpt`.
    public func similar(toClip wav: Data, seed: Seed, k: Int = 20) async throws -> SoundalikeResults {
        var query: [String: String?] = ["k": String(k), "lang": "same"]
        if let id = seed.deezerID {
            query["id"] = String(id)
        } else if let isrc = seed.isrc, !isrc.isEmpty {
            query["isrc"] = isrc
        } else {
            query["title"] = seed.title
            query["artist"] = seed.artist
        }
        let url = base.appendingPathComponent("api/anysong/excerpt").appending(query: query)
        var headers = headers
        headers["Content-Type"] = "audio/wav"
        let (data, response) = try await http.post(url, body: wav, timeout: 75, headers: headers)
        if let error = Self.excerptError(status: response.statusCode, retryAfter: response.value(forHTTPHeaderField: "Retry-After"), body: data) {
            throw error
        }
        var results = try Self.parseAnysong(data, seed: seed)
        results.route = "excerpt"
        results.matches = await DeezerCatalog(http: http).enrich(results.matches)
        return results
    }

    /// A fresh address for a song's 30-second preview, from `/api/preview` (Deezer's expire after about 15 minutes).
    public func previewURL(deezerID: Int) async throws -> URL {
        let url = base.appendingPathComponent("api/preview").appending(query: ["id": String(deezerID)])
        let (data, response) = try await http.get(url, timeout: 15, headers: headers)
        guard response.statusCode == 200 else {
            throw response.statusCode == 404 ? SoundalikeError.notInCatalog : SoundalikeError.server(response.statusCode)
        }
        guard let preview = Self.parsePreview(data) else { throw SoundalikeError.server(200) }
        return preview
    }

    /// `{"ok": true, "preview": "<mp3>", "cover": "<jpg>"}`
    public static func parsePreview(_ data: Data) -> URL? {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any], object["ok"] as? Bool == true else { return nil }
        return url(object["preview"])
    }

    private func platform(_ source: MusicSource) -> String {
        source == .spotify ? "spotify" : "apple"
    }

    // MARK: /api/mobile/similar

    func mobile(_ query: [String: String?], source: MusicSource, sharedURL: URL?, k: Int, part: PartSpan? = nil) async throws -> SoundalikeResults {
        var query = query
        query["k"] = String(k)
        query["part"] = part?.query
        let url = base.appendingPathComponent("api/mobile/similar").appending(query: query)
        let (data, response) = try await http.get(url, timeout: 30, headers: headers)
        // A server that refuses the part is asked once more for the whole preview.
        if part != nil, [400, 404].contains(response.statusCode) {
            return try await mobile(query, source: source, sharedURL: sharedURL, k: k, part: nil)
        }
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
            source: source, sharedURL: sharedURL, deezerID: int(query["deezer_id"])
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
        return SoundalikeResults(seed: seed, matches: matches, route: "mobile", parts: parseParts(query["parts"]), part: parsePart(query["part"]))
    }

    // MARK: /api/anysong/similar

    func anysong(_ seed: Seed, k: Int, part: PartSpan? = nil) async throws -> SoundalikeResults {
        let url = base.appendingPathComponent("api/anysong/similar").appending(query: [
            "title": seed.title, "artist": seed.artist, "isrc": seed.isrc, "k": String(k), "lang": "same", "part": part?.query,
        ])
        let (data, response) = try await http.get(url, timeout: 60, headers: headers)
        if part != nil, [400, 404].contains(response.statusCode) { return try await anysong(seed, k: k, part: nil) }
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
        let query = object["query"] as? [String: Any]
        if let query {
            // Keep the user's own spelling unless we had none.
            if resolved.title.isEmpty { resolved.title = query["title"] as? String ?? "" }
            if resolved.artist.isEmpty { resolved.artist = query["artist"] as? String ?? "" }
            if resolved.deezerID == nil, let id = int(query["id"]), id > 0 { resolved.deezerID = id }
        }
        return SoundalikeResults(seed: resolved, matches: matches, route: "anysong", parts: parseParts(query?["parts"]), part: parsePart(query?["part"]))
    }

    // MARK: Parts

    /// `query.parts`, or nil from a server that can't match a part.
    static func parseParts(_ value: Any?) -> PartLimits? {
        guard let parts = value as? [String: Any] else { return nil }
        var limits = PartLimits(
            minSeconds: double(parts["min_s"]) ?? 10, maxSeconds: double(parts["max_s"]) ?? 30,
            previewSeconds: double(parts["preview_s"]) ?? 30, songOffset: double(parts["song_offset_s"])
        )
        if let excerpt = parts["excerpt"] as? [String: Any], let maxBytes = int(excerpt["max_bytes"]) {
            limits.excerpt = ExcerptLimits(minSeconds: double(excerpt["min_s"]) ?? 10, maxSeconds: double(excerpt["max_s"]) ?? 30, maxBytes: maxBytes)
        }
        return limits
    }

    /// `query.part` (`{"start", "end", "heard"}`), nil for the whole preview.
    static func parsePart(_ value: Any?) -> PartSpan? {
        guard let part = value as? [String: Any], let start = double(part["start"]), let end = double(part["end"]), end > start else { return nil }
        return PartSpan(start: start, end: end)
    }

    /// What a refused clip means to the listener; nil for 200.
    public static func excerptError(status: Int, retryAfter: String?, body: Data) -> SoundalikeError? {
        let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any]
        let message = (object?["error"] as? String) ?? ((object?["error"] as? [String: Any])?["message"] as? String) ?? ""
        let after = retryAfter.flatMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        switch status {
        case 200: return nil
        case 404 where message == "unknown track":
            return .clip("Soundalike couldn't find this song to match your clip against.")
        case 404: return .clip("Matching your own clip isn't available right now.")
        case 400, 411: return .clip("Soundalike couldn't take that clip. Try again.")
        case 408: return .clip("Your clip took too long to upload. Check your connection and try again.")
        case 413: return .clip("That clip is too large. Pick a shorter part.")
        case 415: return .clip("Soundalike can't read that kind of audio.")
        case 422 where message.contains("silent"): return .clip("Your clip is silent. Turn the music up and try again.")
        case 422: return .clip("Soundalike couldn't use that clip. It needs 10 to 30 seconds of music.")
        case 429: return .rateLimited(retryAfter: after)
        case 503: return .busy(retryAfter: after)
        default: return .server(status)
        }
    }

    static func int(_ value: Any?) -> Int? {
        if let value = value as? Int { return value }
        if let value = value as? Double { return Int(value.rounded()) }
        if let value = value as? String { return Int(value) }
        return nil
    }

    static func double(_ value: Any?) -> Double? {
        if let value = value as? Double { return value }
        if let value = value as? Int { return Double(value) }
        if let value = value as? NSNumber { return value.doubleValue }
        return nil
    }

    static func url(_ value: Any?) -> URL? {
        (value as? String).flatMap { $0.isEmpty ? nil : URL(string: $0) }
    }
}
