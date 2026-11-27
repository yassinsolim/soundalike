import Foundation

/// The music app a request came from. Results open back in the same app.
public enum MusicSource: String, Codable, Sendable, CaseIterable {
    case spotify
    case appleMusic = "apple"
    case search

    public var displayName: String {
        switch self {
        case .spotify: "Spotify"
        case .appleMusic: "Apple Music"
        case .search: "Search"
        }
    }

    private static func after(_ seconds: Int?) -> String {
        guard let seconds, seconds > 0 else { return "in a moment" }
        return seconds == 1 ? "in a second" : seconds < 90 ? "in \(seconds) seconds" : "in \((seconds + 59) / 60) minutes"
    }
}

/// The song the user asked about.
public struct Seed: Codable, Hashable, Sendable {
    public var title: String
    public var artist: String
    public var isrc: String?
    public var artworkURL: URL?
    public var source: MusicSource
    /// The link the user shared, when there was one. Sent to the server resolver as is.
    public var sharedURL: URL?
    public var appleMusicID: String?
    public var storefront: String?
    public var spotifyID: String?
    /// The Deezer recording the server ranked from, once an answer named it. A clip of the song is sent with it.
    public var deezerID: Int?

    public init(
        title: String,
        artist: String,
        isrc: String? = nil,
        artworkURL: URL? = nil,
        source: MusicSource,
        sharedURL: URL? = nil,
        appleMusicID: String? = nil,
        storefront: String? = nil,
        spotifyID: String? = nil,
        deezerID: Int? = nil
    ) {
        self.title = title
        self.artist = artist
        self.isrc = isrc
        self.artworkURL = artworkURL
        self.source = source
        self.sharedURL = sharedURL
        self.appleMusicID = appleMusicID
        self.storefront = storefront
        self.spotifyID = spotifyID
        self.deezerID = deezerID
    }
}

/// One soundalike.
public struct Match: Codable, Hashable, Sendable, Identifiable {
    public var title: String
    public var artist: String
    /// 0...100, as the server reports it.
    public var match: Int
    public var isrc: String?
    public var deezerID: Int?
    public var spotifyURL: URL?
    public var appleMusicURL: URL?
    public var artworkURL: URL?
    /// Deezer's 30-second MP3. Its address expires about 15 minutes after the answer;
    /// `SoundalikeClient.previewURL(deezerID:)` gets a new one.
    public var previewURL: URL?

    public var id: String { deezerID.map(String.init) ?? isrc ?? "\(artist)|\(title)" }

    public init(
        title: String,
        artist: String,
        match: Int,
        isrc: String? = nil,
        deezerID: Int? = nil,
        spotifyURL: URL? = nil,
        appleMusicURL: URL? = nil,
        artworkURL: URL? = nil,
        previewURL: URL? = nil
    ) {
        self.title = title
        self.artist = artist
        self.match = match
        self.isrc = isrc
        self.deezerID = deezerID
        self.spotifyURL = spotifyURL
        self.appleMusicURL = appleMusicURL
        self.artworkURL = artworkURL
        self.previewURL = previewURL
    }
}

public struct SoundalikeResults: Codable, Hashable, Sendable {
    public var seed: Seed
    public var matches: [Match]
    /// Which server route answered, for diagnostics.
    public var route: String
    /// What part of the song the server can match (`query.parts`); nil from a server without parts.
    public var parts: PartLimits?
    /// The part of the preview these lists are for (`query.part`); nil for the whole preview.
    public var part: PartSpan?
    /// Set when these lists are for a clip of the listener's own audio.
    public var clip: ClipInfo?

    public init(seed: Seed, matches: [Match], route: String, parts: PartLimits? = nil, part: PartSpan? = nil, clip: ClipInfo? = nil) {
        self.seed = seed
        self.matches = matches
        self.route = route
        self.parts = parts
        self.part = part
        self.clip = clip
    }
}

/// `query.parts`: the shortest and longest part of the preview, its length, and where it starts in the song.
public struct PartLimits: Codable, Hashable, Sendable {
    public var minSeconds: Double
    public var maxSeconds: Double
    public var previewSeconds: Double
    /// Where the preview starts in the song; nil when the server doesn't know.
    public var songOffset: Double?
    /// Present when the server also takes a clip of the listener's own audio.
    public var excerpt: ExcerptLimits?

    public init(minSeconds: Double = 10, maxSeconds: Double = 30, previewSeconds: Double = 30, songOffset: Double? = nil, excerpt: ExcerptLimits? = nil) {
        self.minSeconds = minSeconds
        self.maxSeconds = maxSeconds
        self.previewSeconds = previewSeconds
        self.songOffset = songOffset
        self.excerpt = excerpt
    }
}

/// `query.parts.excerpt`: what `POST /api/anysong/excerpt` takes.
public struct ExcerptLimits: Codable, Hashable, Sendable {
    public var minSeconds: Double
    public var maxSeconds: Double
    public var maxBytes: Int

    public init(minSeconds: Double, maxSeconds: Double, maxBytes: Int) {
        self.minSeconds = minSeconds
        self.maxSeconds = maxSeconds
        self.maxBytes = maxBytes
    }
}

/// Seconds of the 30-second preview.
public struct PartSpan: Codable, Hashable, Sendable {
    public var start: Double
    public var end: Double

    public init(start: Double, end: Double) {
        self.start = start
        self.end = end
    }

    public var length: Double { end - start }

    /// `part=<start>-<end>`, seconds with at most one decimal.
    public var query: String { "\(Self.seconds(start))-\(Self.seconds(end))" }

    static func seconds(_ value: Double) -> String {
        let tenths = (value * 10).rounded()
        return tenths.truncatingRemainder(dividingBy: 10) == 0 ? String(Int(tenths / 10)) : String(format: "%.1f", tenths / 10)
    }
}

/// A clip of the listener's own audio the lists were matched from. Nothing of the audio is kept.
public struct ClipInfo: Codable, Hashable, Sendable {
    public enum Source: String, Codable, Sendable {
        case recording
        case file
    }

    public var source: Source
    /// Seconds of the recording or file.
    public var start: Double
    public var end: Double

    public init(source: Source, start: Double, end: Double) {
        self.source = source
        self.start = start
        self.end = end
    }
}

/// m:ss
public func timeLabel(_ seconds: Double) -> String {
    let total = max(0, Int(seconds.rounded()))
    return "\(total / 60):" + String(format: "%02d", total % 60)
}

public enum SoundalikeError: LocalizedError, Equatable, Sendable {
    case unsupportedLink
    case songNotIdentified
    case notInCatalog
    case offline
    case server(Int)
    /// Too many requests; try again after the given seconds.
    case rateLimited(retryAfter: Int?)
    /// The server is busy with another clip; try again after the given seconds.
    case busy(retryAfter: Int?)
    /// A clip the server refused, with a message fit to show.
    case clip(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedLink: "That isn't a song link. Share a single song from Spotify or Apple Music."
        case .songNotIdentified: "Couldn't work out which song that is."
        case .notInCatalog: "Soundalike doesn't know that song yet."
        case .offline: "Couldn't reach Soundalike. Check your connection and try again."
        case .server(let code): "Soundalike had a problem (\(code)). Try again in a moment."
        case .rateLimited(let after): "That's a lot of clips at once. Try again \(Self.after(after))."
        case .busy(let after): "Soundalike is matching another clip. Try again \(Self.after(after))."
        case .clip(let message): message
        }
    }

    private static func after(_ seconds: Int?) -> String {
        guard let seconds, seconds > 0 else { return "in a moment" }
        return seconds == 1 ? "in a second" : seconds < 90 ? "in \(seconds) seconds" : "in \((seconds + 59) / 60) minutes"
    }
}
