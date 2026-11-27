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

    public init(
        title: String,
        artist: String,
        isrc: String? = nil,
        artworkURL: URL? = nil,
        source: MusicSource,
        sharedURL: URL? = nil,
        appleMusicID: String? = nil,
        storefront: String? = nil,
        spotifyID: String? = nil
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

    public var id: String { deezerID.map(String.init) ?? isrc ?? "\(artist)|\(title)" }

    public init(
        title: String,
        artist: String,
        match: Int,
        isrc: String? = nil,
        deezerID: Int? = nil,
        spotifyURL: URL? = nil,
        appleMusicURL: URL? = nil,
        artworkURL: URL? = nil
    ) {
        self.title = title
        self.artist = artist
        self.match = match
        self.isrc = isrc
        self.deezerID = deezerID
        self.spotifyURL = spotifyURL
        self.appleMusicURL = appleMusicURL
        self.artworkURL = artworkURL
    }
}

public struct SoundalikeResults: Codable, Hashable, Sendable {
    public var seed: Seed
    public var matches: [Match]
    /// Which server route answered, for diagnostics.
    public var route: String

    public init(seed: Seed, matches: [Match], route: String) {
        self.seed = seed
        self.matches = matches
        self.route = route
    }
}

public enum SoundalikeError: LocalizedError, Equatable, Sendable {
    case unsupportedLink
    case songNotIdentified
    case notInCatalog
    case offline
    case server(Int)

    public var errorDescription: String? {
        switch self {
        case .unsupportedLink: "That isn't a song link. Share a single song from Spotify or Apple Music."
        case .songNotIdentified: "Couldn't work out which song that is."
        case .notInCatalog: "Soundalike doesn't know that song yet."
        case .offline: "Couldn't reach Soundalike. Check your connection and try again."
        case .server(let code): "Soundalike had a problem (\(code)). Try again in a moment."
        }
    }
}
