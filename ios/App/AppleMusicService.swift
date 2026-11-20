import MusicKit
import SoundalikeKit
import SoundalikeUI
import UIKit

/// Apple Music features: the current song, ISRC matching, Play Next, playlists.
/// All catalog matching happens on the device with MusicKit.
@MainActor
final class AppleMusicService {
    static let shared = AppleMusicService()

    enum Failure: LocalizedError {
        case notAuthorized
        case nothingPlaying
        case notInAppleMusic
        case needsSubscription

        var errorDescription: String? {
            switch self {
            case .notAuthorized: "Allow Soundalike to use Apple Music in Settings."
            case .nothingPlaying: "Nothing is playing in the Music app."
            case .notInAppleMusic: "That song isn't in Apple Music."
            case .needsSubscription: "Playing songs in the Music app needs an Apple Music subscription."
            }
        }
    }

    private var songCache: [String: Song] = [:]

    func authorize() async throws {
        switch MusicAuthorization.currentStatus {
        case .authorized: return
        case .notDetermined:
            guard await MusicAuthorization.request() == .authorized else { throw Failure.notAuthorized }
        default: throw Failure.notAuthorized
        }
    }

    /// What the Music app is playing right now (asks for access the first time).
    func currentSeed() async throws -> Seed {
        try await authorize()
        guard let seed = await NowPlaying.appleMusic(requireAuthorization: true) else { throw Failure.nothingPlaying }
        return seed
    }


    /// The Apple Music catalog song for a result: by ISRC first, then by name.
    func song(for match: Match) async throws -> Song {
        if let cached = songCache[match.id] { return cached }
        try await authorize()
        var found: Song?
        if let isrc = match.isrc {
            let items = try await MusicCatalogResourceRequest<Song>(matching: \.isrc, equalTo: isrc).response().items
            found = items.max { score($0, match) < score($1, match) }
        }
        if found == nil {
            var request = MusicCatalogSearchRequest(term: "\(match.title) \(match.artist)", types: [Song.self])
            request.limit = 10
            found = try await request.response().songs
                .filter { score($0, match) >= 6 }
                .max { score($0, match) < score($1, match) }
        }
        guard let found else { throw Failure.notInAppleMusic }
        songCache[match.id] = found
        return found
    }

    private func score(_ song: Song, _ match: Match) -> Int {
        TextMatch.score(title: match.title, artist: match.artist, candidateTitle: song.title, candidateArtist: song.artistName)
    }

    func url(for match: Match) async -> URL? {
        if let url = match.appleMusicURL { return url }
        do {
            return try await song(for: match).url
        } catch {
            EventLog.record("musickit-lookup", ["ok": false, "error": "\(error)"])
            return nil
        }
    }

    private func requireSubscription() async throws {
        let subscription = try await MusicSubscription.current
        guard subscription.canPlayCatalogContent else { throw Failure.needsSubscription }
    }

    /// Adds a result right after the current song in the Music app's queue.
    func playNext(_ match: Match) async throws {
        try await requireSubscription()
        let song = try await song(for: match)
        let player = SystemMusicPlayer.shared
        if player.queue.currentEntry == nil {
            player.queue = [song]
            try await player.play()
        } else {
            try await player.queue.insert(song, position: .afterCurrentEntry)
        }
    }

    /// Replaces the Music app's queue with every result that is in Apple Music.
    func playAll(_ matches: [Match]) async throws {
        try await requireSubscription()
        let songs = await songs(for: matches)
        guard !songs.isEmpty else { throw Failure.notInAppleMusic }
        SystemMusicPlayer.shared.queue = SystemMusicPlayer.Queue(for: songs)
        try await SystemMusicPlayer.shared.play()
    }

    func saveAsPlaylist(name: String, _ matches: [Match]) async throws {
        try await authorize()
        let songs = await songs(for: matches)
        guard !songs.isEmpty else { throw Failure.notInAppleMusic }
        _ = try await MusicLibrary.shared.createPlaylist(name: name, description: "Songs that sound alike, found by Soundalike.", items: songs)
    }

    private func songs(for matches: [Match]) async -> [Song] {
        var songs: [Song] = []
        for match in matches {
            if let song = try? await song(for: match) { songs.append(song) }
        }
        return songs
    }
}
