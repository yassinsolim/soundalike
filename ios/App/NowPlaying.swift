import Foundation
import MusicKit
import SoundalikeKit

/// Reads the song the user is listening to without the microphone:
/// the Music app through MusicKit, Spotify through the Web API when connected.
enum NowPlaying {
    /// Reads Spotify through the Spotify app itself (App Remote) when the Web API can't:
    /// set by the app, which links the Spotify SDK; extensions leave it nil.
    @MainActor static var spotifyAppRemote: (() async -> Seed?)?

    @MainActor
    static func read() async -> Seed? {
        if let seed = await appleMusic() { return seed }
        guard SpotifyAccount().isConnected else { return nil }
        if let seed = try? await SpotifyAccount().currentlyPlaying() { return seed }
        return await spotifyAppRemote?()
    }

    /// The Music app's current song when it is playing. No subscription needed.
    @MainActor
    static func appleMusic(requireAuthorization: Bool = false) async -> Seed? {
        if requireAuthorization, MusicAuthorization.currentStatus == .notDetermined {
            _ = await MusicAuthorization.request()
        }
        guard MusicAuthorization.currentStatus == .authorized else { return nil }
        let player = SystemMusicPlayer.shared
        guard player.state.playbackStatus == .playing || requireAuthorization,
              let entry = player.queue.currentEntry else { return nil }
        var seed = Seed(title: entry.title, artist: entry.subtitle ?? "", source: .appleMusic)
        if case .song(let song)? = entry.item {
            seed.title = song.title
            seed.artist = song.artistName
            seed.isrc = song.isrc
            seed.artworkURL = song.artwork?.url(width: 600, height: 600)
            seed.appleMusicID = song.id.rawValue
            // Library and queue items often lack an ISRC; the catalog copy has it.
            if seed.isrc == nil,
               let catalog = try? await MusicCatalogResourceRequest<Song>(matching: \.id, equalTo: song.id).response().items.first {
                seed.isrc = catalog.isrc
                seed.artworkURL = seed.artworkURL ?? catalog.artwork?.url(width: 600, height: 600)
            }
        }
        seed.storefront = try? await MusicDataRequest.currentCountryCode
        return seed
    }
}
