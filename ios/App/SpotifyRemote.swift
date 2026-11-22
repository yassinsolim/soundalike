import Foundation
import SoundalikeKit
import SoundalikeUI
import SpotifyiOS
import UIKit

/// Talks to the Spotify app on this phone (Spotify's App Remote SDK): what it's playing,
/// and play, pause, skip and seek for the player bar.
///
/// App Remote talks to the Spotify app, not the Web API, so it works for Spotify users
/// who aren't on the development-mode allow list (verified on iOS 26.6 in November 2026).
@MainActor
final class SpotifyRemote: NSObject, SPTAppRemoteDelegate, SPTAppRemotePlayerStateDelegate {
    static let shared = SpotifyRemote()

    /// Called with every player change while observing.
    var onState: ((SPTAppRemotePlayerState) -> Void)?

    private lazy var remote: SPTAppRemote = {
        let configuration = SPTConfiguration(clientID: SpotifyAccount.clientID, redirectURL: URL(string: SpotifyAccount.redirectURI)!)
        let remote = SPTAppRemote(configuration: configuration, logLevel: .error)
        remote.delegate = self
        return remote
    }()
    private var waiting: [CheckedContinuation<Bool, Never>] = []

    func currentTrack() async -> Seed? {
        let started = Date()
        guard await connect() else {
            EventLog.record("app-remote", ["ok": false, "stage": "connect", "ms": EventLog.ms(since: started)])
            return nil
        }
        let state = await playerState()
        guard let track = state?.track, state?.isPaused == false else {
            EventLog.record("app-remote", ["ok": false, "stage": "state", "ms": EventLog.ms(since: started)])
            return nil
        }
        EventLog.record("app-remote", ["ok": true, "ms": EventLog.ms(since: started)])
        return Self.seed(for: track)
    }

    static func seed(for track: SPTAppRemoteTrack) -> Seed {
        let id = track.uri.components(separatedBy: ":").last
        return Seed(
            title: track.name, artist: track.artist.name, source: .spotify,
            sharedURL: id.map { URL(string: "https://open.spotify.com/track/\($0)")! }, spotifyID: id
        )
    }

    var isConnected: Bool { remote.isConnected }

    /// Follows the Spotify player until `stopObserving`. Returns false when Spotify
    /// isn't connected in Soundalike or the Spotify app won't take the connection.
    func startObserving() async -> Bool {
        guard SpotifyAccount().isConnected, await connect() else { return false }
        remote.playerAPI?.delegate = self
        remote.playerAPI?.subscribe(toPlayerState: nil)
        if let state = await playerState() { onState?(state) }
        return true
    }

    /// App Remote must not stay connected while Soundalike is in the background.
    func stopObserving() {
        if remote.isConnected { remote.disconnect() }
    }

    func setPlaying(_ playing: Bool) {
        if playing { remote.playerAPI?.resume(nil) } else { remote.playerAPI?.pause(nil) }
    }

    func skipToNext() { remote.playerAPI?.skip(toNext: nil) }
    func skipToPrevious() { remote.playerAPI?.skip(toPrevious: nil) }
    func seek(to seconds: TimeInterval) { remote.playerAPI?.seek(toPosition: Int(seconds * 1000), callback: nil) }

    func image(for track: SPTAppRemoteTrack) async -> UIImage? {
        guard let imageAPI = remote.imageAPI else { return nil }
        return await withCheckedContinuation { continuation in
            imageAPI.fetchImage(forItem: track, with: CGSize(width: 240, height: 240)) { image, _ in
                continuation.resume(returning: image as? UIImage)
            }
        }
    }

    private func playerState() async -> SPTAppRemotePlayerState? {
        guard let playerAPI = remote.playerAPI else { return nil }
        return await withCheckedContinuation { continuation in
            playerAPI.getPlayerState { result, _ in continuation.resume(returning: result as? SPTAppRemotePlayerState) }
        }
    }

    private func connect() async -> Bool {
        if remote.isConnected { return true }
        guard let token = try? await SpotifyAccount().accessToken() else { return false }
        remote.connectionParameters.accessToken = token
        return await withCheckedContinuation { continuation in
            waiting.append(continuation)
            if waiting.count == 1 { remote.connect() }
            Task {
                try? await Task.sleep(for: .seconds(3))
                self.finish(false)
            }
        }
    }

    private func finish(_ connected: Bool) {
        let continuations = waiting
        waiting = []
        continuations.forEach { $0.resume(returning: connected) }
    }

    nonisolated func appRemoteDidEstablishConnection(_ appRemote: SPTAppRemote) {
        Task { @MainActor in finish(true) }
    }

    nonisolated func appRemote(_ appRemote: SPTAppRemote, didFailConnectionAttemptWithError error: Error?) {
        Task { @MainActor in
            EventLog.record("app-remote", ["ok": false, "stage": "connect", "error": "\(String(describing: error))"])
            finish(false)
        }
    }

    nonisolated func appRemote(_ appRemote: SPTAppRemote, didDisconnectWithError error: Error?) {}

    nonisolated func playerStateDidChange(_ playerState: SPTAppRemotePlayerState) {
        Task { @MainActor in onState?(playerState) }
    }
}
