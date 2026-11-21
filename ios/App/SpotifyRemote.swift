import Foundation
import SoundalikeKit
import SoundalikeUI
import SpotifyiOS

/// Asks the Spotify app on this phone what it's playing (Spotify's App Remote SDK).
///
/// App Remote talks to the Spotify app, not the Web API, so it works for Spotify users
/// who aren't on the development-mode allow list (verified on iOS 26.6 in November 2026).
@MainActor
final class SpotifyRemote: NSObject, SPTAppRemoteDelegate {
    static let shared = SpotifyRemote()

    private lazy var remote: SPTAppRemote = {
        let configuration = SPTConfiguration(clientID: SpotifyAccount.clientID, redirectURL: URL(string: SpotifyAccount.redirectURI)!)
        let remote = SPTAppRemote(configuration: configuration, logLevel: .error)
        remote.delegate = self
        return remote
    }()
    private var waiting: [CheckedContinuation<Bool, Never>] = []

    func currentTrack() async -> Seed? {
        let started = Date()
        guard let token = try? await SpotifyAccount().accessToken() else { return nil }
        remote.connectionParameters.accessToken = token
        guard await connect() else {
            EventLog.record("app-remote", ["ok": false, "stage": "connect", "ms": EventLog.ms(since: started)])
            return nil
        }
        let state: SPTAppRemotePlayerState? = await withCheckedContinuation { continuation in
            remote.playerAPI?.getPlayerState { result, _ in continuation.resume(returning: result as? SPTAppRemotePlayerState) }
        }
        guard let track = state?.track, state?.isPaused == false else {
            EventLog.record("app-remote", ["ok": false, "stage": "state", "ms": EventLog.ms(since: started)])
            return nil
        }
        EventLog.record("app-remote", ["ok": true, "ms": EventLog.ms(since: started)])
        let id = track.uri.components(separatedBy: ":").last
        return Seed(
            title: track.name, artist: track.artist.name, source: .spotify,
            sharedURL: id.map { URL(string: "https://open.spotify.com/track/\($0)")! }, spotifyID: id
        )
    }

    private func connect() async -> Bool {
        if remote.isConnected { return true }
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
}
