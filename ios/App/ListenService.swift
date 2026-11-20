import Foundation
import ShazamKit
import SoundalikeKit
import SoundalikeUI

/// Identifies the song playing nearby (or through the phone's speaker) with ShazamKit.
/// Works with any player, so it is the one-press route for Spotify listeners.
@MainActor
final class ListenService: ObservableObject {
    enum Failure: LocalizedError {
        case noMatch
        case microphone
        case unavailable

        var errorDescription: String? {
            switch self {
            case .noMatch: "Couldn't recognise the song. Turn it up and try again."
            case .microphone: "Soundalike needs the microphone to recognise songs. Allow it in Settings."
            case .unavailable: "Song recognition isn't available right now. Try again in a moment."
            }
        }
    }

    @Published private(set) var isListening = false
    private var session: SHManagedSession?

    func identify() async throws -> Seed {
        let session = SHManagedSession()
        self.session = session
        isListening = true
        defer {
            isListening = false
            self.session = nil
        }
        await session.prepare()
        switch await session.result() {
        case .match(let match):
            guard let item = match.mediaItems.first, let title = item.title else { throw Failure.noMatch }
            return Seed(
                title: title, artist: item.artist ?? "", isrc: item.isrc, artworkURL: item.artworkURL,
                source: SharedSettings.preferredApp, appleMusicID: item.appleMusicID
            )
        case .noMatch:
            throw Failure.noMatch
        case .error(let error, _):
            if (error as NSError).domain == SHErrorDomain, (error as NSError).code == SHError.Code.audioDiscontinuity.rawValue {
                throw Failure.noMatch
            }
            EventLog.record("listen-error", ["error": "\(error)"])
            throw (error as NSError).localizedDescription.localizedCaseInsensitiveContains("microphone") ? Failure.microphone : Failure.unavailable
        }
    }

    func cancel() {
        session?.cancel()
    }
}
