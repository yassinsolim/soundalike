import Foundation
import Testing
@testable import SoundalikeKit
@testable import SoundalikeUI

@MainActor
struct ResultsModelTests {
    /// Nothing listens here, so no request leaves the machine; only the target is checked.
    private let offline = SoundalikeClient(base: URL(string: "http://127.0.0.1:9")!)

    @Test func spotifyConnectingAfterResultsOpenMovesThemToSpotify() {
        let model = ResultsModel(client: offline)
        model.load(seed: Seed(title: "Every Breaking Wave", artist: "U2", source: .appleMusic))
        #expect(model.target == .appleMusic)

        // Connect Spotify finishes while these results are on screen.
        model.preferred = .spotify
        #expect(model.target == .spotify)

        // A later song found in the Music app no longer flips the results back.
        model.load(seed: Seed(title: "Blinding Lights", artist: "The Weeknd", source: .appleMusic))
        #expect(model.target == .spotify)
    }
}
