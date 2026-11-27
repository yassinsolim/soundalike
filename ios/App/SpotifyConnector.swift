import AuthenticationServices
import SoundalikeKit
import SoundalikeUI
import SwiftUI
import UIKit

/// Connects Spotify with one tap: Spotify's own app shows an "Agree" screen and
/// hands back a code, so nobody types a password. Falls back to the web sign-in
/// only when the Spotify app isn't installed.
@MainActor
final class SpotifyConnector: ObservableObject {
    static let shared = SpotifyConnector()

    @Published private(set) var isConnected = SpotifyAccount().isConnected
    @Published var failure: String?
    private var pending: SpotifyAccount.Authorization?

    /// Runs the Spotify app's authorization again without the web fallback. Spotify starts
    /// accepting App Remote connections after it handles an authorization, which it doesn't
    /// after a restart; already-authorized accounts go straight back to Soundalike.
    func wakeSpotifyApp() async {
        let authorization = SpotifyAccount.authorization()
        let opened = await UIApplication.shared.open(authorization.appURL)
        EventLog.record("app-remote", ["stage": "wake", "opened": opened])
        if opened { pending = authorization }
    }

    func connect(using web: WebAuthenticationSession) async {
        let authorization = SpotifyAccount.authorization()
        if await UIApplication.shared.open(authorization.appURL) {
            pending = authorization
            EventLog.record("spotify-connect", ["via": "app"])
            return
        }
        do {
            let callback = try await web.authenticate(
                using: authorization.webURL, callbackURLScheme: SpotifyAccount.callbackScheme, preferredBrowserSession: .shared
            )
            try await finish(authorization, callback)
        } catch ASWebAuthenticationSessionError.canceledLogin {
        } catch {
            report(error)
        }
    }

    /// The Spotify app sends the user back here with soundalike://spotify-callback?code=…
    func handleCallback(_ url: URL) async {
        guard let pending else { return }
        self.pending = nil
        do { try await finish(pending, url) } catch { report(error) }
    }

    func disconnect() {
        SpotifyAccount().disconnect()
        isConnected = false
    }

    private func finish(_ authorization: SpotifyAccount.Authorization, _ callback: URL) async throws {
        try await SpotifyAccount().complete(authorization, callback: callback)
        isConnected = true
        EventLog.record("spotify-connect", ["ok": true])
        await PlayerController.shared.reconnectSpotify()
    }

    private func report(_ error: Error) {
        EventLog.record("spotify-connect", ["ok": false, "error": "\(error)"])
        failure = error.localizedDescription
    }
}
