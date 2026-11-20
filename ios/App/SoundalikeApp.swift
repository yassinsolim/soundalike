import AppIntents
import SoundalikeKit
import SoundalikeUI
import SwiftUI

@main
struct SoundalikeApp: App {
    @StateObject private var router = Router()

    var body: some Scene {
        WindowGroup {
            HomeView()
                .environmentObject(router)
                .onReceive(NotificationCenter.default.publisher(for: PendingRoute.posted)) { _ in router.takePending() }
                .onOpenURL { router.handle($0) }
                .onContinueUserActivity(NSUserActivityTypeBrowsingWeb) { activity in
                    if let url = activity.webpageURL { router.handle(url) }
                }
        }
    }
}

/// The one place deep links, intents and buttons ask for results.
@MainActor
final class Router: ObservableObject {
    @Published var pending: AppRoute?

    /// Picks up a route an App Intent or Control left in the app group.
    func takePending() {
        if let route = PendingRoute.take() { pending = route }
    }

    func handle(_ url: URL) {
        if SpotifyAccount.isCallback(url) {
            Task { await SpotifyConnector.shared.handleCallback(url) }
        } else if let route = AppRoute(url: url) {
            pending = route
        } else if SharedLink.parse(url) != nil {
            // A plain Spotify or Apple Music link opened with Soundalike.
            pending = .find(url.absoluteString)
        }
    }
}

struct SoundalikeShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(
            intent: FindSoundalikesIntent(),
            phrases: [
                "\(.applicationName) this song",
                "Find soundalikes with \(.applicationName)",
                "What sounds like this in \(.applicationName)",
            ],
            shortTitle: "Find Soundalikes",
            systemImageName: "waveform"
        )
        AppShortcut(
            intent: FindForLinkIntent(),
            phrases: ["Find soundalikes for a link with \(.applicationName)"],
            shortTitle: "From a Link",
            systemImageName: "link"
        )
    }
}
