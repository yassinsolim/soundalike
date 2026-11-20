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

    func handle(_ url: URL) {
        if let route = AppRoute(url: url) {
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
            intent: ListenAndFindIntent(),
            phrases: [
                "Find soundalikes with \(.applicationName)",
                "What sounds like this in \(.applicationName)",
                "\(.applicationName) this song",
            ],
            shortTitle: "Recognise Song",
            systemImageName: "waveform"
        )
        AppShortcut(
            intent: FindForCurrentSongIntent(),
            phrases: [
                "Find soundalikes for the current song in \(.applicationName)",
                "\(.applicationName) what's playing",
            ],
            shortTitle: "Now Playing",
            systemImageName: "music.note"
        )
        AppShortcut(
            intent: FindForLinkIntent(),
            phrases: ["Find soundalikes for a link with \(.applicationName)"],
            shortTitle: "From a Link",
            systemImageName: "link"
        )
    }
}
