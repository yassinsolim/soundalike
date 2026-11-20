import AppIntents
import Foundation

/// Hands a route to the app. Intents run with `openAppWhenRun`, so the app comes
/// forward; the route travels through the app group so it arrives whichever
/// process ran `perform()` (the app for Siri and Shortcuts, the controls
/// extension for Control Center and the Action Button).
enum PendingRoute {
    static let appGroup = "group.app.yassin.soundalike"
    private static let key = "pendingRoute"
    private static let dateKey = "pendingRouteDate"
    static let posted = Notification.Name("app.yassin.soundalike.pendingRoute")

    static func post(_ route: AppRoute) {
        let defaults = UserDefaults(suiteName: appGroup)
        defaults?.set(route.url.absoluteString, forKey: key)
        defaults?.set(Date(), forKey: dateKey)
        NotificationCenter.default.post(name: posted, object: nil)
    }

    /// The route posted in the last minute, removed once read.
    static func take() -> AppRoute? {
        let defaults = UserDefaults(suiteName: appGroup)
        defer {
            defaults?.removeObject(forKey: key)
            defaults?.removeObject(forKey: dateKey)
        }
        guard let raw = defaults?.string(forKey: key), let url = URL(string: raw),
              let date = defaults?.object(forKey: dateKey) as? Date, Date().timeIntervalSince(date) < 60
        else { return nil }
        return AppRoute(url: url)
    }
}

struct FindSoundalikesIntent: AppIntent {
    static let title: LocalizedStringResource = "Find Soundalikes"
    static let description = IntentDescription("Finds songs that sound like what you're listening to: the song playing in the Music app, or the one Soundalike hears.")
    static let openAppWhenRun = true

    func perform() async throws -> some IntentResult {
        PendingRoute.post(.auto)
        return .result()
    }
}

struct FindForLinkIntent: AppIntent {
    static let title: LocalizedStringResource = "Find Soundalikes for a Link"
    static let description = IntentDescription("Finds songs that sound like a shared song link.")
    static let openAppWhenRun = true

    @Parameter(title: "Song link", description: "A song link from a music app, or text containing one.")
    var link: String

    static var parameterSummary: some ParameterSummary {
        Summary("Find soundalikes for \(\.$link)")
    }

    func perform() async throws -> some IntentResult {
        PendingRoute.post(.find(link))
        return .result()
    }
}
