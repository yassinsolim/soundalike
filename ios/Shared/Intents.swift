import AppIntents
import Foundation

/// Hands a route to the app through the app group, so it arrives whichever
/// process ran the intent (the app, or the controls extension).
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

/// Control Center, Action Button, Siri and Back Tap: opens Soundalike straight onto
/// soundalikes for what's playing. The app reads the song (Apple Music, or Spotify
/// through the Spotify app) and only listens when it can't.
///
/// Tried and dropped on 2026-11-20: a snippet card (iOS never shows snippets or dialogs
/// from Control Center or the Action Button) and a Live Activity (capped at 160 points
/// tall and collapses after a few seconds, too small to use).
struct FindSoundalikesIntent: AppIntent {
    static let title: LocalizedStringResource = "Find Soundalikes"
    static let description = IntentDescription("Opens songs that sound like what you're listening to.")
    static let supportedModes: IntentModes = .foreground(.immediate)

    func perform() async throws -> some IntentResult {
        PendingRoute.post(.auto)
        return .result()
    }
}

struct FindForLinkIntent: AppIntent {
    static let title: LocalizedStringResource = "Find Soundalikes for a Link"
    static let description = IntentDescription("Opens songs that sound like a shared song link.")
    static let supportedModes: IntentModes = .foreground(.immediate)

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
