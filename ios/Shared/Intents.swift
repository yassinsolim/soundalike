import AppIntents
import Foundation
import SoundalikeKit
import SoundalikeUI
import SwiftUI

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

/// The last results an intent showed, so "See all" opens the same list in the app.
enum LastResults {
    private static let key = "lastResults"

    static func save(_ results: SoundalikeResults) {
        UserDefaults(suiteName: PendingRoute.appGroup)?.set(try? JSONEncoder().encode(results), forKey: key)
    }

    static func load() -> SoundalikeResults? {
        guard let data = UserDefaults(suiteName: PendingRoute.appGroup)?.data(forKey: key) else { return nil }
        return try? JSONDecoder().decode(SoundalikeResults.self, from: data)
    }
}

/// Control Center, Action Button, Siri and Back Tap: reads what's playing and shows
/// soundalikes in a system card over the current app. Only when nothing can be read
/// (another app playing, Spotify not connected) does it open Soundalike to listen.
struct FindSoundalikesIntent: LiveActivityIntent {
    static let title: LocalizedStringResource = "Find Soundalikes"
    static let description = IntentDescription("Shows songs that sound like what you're listening to, right over the app you're in.")
    static let supportedModes: IntentModes = [.background, .foreground(.dynamic)]

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog & ShowsSnippetIntent {
        let started = Date()
        guard let seed = await NowPlaying.read() else {
            EventLog.record("intent", ["path": "listen", "ms": EventLog.ms(since: started)])
            PendingRoute.post(.listen)
            try await continueInForeground(nil, alwaysConfirm: false)
            return .result(dialog: "Listening…")
        }
        let results = try await SoundalikeClient().similar(to: seed)
        let dialog = try await snippet(for: results, started: started)
        // Siri and Shortcuts draw the card; the Action Button and Back Tap don't, so the
        // Dynamic Island shows the same results everywhere.
        await SoundalikeActivity.show(results)
        return .result(dialog: dialog, snippetIntent: SoundalikesSnippetIntent())
    }
}

struct FindForLinkIntent: AppIntent {
    static let title: LocalizedStringResource = "Find Soundalikes for a Link"
    static let description = IntentDescription("Shows songs that sound like a shared song link.")
    static let supportedModes: IntentModes = [.background, .foreground(.dynamic)]

    @Parameter(title: "Song link", description: "A song link from a music app, or text containing one.")
    var link: String

    static var parameterSummary: some ParameterSummary {
        Summary("Find soundalikes for \(\.$link)")
    }

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog & ShowsSnippetIntent {
        let started = Date()
        let dialog = try await snippet(for: try await SoundalikeClient().similar(toShared: link), started: started)
        return .result(dialog: dialog, snippetIntent: SoundalikesSnippetIntent())
    }
}

/// Prepares and saves the card, and returns what Siri says with it.
@MainActor
private func snippet(for results: SoundalikeResults, started: Date) async throws -> IntentDialog {
    LastResults.save(results)
    let rows = await SnippetRow.make(from: results)
    SnippetCard(seed: results.seed, rows: rows).save()
    EventLog.record("intent", ["path": "snippet", "ms": EventLog.ms(since: started), "source": results.seed.source.rawValue, "seed": results.seed.title])
    return IntentDialog(stringLiteral: "Sounds like \(results.seed.title)")
}

/// Opens one result in the music app the song came from.
struct OpenSongIntent: AppIntent {
    static let title: LocalizedStringResource = "Open Song"
    static let isDiscoverable = false
    static let openAppWhenRun = false

    @Parameter(title: "URL") var url: URL

    init() {}
    init(url: URL) { self.url = url }

    func perform() async throws -> some IntentResult & OpensIntent {
        .result(opensIntent: OpenURLIntent(url))
    }
}

/// Adds one result to the Spotify queue without leaving the current app.
/// A `LiveActivityIntent` so it runs in the app's process, where the Spotify token lives,
/// also when tapped in the Dynamic Island.
struct QueueInSpotifyIntent: LiveActivityIntent {
    static let title: LocalizedStringResource = "Queue Song"
    static let isDiscoverable = false

    @Parameter(title: "Track") var trackID: String

    init() {}
    init(trackID: String) { self.trackID = trackID }

    func perform() async throws -> some IntentResult {
        do {
            try await SpotifyAccount().queue(trackID: trackID)
            EventLog.record("spotify-queue", ["ok": true, "from": "card"])
        } catch {
            EventLog.record("spotify-queue", ["ok": false, "from": "card", "error": "\(error)"])
            throw error
        }
        return .result()
    }
}

/// Opens the full list in the app.
struct ShowAllIntent: AppIntent {
    static let title: LocalizedStringResource = "Show All Soundalikes"
    static let isDiscoverable = false
    static let supportedModes: IntentModes = .foreground(.immediate)

    func perform() async throws -> some IntentResult {
        PendingRoute.post(.last)
        return .result()
    }
}
