import ActivityKit
import AppIntents
import Foundation
import SoundalikeKit
import SoundalikeUI

/// The Dynamic Island / Lock Screen card a Control Center or Action Button press shows.
/// Controls can't present snippets or dialogs (measured on iOS 26.6: the intent ran and
/// returned a snippet, nothing appeared), but they can start a Live Activity.
struct SoundalikeActivityAttributes: ActivityAttributes {
    struct Row: Codable, Hashable {
        var title: String
        var artist: String
        var match: Int
        var openURL: URL
        var spotifyTrackID: String?
    }

    struct ContentState: Codable, Hashable {
        var seedTitle: String
        var seedArtist: String
        var rows: [Row]
    }
}

enum SoundalikeActivity {
    static let rowCount = 3

    /// Replaces any card already showing with the new results.
    @MainActor
    static func show(_ results: SoundalikeResults) async {
        let rows = await SnippetRow.make(from: results).prefix(rowCount).map {
            SoundalikeActivityAttributes.Row(
                title: $0.match.title, artist: $0.match.artist, match: $0.match.match,
                openURL: $0.openURL, spotifyTrackID: $0.spotifyTrackID
            )
        }
        let state = SoundalikeActivityAttributes.ContentState(seedTitle: results.seed.title, seedArtist: results.seed.artist, rows: Array(rows))
        let content = ActivityContent(state: state, staleDate: Date().addingTimeInterval(15 * 60))
        for activity in Activity<SoundalikeActivityAttributes>.activities {
            await activity.end(nil, dismissalPolicy: .immediate)
        }
        do {
            let activity = try Activity.request(attributes: SoundalikeActivityAttributes(), content: content)
            // An alerting update is what makes the Dynamic Island expand over the current app.
            await activity.update(content, alertConfiguration: AlertConfiguration(
                title: LocalizedStringResource(stringLiteral: "Sounds like \(results.seed.title)"),
                body: LocalizedStringResource(stringLiteral: rows.map { "\($0.title) · \($0.artist)" }.joined(separator: "\n")),
                sound: .default
            ))
            EventLog.record("live-activity", ["ok": true])
        } catch {
            EventLog.record("live-activity", ["ok": false, "error": "\(error)"])
        }
    }
}

/// What the Control Center / Action Button control runs.
struct FindSoundalikesControlIntent: LiveActivityIntent {
    static let title: LocalizedStringResource = "Find Soundalikes"
    static let description = IntentDescription("Shows songs that sound like what you're playing, in the Dynamic Island.")
    static let isDiscoverable = false
    static let supportedModes: IntentModes = [.background, .foreground(.dynamic)]

    @MainActor
    func perform() async throws -> some IntentResult {
        let started = Date()
        guard let seed = await NowPlaying.read() else {
            EventLog.record("control", ["path": "listen", "ms": EventLog.ms(since: started)])
            PendingRoute.post(.listen)
            try await continueInForeground(nil, alwaysConfirm: false)
            return .result()
        }
        let results = try await SoundalikeClient().similar(to: seed)
        LastResults.save(results)
        await SoundalikeActivity.show(results)
        EventLog.record("control", ["path": "activity", "ms": EventLog.ms(since: started), "seed": seed.title])
        return .result()
    }
}

/// Closes the card.
struct DismissSoundalikesIntent: LiveActivityIntent {
    static let title: LocalizedStringResource = "Dismiss Soundalikes"
    static let isDiscoverable = false

    func perform() async throws -> some IntentResult {
        for activity in Activity<SoundalikeActivityAttributes>.activities {
            await activity.end(nil, dismissalPolicy: .immediate)
        }
        return .result()
    }
}
