import AppIntents
import Foundation

// Every intent hands off to the app through a soundalike:// URL. That keeps them
// working the same way from Siri, Shortcuts, Spotlight, the Action Button and
// Control Center, whichever process runs `perform()`.

struct FindForCurrentSongIntent: AppIntent {
    static let title: LocalizedStringResource = "Find Soundalikes for the Current Song"
    static let description = IntentDescription("Finds songs that sound like what's playing in the Music app.")

    func perform() async throws -> some IntentResult & OpensIntent {
        .result(opensIntent: OpenURLIntent(AppRoute.nowPlaying.url))
    }
}

struct ListenAndFindIntent: AppIntent {
    static let title: LocalizedStringResource = "Recognise and Find Soundalikes"
    static let description = IntentDescription("Listens to the song playing around you, from any app or speaker, and finds songs that sound like it.")

    func perform() async throws -> some IntentResult & OpensIntent {
        .result(opensIntent: OpenURLIntent(AppRoute.listen.url))
    }
}

struct FindForLinkIntent: AppIntent {
    static let title: LocalizedStringResource = "Find Soundalikes for a Link"
    static let description = IntentDescription("Finds songs that sound like a shared song link.")

    @Parameter(title: "Song link", description: "A song link from a music app, or text containing one.")
    var link: String

    static var parameterSummary: some ParameterSummary {
        Summary("Find soundalikes for \(\.$link)")
    }

    func perform() async throws -> some IntentResult & OpensIntent {
        .result(opensIntent: OpenURLIntent(AppRoute.find(link).url))
    }
}
