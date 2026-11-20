import Foundation

/// Deep links into the app. App Intents, Controls and the share extension's
/// fallback all speak this one URL scheme, so they work from any process.
enum AppRoute: Equatable {
    /// soundalike://find?url=<shared link or text>
    case find(String)
    /// soundalike://now-playing
    case nowPlaying
    /// soundalike://listen
    case listen
    /// soundalike://search?q=<text>
    case search(String)

    init?(url: URL) {
        let components = URLComponents(url: url, resolvingAgainstBaseURL: false)
        let query = components?.queryItems ?? []
        func value(_ name: String) -> String? { query.first { $0.name == name }?.value }

        let isWebFind = url.scheme == "https" && url.host() == "soundalike.yassin.app" && url.path().hasPrefix("/find")
        let action = isWebFind ? "find" : (url.scheme == "soundalike" ? url.host() : nil)
        switch action {
        case "find": guard let shared = value("url"), !shared.isEmpty else { return nil }; self = .find(shared)
        case "now-playing": self = .nowPlaying
        case "listen": self = .listen
        case "search": self = .search(value("q") ?? "")
        default: return nil
        }
    }

    var url: URL {
        var components = URLComponents()
        components.scheme = "soundalike"
        switch self {
        case .find(let shared):
            components.host = "find"
            components.queryItems = [URLQueryItem(name: "url", value: shared)]
        case .nowPlaying: components.host = "now-playing"
        case .listen: components.host = "listen"
        case .search(let text):
            components.host = "search"
            components.queryItems = [URLQueryItem(name: "q", value: text)]
        }
        return components.url!
    }
}
