import Foundation
import SoundalikeKit

/// Settings shared by the app and its extensions through the app group.
public enum SharedSettings {
    public static let appGroup = "group.app.yassin.soundalike"
    static var defaults: UserDefaults { UserDefaults(suiteName: appGroup) ?? .standard }

    /// The app results open in when the song did not come from a music app.
    public static var preferredApp: MusicSource {
        get { MusicSource(rawValue: defaults.string(forKey: "preferredApp") ?? "") ?? .spotify }
        set { defaults.set(newValue.rawValue, forKey: "preferredApp") }
    }
}

/// Loads soundalikes for shared text or a known seed and exposes a `ResultsState`.
@MainActor
public final class ResultsModel: ObservableObject {
    @Published public var state: ResultsState = .loading(nil)
    @Published public var target: MusicSource = SharedSettings.preferredApp {
        didSet { if target != oldValue { SharedSettings.preferredApp = target } }
    }
    /// A one-line message from the host, for example why a link could not open.
    @Published public var notice: String?

    private let client: SoundalikeClient
    private var task: Task<Void, Never>?
    private var lastRequest: (() async throws -> SoundalikeResults)?

    public init(client: SoundalikeClient = SoundalikeClient()) {
        self.client = client
    }

    public func load(shared text: String) {
        guard let link = SharedLink.find(in: text) else {
            state = .failed(SoundalikeError.unsupportedLink.localizedDescription)
            return
        }
        target = link.source
        let client = self.client
        start(seed: nil) { try await client.similar(to: link) }
    }

    public func load(seed: Seed) {
        if seed.source != .search { target = seed.source }
        let client = self.client
        start(seed: seed) { try await client.similar(to: seed) }
    }

    public func retry() {
        guard let lastRequest else { return }
        start(seed: nil, lastRequest)
    }

    private func start(seed: Seed?, _ request: @escaping () async throws -> SoundalikeResults) {
        task?.cancel()
        lastRequest = request
        state = .loading(seed)
        let started = Date()
        task = Task {
            do {
                let results = try await request()
                EventLog.record("results", [
                    "ms": EventLog.ms(since: started), "route": results.route, "count": results.matches.count,
                    "seed": "\(results.seed.title) — \(results.seed.artist)", "source": results.seed.source.rawValue,
                ])
                if !Task.isCancelled { state = .loaded(results) }
            } catch is CancellationError {
            } catch {
                EventLog.record("results-failed", ["ms": EventLog.ms(since: started), "error": "\(error)"])
                if !Task.isCancelled { state = .failed(error.localizedDescription) }
            }
        }
    }
}
