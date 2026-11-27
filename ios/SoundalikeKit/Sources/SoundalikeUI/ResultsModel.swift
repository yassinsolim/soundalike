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

/// What the lists should follow instead of the whole preview.
public enum PartChoice: Sendable {
    /// Seconds of the 30-second preview.
    case preview(PartSpan)
    /// A clip of the listener's own audio as a WAV, sent once and not kept.
    case clip(wav: Data, info: ClipInfo)
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
    /// The app the user plays music in, set by the host (Spotify once it's connected). Results
    /// then open there, whichever app the song came from.
    public var preferred: MusicSource? {
        didSet { if let preferred { target = preferred } }
    }

    private let client: SoundalikeClient
    private var task: Task<Void, Never>?
    private var lastRequest: (() async throws -> SoundalikeResults)?
    /// The request for the song on screen, for another part of it.
    private var base: ((PartSpan?) async throws -> SoundalikeResults)?

    public init(client: SoundalikeClient = SoundalikeClient()) {
        self.client = client
    }

    public func load(shared text: String) {
        guard let link = SharedLink.find(in: text) else {
            state = .failed(SoundalikeError.unsupportedLink.localizedDescription)
            return
        }
        if preferred == nil { target = link.source }
        let client = self.client
        let base: (PartSpan?) async throws -> SoundalikeResults = { part in try await client.similar(to: link, part: part) }
        self.base = base
        start(seed: nil) { try await base(nil) }
    }

    public func load(seed: Seed) {
        if preferred == nil, seed.source != .search { target = seed.source }
        let client = self.client
        let base: (PartSpan?) async throws -> SoundalikeResults = { part in try await client.similar(to: seed, part: part) }
        self.base = base
        start(seed: seed) { try await base(nil) }
    }

    /// Lists for a part of the song on screen, or for the whole preview again (`nil`). The lists on screen
    /// stay until the new ones arrive; a failure is thrown to the caller and leaves them. History keeps the
    /// song as it is: only the song is remembered, never a part or a clip.
    public func match(_ choice: PartChoice?) async throws {
        guard case .loaded(let current) = state, let base else { return }
        let client = self.client
        let started = Date()
        var fields: [String: Any] = [:]
        switch choice {
        case nil: fields["kind"] = "whole"
        case .preview(let span): fields["kind"] = "preview"; fields["part"] = span.query
        case .clip(let wav, let info):
            fields["kind"] = info.source.rawValue
            fields["seconds"] = Int((info.end - info.start).rounded())
            fields["bytes"] = wav.count
        }
        do {
            var results: SoundalikeResults
            switch choice {
            case nil: results = try await base(nil)
            case .preview(let span): results = try await base(span)
            case .clip(let wav, let info):
                results = try await client.similar(toClip: wav, seed: current.seed)
                results.clip = info
            }
            // Keep the control even if an answer lacks the limits.
            if results.parts == nil { results.parts = current.parts }
            fields["ms"] = EventLog.ms(since: started)
            fields["route"] = results.route
            fields["ok"] = true
            // An answer to part= without query.part is for the whole preview: the server ignored the part.
            if case .preview = choice { fields["echoed"] = results.part != nil }
            EventLog.record("part", fields)
            state = .loaded(results)
        } catch {
            fields["ms"] = EventLog.ms(since: started)
            fields["ok"] = false
            fields["error"] = "\(error)"
            EventLog.record("part", fields)
            throw error
        }
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
                if !Task.isCancelled {
                    state = .loaded(results)
                    HistoryStore.shared.add(results.seed)
                }
            } catch is CancellationError {
            } catch {
                EventLog.record("results-failed", ["ms": EventLog.ms(since: started), "error": "\(error)"])
                if !Task.isCancelled { state = .failed(error.localizedDescription) }
            }
        }
    }
}
