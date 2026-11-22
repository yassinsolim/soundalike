import Foundation
import SoundalikeKit

/// A song the user found soundalikes for.
public struct HistoryEntry: Codable, Hashable, Identifiable, Sendable {
    public var seed: Seed
    public var date: Date

    public var id: String { HistoryStore.key(seed) }
}

/// Recently searched songs, newest first. Kept in the app group so lookups from the
/// share extension show up in the app too. Stays on the phone.
@MainActor
public final class HistoryStore: ObservableObject {
    public static let shared = HistoryStore()
    static let limit = 200

    @Published public private(set) var entries: [HistoryEntry] = []

    private let fileURL: URL?

    init(fileURL: URL? = FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: SharedSettings.appGroup)?
        .appendingPathComponent("Library/history.json")) {
        self.fileURL = fileURL
        reload()
    }

    public nonisolated static func key(_ seed: Seed) -> String {
        TextMatch.normalize(seed.title) + "|" + TextMatch.normalize(seed.artist)
    }

    /// Picks up entries another process (the share extension) added.
    public func reload() {
        guard let fileURL, let data = try? Data(contentsOf: fileURL),
              let saved = try? JSONDecoder().decode([HistoryEntry].self, from: data) else { return }
        if saved != entries { entries = saved }
    }

    /// Moves the song to the top, so the list holds each song once.
    public func add(_ seed: Seed, at date: Date = Date()) {
        reload()
        let entry = HistoryEntry(seed: seed, date: date)
        entries.removeAll { $0.id == entry.id }
        entries.insert(entry, at: 0)
        if entries.count > Self.limit { entries.removeLast(entries.count - Self.limit) }
        save()
    }

    public func remove(ids: Set<String>) {
        entries.removeAll { ids.contains($0.id) }
        save()
    }

    public func clear() {
        entries = []
        save()
    }

    private func save() {
        guard let fileURL, let data = try? JSONEncoder().encode(entries) else { return }
        try? FileManager.default.createDirectory(at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? data.write(to: fileURL, options: .atomic)
    }
}
