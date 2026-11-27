import Foundation
import Testing
@testable import SoundalikeKit
@testable import SoundalikeUI

@MainActor
struct HistoryStoreTests {
    private func store() -> (HistoryStore, URL) {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("history-\(UUID().uuidString).json")
        return (HistoryStore(fileURL: url), url)
    }

    @Test func searchingASongAgainMovesItToTheTopOnce() {
        let (history, _) = store()
        history.add(Seed(title: "Blinding Lights", artist: "The Weeknd", source: .spotify))
        history.add(Seed(title: "Heartbeat", artist: "DJ Planet Express", source: .spotify))
        history.add(Seed(title: "Blinding lights", artist: "the weeknd", source: .appleMusic))
        #expect(history.entries.map(\.seed.title) == ["Blinding lights", "Heartbeat"])
    }

    @Test func keepsTheNewestEntriesUpToTheLimit() {
        let (history, _) = store()
        for index in 0..<(HistoryStore.limit + 5) {
            history.add(Seed(title: "Song \(index)", artist: "Artist", source: .search))
        }
        #expect(history.entries.count == HistoryStore.limit)
        #expect(history.entries.first?.seed.title == "Song \(HistoryStore.limit + 4)")
    }

    @Test func anotherProcessSeesTheSavedHistory() {
        let (history, url) = store()
        history.add(Seed(title: "UP", artist: "Effy", source: .spotify))
        let shareExtension = HistoryStore(fileURL: url)
        #expect(shareExtension.entries.map(\.seed.title) == ["UP"])
        shareExtension.remove(ids: [HistoryStore.key(Seed(title: "UP", artist: "Effy", source: .spotify))])
        history.reload()
        #expect(history.entries.isEmpty)
    }
}
