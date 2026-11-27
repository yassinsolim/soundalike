import SoundalikeKit
import SoundalikeUI
import SwiftUI

/// Every song the user found soundalikes for, by day. Tapping one shows its soundalikes again.
struct HistoryView: View {
    @ObservedObject private var history = HistoryStore.shared
    let open: (Seed) -> Void
    @State private var confirmClear = false

    var body: some View {
        List {
            ForEach(days, id: \.title) { day in
                Section(day.title) {
                    ForEach(day.entries) { entry in
                        Button { open(entry.seed) } label: { HistoryRow(entry: entry, timeOnly: true) }
                    }
                    .onDelete { offsets in history.remove(ids: Set(offsets.map { day.entries[$0].id })) }
                }
            }
        }
        .overlay {
            if history.entries.isEmpty {
                ContentUnavailableView("No history yet", systemImage: "clock",
                                       description: Text("Songs you find soundalikes for show up here."))
            }
        }
        .tint(.primary)
        .navigationTitle("History")
        .toolbar {
            if !history.entries.isEmpty {
                Button("Clear", role: .destructive) { confirmClear = true }
            }
        }
        .confirmationDialog("Clear your history?", isPresented: $confirmClear, titleVisibility: .visible) {
            Button("Clear History", role: .destructive) { history.clear() }
        }
        .onAppear { history.reload() }
    }

    private var days: [(title: String, entries: [HistoryEntry])] {
        let calendar = Calendar.current
        var result: [(title: String, entries: [HistoryEntry])] = []
        for entry in history.entries {
            let title = Self.dayTitle(entry.date, calendar: calendar)
            if result.last?.title == title {
                result[result.count - 1].entries.append(entry)
            } else {
                result.append((title, [entry]))
            }
        }
        return result
    }

    static func dayTitle(_ date: Date, calendar: Calendar) -> String {
        if calendar.isDateInToday(date) { return "Today" }
        if calendar.isDateInYesterday(date) { return "Yesterday" }
        let sameYear = calendar.isDate(date, equalTo: Date(), toGranularity: .year)
        return date.formatted(sameYear ? .dateTime.weekday(.wide).month(.wide).day() : .dateTime.month(.wide).day().year())
    }
}

struct HistoryRow: View {
    let entry: HistoryEntry
    /// Under a day heading only the time is needed; elsewhere older entries show their day.
    var timeOnly = false

    var body: some View {
        HStack(spacing: 12) {
            Artwork(url: entry.seed.artworkURL, size: 44)
            VStack(alignment: .leading, spacing: 2) {
                Text(entry.seed.title).font(.body.weight(.semibold)).foregroundStyle(.primary).lineLimit(1)
                Text(entry.seed.artist).font(.subheadline).foregroundStyle(.secondary).lineLimit(1)
            }
            Spacer(minLength: 8)
            Text(when).font(.caption).foregroundStyle(.secondary)
        }
    }

    private var when: String {
        let calendar = Calendar.current
        if timeOnly || calendar.isDateInToday(entry.date) { return entry.date.formatted(date: .omitted, time: .shortened) }
        if calendar.isDateInYesterday(entry.date) { return "Yesterday" }
        if let days = calendar.dateComponents([.day], from: entry.date, to: Date()).day, days < 7 {
            return entry.date.formatted(.dateTime.weekday(.wide))
        }
        return entry.date.formatted(.dateTime.month(.abbreviated).day())
    }
}
