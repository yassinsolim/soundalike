import AppIntents
import SoundalikeKit
import SwiftUI
import UIKit

/// One result, prepared ahead so the system card renders instantly: snippets can't
/// load remote images, so the cover is fetched here.
struct SnippetRow: Identifiable, Codable {
    let match: Match
    let coverData: Data?
    let openURL: URL
    let spotifyTrackID: String?

    var id: String { match.id }
    var cover: UIImage? { coverData.flatMap(UIImage.init(data:)) }

    static let count = 5

    @MainActor
    static func make(from results: SoundalikeResults) async -> [SnippetRow] {
        let spotify = SpotifyAccount()
        let connected = spotify.isConnected
        let destinations = Destinations()
        let source = results.seed.source
        return await withTaskGroup(of: (Int, SnippetRow).self) { group in
            for (index, match) in results.matches.prefix(count).enumerated() {
                group.addTask {
                    async let cover = image(match.artworkURL)
                    var trackID: String?
                    if connected, let isrc = match.isrc { trackID = try? await spotify.trackID(isrc: isrc) }
                    let url: URL
                    if source == .appleMusic {
                        url = await destinations.appleMusic(match)
                    } else if let trackID {
                        url = URL(string: "https://open.spotify.com/track/\(trackID)")!
                    } else {
                        url = Destinations.spotify(match)
                    }
                    return (index, SnippetRow(match: match, coverData: await cover, openURL: url, spotifyTrackID: trackID))
                }
            }
            var rows: [(Int, SnippetRow)] = []
            for await row in group { rows.append(row) }
            return rows.sorted { $0.0 < $1.0 }.map(\.1)
        }
    }

    private static func image(_ url: URL?) async -> Data? {
        guard let url, let (data, _) = try? await URLSession.shared.data(from: url) else { return nil }
        return UIImage(data: data)?.preparingThumbnail(of: CGSize(width: 96, height: 96))?.jpegData(compressionQuality: 0.8)
    }
}

/// What the card shows, saved by `FindSoundalikesIntent` and read by the snippet
/// intent, which the system may run again whenever it redraws the card.
struct SnippetCard: Codable {
    let seed: Seed
    let rows: [SnippetRow]

    private static var url: URL? {
        FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: PendingRoute.appGroup)?
            .appendingPathComponent("Library/Caches/snippet.json")
    }

    func save() {
        guard let url = Self.url, let data = try? JSONEncoder().encode(self) else { return }
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? data.write(to: url, options: .atomic)
    }

    static func load() -> SnippetCard? {
        guard let url, let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(SnippetCard.self, from: data)
    }
}

/// Draws the card over the current app (Control Center, Action Button, Siri).
struct SoundalikesSnippetIntent: SnippetIntent {
    static let title: LocalizedStringResource = "Soundalikes"
    static let isDiscoverable = false

    @MainActor
    func perform() async throws -> some IntentResult & ShowsSnippetView {
        guard let card = SnippetCard.load() else { return .result(view: AnyView(EmptyView())) }
        return .result(view: AnyView(SoundalikesSnippet(seed: card.seed, rows: card.rows)))
    }
}

struct SoundalikesSnippet: View {
    let seed: Seed
    let rows: [SnippetRow]

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(rows) { row in
                HStack(spacing: 10) {
                    Button(intent: OpenSongIntent(url: row.openURL)) {
                        HStack(spacing: 10) {
                            cover(row.cover)
                            VStack(alignment: .leading, spacing: 1) {
                                Text(row.match.title).font(.subheadline.weight(.semibold)).lineLimit(1)
                                Text(row.match.artist).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                            }
                            Spacer(minLength: 4)
                            Text("\(row.match.match)%").font(.caption.monospacedDigit().weight(.bold)).foregroundStyle(.secondary)
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    if let trackID = row.spotifyTrackID {
                        Button(intent: QueueInSpotifyIntent(trackID: trackID)) {
                            Image(systemName: "text.line.first.and.arrowtriangle.forward")
                        }
                        .buttonStyle(.bordered)
                        .accessibilityLabel("Add to Spotify queue")
                    }
                }
            }
            Button(intent: ShowAllIntent()) {
                Label("See all in Soundalike", systemImage: "list.bullet").frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
        }
        .padding()
    }

    @ViewBuilder
    private func cover(_ image: UIImage?) -> some View {
        Group {
            if let image {
                Image(uiImage: image).resizable().scaledToFill()
            } else {
                Image(systemName: "music.note").frame(maxWidth: .infinity, maxHeight: .infinity).background(.quaternary)
            }
        }
        .frame(width: 40, height: 40)
        .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
    }
}
