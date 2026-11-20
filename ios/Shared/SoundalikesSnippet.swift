import AppIntents
import SoundalikeKit
import SwiftUI
import UIKit

/// One result, prepared ahead so the system card renders instantly: snippets can't
/// load remote images, so the cover is fetched here.
struct SnippetRow: Identifiable {
    let match: Match
    let cover: UIImage?
    let openURL: URL
    let spotifyTrackID: String?

    var id: String { match.id }

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
                    return (index, SnippetRow(match: match, cover: await cover, openURL: url, spotifyTrackID: trackID))
                }
            }
            var rows: [(Int, SnippetRow)] = []
            for await row in group { rows.append(row) }
            return rows.sorted { $0.0 < $1.0 }.map(\.1)
        }
    }

    private static func image(_ url: URL?) async -> UIImage? {
        guard let url, let (data, _) = try? await URLSession.shared.data(from: url) else { return nil }
        return UIImage(data: data)?.preparingThumbnail(of: CGSize(width: 96, height: 96))
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
