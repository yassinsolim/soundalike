import MusicKit
import SoundalikeUI
import SwiftUI

/// A Control Center–style player at the bottom of the app for the song playing in
/// Spotify or the Music app: cover, title, skip back, play/pause, skip ahead and a scrubber.
struct PlayerBar: View {
    @ObservedObject private var player = PlayerController.shared
    /// Opens soundalikes for the song in the bar.
    let findSoundalikes: () -> Void
    @State private var scrubbing: TimeInterval?

    var body: some View {
        VStack {
            // A song playing in the Music app wins; otherwise offer Spotify before a paused Music song.
            if player.spotifyNeedsWake, player.now?.isPlaying != true {
                wakeRow
            } else if let now = player.now {
                bar(now)
            }
        }
        .animation(.spring(duration: 0.35), value: player.now == nil)
        .animation(.spring(duration: 0.35), value: player.spotifyNeedsWake)
    }

    private func bar(_ now: PlayerController.Snapshot) -> some View {
        Group {
            VStack(spacing: 6) {
                HStack(spacing: 12) {
                    Button(action: findSoundalikes) {
                        HStack(spacing: 12) {
                            cover(now.track.cover)
                            VStack(alignment: .leading, spacing: 1) {
                                Text(now.track.title).font(.subheadline.weight(.semibold)).lineLimit(1)
                                Text(subtitle(now)).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                            }
                            Spacer(minLength: 0)
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .accessibilityHint("Finds soundalikes for this song")

                    HStack(spacing: 18) {
                        control("backward.fill", label: "Previous", enabled: now.canSkipPrevious) { player.skipToPrevious() }
                        control(now.isPlaying ? "pause.fill" : "play.fill", label: now.isPlaying ? "Pause" : "Play", size: 26) {
                            player.togglePlay()
                        }
                        control("forward.fill", label: "Next", enabled: now.canSkipNext) { player.skipToNext() }
                    }
                }
                if now.track.duration > 0 {
                    TimelineView(.periodic(from: .now, by: 0.5)) { context in
                        scrubber(duration: now.track.duration, position: scrubbing ?? player.position(at: context.date), enabled: now.canSeek)
                    }
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .glassEffect(.regular, in: .rect(cornerRadius: 24))
            .padding(.horizontal, 12)
            .padding(.bottom, 4)
            .transition(.move(edge: .bottom).combined(with: .opacity))
        }
    }

    /// Shown when Spotify is connected but its player can't be reached yet.
    private var wakeRow: some View {
        Button { player.wakeSpotify() } label: {
            HStack(spacing: 12) {
                Image(systemName: "playpause.fill").font(.system(size: 18, weight: .semibold))
                    .frame(width: 44, height: 44)
                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                VStack(alignment: .leading, spacing: 1) {
                    Text("Show Spotify controls").font(.subheadline.weight(.semibold))
                    Text("Opens Spotify for a moment, then comes back").font(.caption).foregroundStyle(.secondary)
                }
                Spacer(minLength: 0)
                Image(systemName: "chevron.right").font(.footnote.weight(.semibold)).foregroundStyle(.tertiary)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .glassEffect(.regular, in: .rect(cornerRadius: 24))
        .padding(.horizontal, 12)
        .padding(.bottom, 4)
        .transition(.move(edge: .bottom).combined(with: .opacity))
    }

    private func subtitle(_ now: PlayerController.Snapshot) -> String {
        let app = player.source == .spotify ? "Spotify" : "Music"
        return now.track.artist.isEmpty ? app : "\(now.track.artist) · \(app)"
    }

    private func control(_ symbol: String, label: String, size: CGFloat = 20, enabled: Bool = true, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: symbol).font(.system(size: size, weight: .semibold))
                .frame(width: 32, height: 36)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
        .opacity(enabled ? 1 : 0.35)
        .accessibilityLabel(label)
    }

    @ViewBuilder
    private func cover(_ cover: PlayerController.Cover?) -> some View {
        let shape = RoundedRectangle(cornerRadius: 8, style: .continuous)
        Group {
            switch cover {
            case .image(let image): Image(uiImage: image).resizable().scaledToFill()
            case .url(let url): Artwork(url: url, size: 44)
            case .musicKit(let artwork): ArtworkImage(artwork, width: 44, height: 44)
            case nil: Image(systemName: "music.note").foregroundStyle(.secondary)
            }
        }
        .frame(width: 44, height: 44)
        .background(.quaternary)
        .clipShape(shape)
    }

    /// A thin bar like Control Center's: drag anywhere on it to scrub, it thickens while dragging.
    private func scrubber(duration: TimeInterval, position: TimeInterval, enabled: Bool) -> some View {
        let fraction = duration > 0 ? min(1, max(0, position / duration)) : 0
        let dragging = scrubbing != nil
        return VStack(spacing: 4) {
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Capsule().fill(.primary.opacity(0.18))
                    Capsule().fill(.primary.opacity(dragging ? 0.95 : 0.7))
                        .frame(width: max(0, geometry.size.width * fraction))
                }
                .frame(height: dragging ? 10 : 5)
                .frame(maxHeight: .infinity)
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { value in
                            guard enabled else { return }
                            scrubbing = duration * min(1, max(0, value.location.x / geometry.size.width))
                        }
                        .onEnded { _ in
                            if let target = scrubbing { player.seek(to: target) }
                            scrubbing = nil
                        }
                )
            }
            .frame(height: 16)
            .animation(.easeOut(duration: 0.15), value: dragging)
            HStack {
                Text(Self.time(position))
                Spacer()
                Text("-" + Self.time(max(0, duration - position)))
            }
            .font(.caption2.monospacedDigit())
            .foregroundStyle(.secondary)
        }
        .accessibilityElement()
        .accessibilityLabel("Position")
        .accessibilityValue("\(Self.time(position)) of \(Self.time(duration))")
        .accessibilityAdjustableAction { direction in
            guard enabled else { return }
            player.seek(to: min(duration, max(0, position + (direction == .increment ? 15 : -15))))
        }
    }

    static func time(_ seconds: TimeInterval) -> String {
        let total = Int(seconds.rounded(.down))
        return String(format: "%d:%02d", total / 60, total % 60)
    }
}
