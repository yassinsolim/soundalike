import ActivityKit
import AppIntents
import SwiftUI
import WidgetKit

/// The Control Center / Lock Screen / Action Button control, and the Live Activity it starts.
@main
struct SoundalikeControls: WidgetBundle {
    var body: some Widget {
        FindSoundalikesControl()
        SoundalikeLiveActivity()
    }
}

struct FindSoundalikesControl: ControlWidget {
    var body: some ControlWidgetConfiguration {
        StaticControlConfiguration(kind: "app.yassin.soundalike.control.island") {
            ControlWidgetButton(action: FindSoundalikesControlIntent()) {
                Label("Soundalike", systemImage: "waveform")
            }
        }
        .displayName("Find Soundalikes")
        .description("Shows songs that sound like what you're playing.")
    }
}

struct SoundalikeLiveActivity: Widget {
    var body: some WidgetConfiguration {
        ActivityConfiguration(for: SoundalikeActivityAttributes.self) { context in
            ActivityCard(state: context.state)
                .padding(14)
                .activityBackgroundTint(.black.opacity(0.75))
                .activitySystemActionForegroundColor(.white)
        } dynamicIsland: { context in
            DynamicIsland {
                DynamicIslandExpandedRegion(.leading) {
                    Label("Sounds like", systemImage: "waveform").font(.caption2).foregroundStyle(.secondary).labelStyle(.titleAndIcon)
                }
                DynamicIslandExpandedRegion(.trailing) {
                    Button(intent: DismissSoundalikesIntent()) { Image(systemName: "xmark.circle.fill") }
                        .buttonStyle(.plain).foregroundStyle(.secondary)
                }
                DynamicIslandExpandedRegion(.bottom) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(context.state.seedTitle).font(.caption.weight(.semibold)).lineLimit(1)
                        Rows(rows: context.state.rows, compact: true)
                    }
                }
            } compactLeading: {
                Image(systemName: "waveform").foregroundStyle(.tint)
            } compactTrailing: {
                Text("\(context.state.rows.count)").font(.caption.weight(.bold))
            } minimal: {
                Image(systemName: "waveform").foregroundStyle(.tint)
            }
            .widgetURL(AppRoute.last.url)
        }
    }
}

private struct ActivityCard: View {
    let state: SoundalikeActivityAttributes.ContentState

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("Sounds like \(state.seedTitle)", systemImage: "waveform").font(.subheadline.weight(.semibold)).lineLimit(1)
                Spacer()
                Button(intent: DismissSoundalikesIntent()) { Image(systemName: "xmark") }.buttonStyle(.plain).foregroundStyle(.secondary)
            }
            Rows(rows: state.rows, compact: false)
        }
        .foregroundStyle(.white)
    }
}

private struct Rows: View {
    let rows: [SoundalikeActivityAttributes.Row]
    /// The expanded Dynamic Island is short: one line per song so all three fit.
    let compact: Bool

    var body: some View {
        VStack(spacing: compact ? 3 : 6) {
            ForEach(rows, id: \.self) { row in
                HStack(spacing: 8) {
                    Button(intent: OpenSongIntent(url: row.openURL)) {
                        HStack(spacing: 6) {
                            if compact {
                                (Text(row.title).fontWeight(.semibold) + Text("  \(row.artist)").foregroundStyle(.secondary))
                                    .font(.caption).lineLimit(1)
                            } else {
                                VStack(alignment: .leading, spacing: 0) {
                                    Text(row.title).font(.footnote.weight(.semibold)).lineLimit(1)
                                    Text(row.artist).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                                }
                            }
                            Spacer(minLength: 4)
                            Text("\(row.match)%").font(.caption2.monospacedDigit().weight(.bold)).foregroundStyle(.secondary)
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    if let trackID = row.spotifyTrackID {
                        Button(intent: QueueInSpotifyIntent(trackID: trackID)) {
                            Image(systemName: "text.line.first.and.arrowtriangle.forward").font(.caption2)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                        .accessibilityLabel("Add to Spotify queue")
                    }
                }
            }
        }
    }
}
