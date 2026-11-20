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
                .padding(.horizontal, 14).padding(.vertical, 10)
                .activityBackgroundTint(.black.opacity(0.8))
                .activitySystemActionForegroundColor(.white)
        } dynamicIsland: { context in
            DynamicIsland {
                DynamicIslandExpandedRegion(.leading) {
                    Image(systemName: "waveform").foregroundStyle(.tint)
                }
                DynamicIslandExpandedRegion(.center) {
                    Text("Sounds like \(context.state.seedTitle)").font(.caption.weight(.semibold)).lineLimit(1)
                }
                DynamicIslandExpandedRegion(.trailing) {
                    Button(intent: DismissSoundalikesIntent()) { Image(systemName: "xmark.circle.fill") }
                        .buttonStyle(.plain).foregroundStyle(.secondary)
                }
                DynamicIslandExpandedRegion(.bottom) {
                    Rows(rows: context.state.rows)
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
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Label("Sounds like \(state.seedTitle)", systemImage: "waveform").font(.caption.weight(.semibold)).lineLimit(1)
                Spacer()
                Button(intent: DismissSoundalikesIntent()) { Image(systemName: "xmark.circle.fill") }.buttonStyle(.plain).foregroundStyle(.secondary)
            }
            Rows(rows: state.rows)
        }
        .foregroundStyle(.white)
    }
}

/// Live Activities are capped at 160 points tall, so five songs get one roomy line each.
private struct Rows: View {
    let rows: [SoundalikeActivityAttributes.Row]

    var body: some View {
        VStack(spacing: 2) {
            ForEach(rows, id: \.self) { row in
                HStack(spacing: 8) {
                    Button(intent: OpenSongIntent(url: row.openURL)) {
                        HStack(spacing: 6) {
                            (Text(row.title).fontWeight(.semibold) + Text("  \(row.artist)").foregroundStyle(.secondary))
                                .font(.subheadline).lineLimit(1)
                            Spacer(minLength: 4)
                            Text("\(row.match)%").font(.caption.monospacedDigit().weight(.bold)).foregroundStyle(.secondary)
                        }
                        .frame(minHeight: 24)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    if let trackID = row.spotifyTrackID {
                        Button(intent: QueueInSpotifyIntent(trackID: trackID)) {
                            Image(systemName: "text.line.first.and.arrowtriangle.forward").font(.caption)
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.tint)
                        .accessibilityLabel("Add to Spotify queue")
                    }
                }
            }
        }
    }
}
