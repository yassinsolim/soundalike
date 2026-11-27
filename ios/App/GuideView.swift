import SwiftUI

/// First-run guide: the quickest ways to reach Soundalike from where people listen.
struct GuideView: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Step(symbol: "waveform", text: "Open Soundalike while music plays. It reads the song from Apple Music, or from Spotify once you tap Connect Spotify, and listens for anything else playing out loud.")
                    Step(symbol: "switch.2", text: "One press from anywhere: add Soundalike to Control Center, or to the Action Button in Settings › Action Button › Controls.")
                    Step(symbol: "hand.tap", text: "Or tap the back of your phone: Settings › Accessibility › Touch › Back Tap › Double Tap › Find Soundalikes.")
                } header: {
                    Text("Find what's playing")
                }
                Section {
                    Step(symbol: "square.and.arrow.up", text: "Spotify: tap Share › More › Soundalike. In More, tap Edit and add Soundalike to Favorites so it's always in the first row.")
                    Step(symbol: "square.and.arrow.up", text: "Apple Music: tap ••• › Share › Soundalike.")
                } header: {
                    Text("Share a song")
                } footer: {
                    Text("Handy with headphones on if you haven't connected Spotify.")
                }
                Section {
                    Text("Tap a song to play it in your music app. The arrow opens it there.")
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("How to use Soundalike")
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }
    }
}

private struct Step: View {
    let symbol: String
    let text: String

    var body: some View {
        Label {
            Text(text)
        } icon: {
            Image(systemName: symbol).foregroundStyle(Color.accentColor)
        }
    }
}
