import SwiftUI

/// First-run guide: how to reach Soundalike from the music apps people already use.
struct GuideView: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section("From Spotify") {
                    Step(number: 1, text: "On any song, tap ••• and then Share.")
                    Step(number: 2, text: "Tap More, then Soundalike.")
                    Step(number: 3, text: "Tip: in More, tap Edit and add Soundalike to Favorites so it's always first.")
                }
                Section("From Apple Music") {
                    Step(number: 1, text: "On any song, tap ••• and then Share.")
                    Step(number: 2, text: "Tap Soundalike in the row of apps.")
                }
                Section("Without touching your music app") {
                    Step(number: 1, text: "Say \"Hey Siri, Soundalike this song\" to recognise what's playing.")
                    Step(number: 2, text: "Add Recognise Song to the Action Button or Control Center: Settings › Action Button, or edit Control Center and search for Soundalike.")
                }
                Section {
                    Text("Tap a result to hear a 30-second preview. Tap the arrow to open it in your music app.")
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("How to use Soundalike")
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }
    }
}

private struct Step: View {
    let number: Int
    let text: String

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text("\(number)").font(.subheadline.weight(.bold)).foregroundStyle(.white)
                .frame(width: 24, height: 24).background(Color.accentColor, in: Circle())
            Text(text)
        }
    }
}
