import SwiftUI

/// Auto-detect, the Spotify connection and the how-to guide, out of the way of the home screen.
struct SettingsView: View {
    @Binding var autoDetect: Bool
    @State private var showGuide = false
    @Environment(\.dismiss) private var dismiss
    @Environment(\.webAuthenticationSession) private var webAuthenticationSession
    @ObservedObject private var spotify = SpotifyConnector.shared

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Toggle("Find what's playing on open", isOn: $autoDetect)
                } footer: {
                    Text("Opening Soundalike shows soundalikes for the song playing in Spotify or Apple Music.")
                }
                Section {
                    if spotify.isConnected {
                        Label("Spotify connected", systemImage: "checkmark.circle.fill")
                        Button("Disconnect Spotify", role: .destructive) { spotify.disconnect() }
                    } else {
                        Button { Task { await spotify.connect(using: webAuthenticationSession) } } label: {
                            Label("Connect Spotify", systemImage: "link")
                        }
                    }
                } header: {
                    Text("Spotify")
                } footer: {
                    Text(spotify.isConnected
                         ? "Soundalike reads what Spotify is playing, controls it from the player bar and plays results there."
                         : "One tap in the Spotify app, no password. Soundalike can then read what Spotify is playing, even in headphones.")
                }
                Section {
                    Button { showGuide = true } label: {
                        Label("Use it from Spotify and Apple Music", systemImage: "square.and.arrow.up")
                    }
                }
            }
            .sheet(isPresented: $showGuide) { GuideView() }
            // The home screen's alert can't show over this sheet, so Spotify failures surface here.
            .alert("Spotify", isPresented: Binding(get: { spotify.failure != nil }, set: { if !$0 { spotify.failure = nil } })) {
                Button("OK", role: .cancel) {}
            } message: { Text(spotify.failure ?? "") }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
        }
    }
}
