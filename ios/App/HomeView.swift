import AuthenticationServices
import MusicKit
import SoundalikeKit
import SoundalikeUI
import SwiftUI

struct ResultsRequest: Identifiable, Hashable {
    enum Kind: Hashable {
        case shared(String)
        case seed(Seed)
        case results(SoundalikeResults)
    }

    let id = UUID()
    let kind: Kind
}

struct HomeView: View {
    @EnvironmentObject private var router: Router
    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.webAuthenticationSession) private var webAuthenticationSession
    @StateObject private var listener = ListenService()
    @AppStorage("onboarded") private var onboarded = false
    @AppStorage("autoDetect") private var autoDetect = true
    @State private var query = ""
    @State private var suggestions: [Seed] = []
    @State private var path: [ResultsRequest] = []
    @State private var showGuide = false
    @State private var error: String?
    @State private var hint: Hint?
    @State private var lastAuto = Date.distantPast
    /// The song behind results that auto-detect opened, so returning to the app can follow a song change.
    @State private var autoSeedKey: String?
    @State private var spotifyConnected = SpotifyAccount().isConnected

    enum Hint {
        case headphones
        case nothingPlaying
    }

    var body: some View {
        NavigationStack(path: $path) {
            List {
                if query.isEmpty {
                    hero
                    if let hint { HintCard(hint: hint) { self.hint = nil } }
                    quickActions
                    Section {
                        Toggle("Find what's playing when I open Soundalike", isOn: $autoDetect)
                    }
                    spotifySection
                    Section {
                        Button { showGuide = true } label: {
                            Label("Use it from Spotify and Apple Music", systemImage: "square.and.arrow.up")
                        }
                    }
                } else {
                    ForEach(suggestions, id: \.self) { seed in
                        Button { path.append(ResultsRequest(kind: .seed(seed))) } label: {
                            HStack(spacing: 12) {
                                Artwork(url: seed.artworkURL, size: 44)
                                VStack(alignment: .leading) {
                                    Text(seed.title).font(.body.weight(.semibold)).foregroundStyle(.primary)
                                    Text(seed.artist).font(.subheadline).foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }
            }
            .navigationTitle("Soundalike")
            .searchable(text: $query, prompt: "Song or artist")
            .onSubmit(of: .search) {
                if SharedLink.find(in: query) != nil { path.append(ResultsRequest(kind: .shared(query))) }
            }
            .task(id: query) { await suggest() }
            .navigationDestination(for: ResultsRequest.self) { ResultsScreen(request: $0) }
            .overlay { if listener.isListening { ListeningOverlay { listener.cancel() } } }
            .alert("Soundalike", isPresented: Binding(get: { error != nil }, set: { if !$0 { error = nil } })) {
                Button("OK", role: .cancel) {}
            } message: { Text(error ?? "") }
            .sheet(isPresented: $showGuide) { GuideView() }
            .onAppear { if !onboarded { showGuide = true; onboarded = true } }
            .onChange(of: path) { _, newPath in
                if newPath.isEmpty { autoSeedKey = nil }
            }
            .onChange(of: router.pending) { _, route in
                guard let route else { return }
                router.pending = nil
                EventLog.record("route", ["route": route.url.absoluteString])
                handle(route, explicit: true)
            }
            .onChange(of: scenePhase, initial: true) { _, phase in
                guard phase == .active else { return }
                router.takePending()
                // Opening the app finds what's playing, unless a link or intent already asked for something.
                Task {
                    try? await Task.sleep(for: .milliseconds(300))
                    guard autoDetect, onboarded, !showGuide, router.pending == nil, query.isEmpty, !listener.isListening else { return }
                    if path.isEmpty {
                        if Date().timeIntervalSince(lastAuto) > 30 { handle(.auto, explicit: false) }
                    } else if autoSeedKey != nil {
                        // Back from the music app: follow the song if it changed.
                        if let seed = await NowPlaying.read() {
                            if key(seed) != autoSeedKey { show(seed) }
                        } else if Date().timeIntervalSince(lastAuto) > 30, await PlaybackProbe.detect() == .speaker {
                            handle(.auto, explicit: false)
                        }
                    }
                }
            }
        }
    }

    private var hero: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Find songs that sound like the ones you love.")
                .font(.title2.weight(.bold))
            Text("Search for a song, recognise one that's playing, or share one from your music app.")
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 8)
        .listRowSeparator(.hidden)
    }

    private var spotifySection: some View {
        Section {
            if spotifyConnected {
                Label("Spotify connected", systemImage: "checkmark.circle.fill")
                Button("Disconnect Spotify", role: .destructive) {
                    SpotifyAccount().disconnect()
                    spotifyConnected = false
                }
            } else {
                Button { Task { await connectSpotify() } } label: {
                    Label("Connect Spotify", systemImage: "link")
                }
            }
        } footer: {
            Text("Lets Soundalike read the song Spotify is playing, even in headphones, and add results to your Spotify queue. Beta: only invited Spotify accounts can connect.")
        }
    }

    private func connectSpotify() async {
        let authorization = SpotifyAccount.authorization()
        do {
            let callback = try await webAuthenticationSession.authenticate(
                using: authorization.url, callbackURLScheme: SpotifyAccount.callbackScheme, preferredBrowserSession: .shared
            )
            try await SpotifyAccount().complete(authorization, callback: callback)
            spotifyConnected = true
            EventLog.record("spotify-connect", ["ok": true])
        } catch ASWebAuthenticationSessionError.canceledLogin {
        } catch {
            EventLog.record("spotify-connect", ["ok": false, "error": "\(error)"])
            self.error = error.localizedDescription
        }
    }

    private var quickActions: some View {
        Section {
            Button { handle(.auto, explicit: true) } label: {
                ActionRow(title: "What's playing now", detail: "From the Music app, or by listening", symbol: "waveform", tint: .blue)
            }
            Button { handle(.listen, explicit: true) } label: {
                ActionRow(title: "Listen with the microphone", detail: "For music playing around you", symbol: "mic.fill", tint: .orange)
            }
            PasteButton(payloadType: String.self) { strings in
                guard let text = strings.first else { return }
                Task { @MainActor in
                    if SharedLink.find(in: text) != nil {
                        path.append(ResultsRequest(kind: .shared(text)))
                    } else {
                        error = SoundalikeError.unsupportedLink.localizedDescription
                    }
                }
            }
            .labelStyle(.titleAndIcon)
        } footer: {
            Text("Paste a Spotify or Apple Music song link.")
        }
    }

    private func handle(_ route: AppRoute, explicit: Bool, privateOutput: Bool = false) {
        hint = nil
        switch route {
        case .auto:
            lastAuto = Date()
            Task {
                let started = Date()
                let finding = await PlaybackProbe.detect()
                if let seed = await NowPlaying.read() {
                    EventLog.record("auto", ["finding": "read", "source": seed.source.rawValue, "ms": EventLog.ms(since: started), "explicit": explicit])
                    show(seed)
                    return
                }
                EventLog.record("auto", ["finding": finding.rawValue, "explicit": explicit])
                switch finding {
                case .speaker: handle(.listen, explicit: explicit)
                case .external: handle(.listen, explicit: explicit, privateOutput: true)
                case .silence: if explicit { hint = .nothingPlaying }
                }
            }
        case .last:
            if let results = LastResults.load() { path = [ResultsRequest(kind: .results(results))] }
        case .find(let shared):
            path = [ResultsRequest(kind: .shared(shared))]
        case .search(let text):
            path = []
            query = text
        case .nowPlaying:
            Task {
                let started = Date()
                do {
                    let seed = try await AppleMusicService.shared.currentSeed()
                    EventLog.record("now-playing", ["ms": EventLog.ms(since: started), "ok": true, "isrc": seed.isrc ?? ""])
                    show(seed)
                } catch {
                    EventLog.record("now-playing", ["ms": EventLog.ms(since: started), "ok": false, "error": "\(error)"])
                    self.error = error.localizedDescription
                }
            }
        case .listen:
            guard !listener.isListening else { return }
            Task {
                let started = Date()
                do {
                    let seed = try await listener.identify()
                    EventLog.record("listen", ["ms": EventLog.ms(since: started), "ok": true, "seed": "\(seed.title) — \(seed.artist)"])
                    path = [ResultsRequest(kind: .seed(seed))]
                } catch is CancellationError {
                } catch {
                    EventLog.record("listen", ["ms": EventLog.ms(since: started), "ok": false, "error": "\(error)", "privateOutput": privateOutput])
                    if privateOutput, case ListenService.Failure.noMatch = error {
                        hint = .headphones
                    } else {
                        self.error = error.localizedDescription
                    }
                }
            }
        }
    }

    private func key(_ seed: Seed) -> String {
        TextMatch.normalize(seed.title) + "|" + TextMatch.normalize(seed.artist)
    }

    /// Results for a song that was read or heard rather than picked.
    private func show(_ seed: Seed) {
        autoSeedKey = key(seed)
        path = [ResultsRequest(kind: .seed(seed))]
    }

    private func suggest() async {
        let text = query
        guard !text.isEmpty, SharedLink.find(in: text) == nil else {
            suggestions = []
            return
        }
        try? await Task.sleep(for: .milliseconds(250))
        guard !Task.isCancelled else { return }
        if let found = try? await DeezerCatalog().search(text), !Task.isCancelled {
            suggestions = found
        }
    }
}

struct HintCard: View {
    let hint: HomeView.Hint
    let dismiss: () -> Void

    var body: some View {
        Section {
            VStack(alignment: .leading, spacing: 8) {
                Label(title, systemImage: hint == .headphones ? "headphones" : "speaker.slash")
                    .font(.headline)
                Text(message).font(.subheadline).foregroundStyle(.secondary)
            }
            .padding(.vertical, 4)
            .swipeActions { Button("Hide", action: dismiss) }
        }
    }

    private var title: String {
        hint == .headphones ? "Playing in your headphones?" : "Nothing is playing"
    }

    private var message: String {
        switch hint {
        case .headphones:
            "Soundalike couldn't hear it. iOS doesn't let apps read what other apps play into headphones. In Spotify, tap Share › Soundalike, or Share › Copy link and paste it here. Apple Music songs are found automatically."
        case .nothingPlaying:
            "Play a song in Apple Music or Spotify and try again, or search for one."
        }
    }
}

struct ActionRow: View {
    let title: String
    let detail: String
    let symbol: String
    let tint: Color

    var body: some View {
        HStack(spacing: 14) {
            Image(systemName: symbol).font(.title3.weight(.semibold)).foregroundStyle(.white)
                .frame(width: 42, height: 42)
                .background(tint.gradient, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.body.weight(.semibold)).foregroundStyle(.primary)
                Text(detail).font(.subheadline).foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
    }
}

struct ListeningOverlay: View {
    let cancel: () -> Void
    @State private var pulse = false

    var body: some View {
        ZStack {
            Rectangle().fill(.ultraThinMaterial).ignoresSafeArea()
            VStack(spacing: 24) {
                Image(systemName: "waveform").font(.system(size: 56, weight: .semibold)).foregroundStyle(.blue)
                    .scaleEffect(pulse ? 1.12 : 0.92)
                    .animation(.easeInOut(duration: 0.7).repeatForever(), value: pulse)
                Text("Listening…").font(.title2.weight(.bold))
                Text("Hold your phone near the music.").foregroundStyle(.secondary)
                Button("Cancel", role: .cancel, action: cancel).buttonStyle(.bordered)
            }
        }
        .onAppear { pulse = true }
    }
}

struct ResultsScreen: View {
    let request: ResultsRequest
    @StateObject private var model = ResultsModel()
    /// Queue and playlist actions need an Apple Music subscription; without one they stay hidden.
    @State private var canUseAppleMusic = false

    var body: some View {
        ResultsView(state: model.state, target: $model.target, actions: actions, retry: { model.retry() })
            .navigationTitle("Soundalikes")
            .navigationBarTitleDisplayMode(.inline)
            .task {
                if MusicAuthorization.currentStatus == .authorized {
                    canUseAppleMusic = (try? await MusicSubscription.current.canPlayCatalogContent) ?? false
                }
            }
            .task {
                guard case .loading(nil) = model.state else { return }
                switch request.kind {
                case .shared(let text): model.load(shared: text)
                case .seed(let seed): model.load(seed: seed)
                case .results(let results): model.show(results)
                }
            }
    }

    private var actions: ResultActions {
        let music = AppleMusicService.shared
        return ResultActions(
            open: { match, target in
                let url: URL
                if target == .appleMusic {
                    let viaMusicKit = MusicAuthorization.currentStatus == .authorized ? await music.url(for: match) : nil
                    if let viaMusicKit { url = viaMusicKit } else { url = await Destinations().appleMusic(match) }
                } else {
                    url = Destinations.spotify(match)
                }
                await Opener.open(url)
            },
            playNext: canUseAppleMusic ? { match in try await logged("play-next") { try await music.playNext(match) } } : nil,
            playAll: canUseAppleMusic ? { matches in try await logged("play-all") { try await music.playAll(matches) } } : nil,
            saveAsPlaylist: canUseAppleMusic ? { name, matches in try await logged("save-playlist") { try await music.saveAsPlaylist(name: name, matches) } } : nil
        )
    }
}

private func logged(_ event: String, _ work: () async throws -> Void) async throws {
    let started = Date()
    do {
        try await work()
        EventLog.record(event, ["ms": EventLog.ms(since: started), "ok": true])
    } catch {
        EventLog.record(event, ["ms": EventLog.ms(since: started), "ok": false, "error": "\(error)"])
        throw error
    }
}

/// Opens a music link: in the app that claims it when there is one, otherwise in Safari.
@MainActor
enum Opener {
    static func open(_ url: URL) async {
        let inApp = await UIApplication.shared.open(url, options: [.universalLinksOnly: true])
        let opened = inApp ? true : await UIApplication.shared.open(url)
        EventLog.record("open", ["url": url.absoluteString, "universalLink": inApp, "opened": opened])
    }
}
