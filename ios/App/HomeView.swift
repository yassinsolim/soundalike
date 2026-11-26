import AuthenticationServices
import MusicKit
import SoundalikeKit
import SoundalikeUI
import SwiftUI

struct ResultsRequest: Identifiable, Hashable {
    enum Kind: Hashable {
        case shared(String)
        case seed(Seed)
    }

    let id = UUID()
    let kind: Kind
}

/// Screens pushed on the home navigation stack.
enum Screen: Hashable {
    case results(ResultsRequest)
    case history
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
    @State private var path: [Screen] = []
    @State private var showGuide = false
    @State private var showSettings = false
    @State private var error: String?
    @State private var hint: Hint?
    @State private var lastAuto = Date.distantPast
    /// True at launch and after the app was in the background; Control Center or a
    /// notification only make it inactive, which shouldn't re-run auto-detect.
    @State private var returning = true
    /// The song behind results that auto-detect opened, so returning to the app can follow a song change.
    @State private var autoSeedKey: String?
    @ObservedObject private var spotify = SpotifyConnector.shared
    @ObservedObject private var history = HistoryStore.shared

    enum Hint {
        case headphones
        case nothingPlaying
    }

    var body: some View {
        NavigationStack(path: $path) {
            List {
                if query.isEmpty {
                    if let hint { HintCard(hint: hint) { self.hint = nil } }
                    quickActions
                    recent
                } else if SharedLink.find(in: query) != nil {
                    Button { path.append(.results(ResultsRequest(kind: .shared(query)))) } label: {
                        ActionRow(title: "Find soundalikes for this link", detail: "Spotify or Apple Music song", symbol: "link", tint: .indigo)
                    }
                } else {
                    ForEach(suggestions, id: \.self) { seed in
                        Button { path.append(.results(ResultsRequest(kind: .seed(seed)))) } label: {
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
            .tint(.primary)
            .navigationTitle("Soundalike")
            .searchable(text: $query, placement: .navigationBarDrawer(displayMode: .always), prompt: "Song, artist or link")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button { showSettings = true } label: { Image(systemName: "gearshape") }
                        .accessibilityLabel("Settings")
                }
            }
            .sheet(isPresented: $showSettings) { SettingsView(autoDetect: $autoDetect) }
            .onSubmit(of: .search) {
                if SharedLink.find(in: query) != nil { path.append(.results(ResultsRequest(kind: .shared(query)))) }
            }
            .task(id: query) { await suggest() }
            .navigationDestination(for: Screen.self) { screen in
                switch screen {
                case .results(let request): ResultsScreen(request: request)
                case .history: HistoryView { path.append(.results(ResultsRequest(kind: .seed($0)))) }
                }
            }
            .overlay { if listener.isListening { ListeningOverlay { listener.cancel() } } }
            .onChange(of: spotify.failure) { _, failure in
                if let failure, !showSettings { error = failure; spotify.failure = nil }
            }
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
                if phase == .background {
                    returning = true
                    PlayerController.shared.deactivate()
                }
                guard phase == .active, returning else { return }
                returning = false
                PlayerController.shared.activate()
                history.reload()
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
        .safeAreaInset(edge: .bottom, spacing: 0) {
            PlayerBar(findSoundalikes: findForPlayer)
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
            if !spotify.isConnected {
                Button { Task { await spotify.connect(using: webAuthenticationSession) } } label: {
                    ActionRow(title: "Connect Spotify", detail: "Read Spotify songs, even in headphones", symbol: "link", tint: .green)
                }
            }
        }
    }

    @ViewBuilder
    private var recent: some View {
        if !history.entries.isEmpty {
            Section {
                ForEach(history.entries.prefix(4)) { entry in
                    Button { path.append(.results(ResultsRequest(kind: .seed(entry.seed)))) } label: { HistoryRow(entry: entry) }
                }
            } header: {
                HStack {
                    Text("Recent")
                    Spacer()
                    Button("See All") { path.append(.history) }.font(.subheadline).textCase(nil).foregroundStyle(Color.accentColor)
                }
            }
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
        case .find(let shared):
            path = [.results(ResultsRequest(kind: .shared(shared)))]
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
                    path = [.results(ResultsRequest(kind: .seed(seed)))]
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
        path = [.results(ResultsRequest(kind: .seed(seed)))]
    }

    /// Soundalikes for the song in the player bar.
    private func findForPlayer() {
        guard let player = PlayerController.shared.now else { return }
        Task {
            if PlayerController.shared.source == .appleMusic, let seed = await NowPlaying.appleMusic(requireAuthorization: true) {
                show(seed)
            } else if let seed = player.track.seed {
                show(seed)
            }
        }
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
    @ObservedObject private var spotify = SpotifyConnector.shared
    @ObservedObject private var playing = PlayerController.shared
    /// Queue and playlist actions need an Apple Music subscription; without one they stay hidden.
    @State private var canUseAppleMusic = false
    @State private var checkedAppleMusic = false

    var body: some View {
        ResultsView(
            state: model.state, target: $model.target, actions: actions, live: playing.live,
            // One app to play in (Spotify connected, or an Apple Music subscription): no switch.
            // It's offered only to people who can play in both, or in neither.
            showsTargetPicker: checkedAppleMusic && spotify.isConnected == canUseAppleMusic,
            retry: { model.retry() }
        )
            .navigationTitle("Soundalikes")
            .navigationBarTitleDisplayMode(.inline)
            .task {
                if MusicAuthorization.currentStatus == .authorized {
                    canUseAppleMusic = (try? await MusicSubscription.current.canPlayCatalogContent) ?? false
                }
                checkedAppleMusic = true
                applyPreferredApp()
            }
            // Connect Spotify can finish after these results opened (on Apple Music, say).
            .onChange(of: spotify.isConnected) { applyPreferredApp() }
            .task {
                guard case .loading(nil) = model.state else { return }
                applyPreferredApp()
                switch request.kind {
                case .shared(let text): model.load(shared: text)
                case .seed(let seed): model.load(seed: seed)
                }
            }
            .task(id: PrefetchKey(ids: matches.map(\.id), target: model.target, spotify: spotify.isConnected, music: canUseAppleMusic)) {
                // Look the list up in the music app now, so tapping a result starts it at once.
                switch model.target {
                case .spotify where spotify.isConnected: _ = await SpotifyAccount().trackIDs(for: matches)
                case .appleMusic where canUseAppleMusic: _ = await AppleMusicService.shared.songs(for: matches)
                default: break
                }
            }
    }

    /// Results open in the app the user plays in, whatever app the song came from: Spotify once
    /// it's connected, else Apple Music for a subscriber. With neither, the song's own app.
    private func applyPreferredApp() {
        model.preferred = spotify.isConnected ? .spotify : canUseAppleMusic ? .appleMusic : nil
    }

    private struct PrefetchKey: Equatable {
        var ids: [String]
        var target: MusicSource
        var spotify: Bool
        var music: Bool
    }

    private var matches: [Match] {
        if case .loaded(let results) = model.state { results.matches } else { [] }
    }

    private var actions: ResultActions {
        let music = AppleMusicService.shared
        let open: (Match, MusicSource) async -> Void = { match, target in
            let url: URL
            if target == .appleMusic {
                let viaMusicKit = MusicAuthorization.currentStatus == .authorized ? await music.url(for: match) : nil
                if let viaMusicKit { url = viaMusicKit } else { url = await Destinations().appleMusic(match) }
            } else if spotify.isConnected, let id = try? await SpotifyAccount().trackID(for: match) {
                url = URL(string: "https://open.spotify.com/track/\(id)")!
            } else {
                url = Destinations.spotify(match)
            }
            await Opener.open(url)
        }
        switch model.target {
        case .spotify where spotify.isConnected:
            return ResultActions(
                open: open,
                // The whole list goes to Spotify, so skipping anywhere walks it.
                play: { match in
                    try await shownInBar(match, .spotify) { try await logged("spotify-play") {
                        guard let id = try await SpotifyAccount().trackID(for: match) else { throw SoundalikeError.notInCatalog }
                        var seen = Set<String>()
                        let list = await SpotifyAccount().trackIDs(for: matches).compactMap { $0 }.filter { seen.insert($0).inserted }
                        do {
                            try await SpotifyAccount().play(trackIDs: list.contains(id) ? list : [id], startingAt: id)
                        } catch SpotifyError.noActiveDevice {
                            // Spotify isn't running anywhere: open the song there instead.
                            await Opener.open(URL(string: "https://open.spotify.com/track/\(id)")!)
                        }
                    } }
                },
                playMessage: "Playing in Spotify",
                queue: { match in
                    try await logged("spotify-queue") {
                        guard let id = try await SpotifyAccount().trackID(for: match) else { throw SoundalikeError.notInCatalog }
                        try await SpotifyAccount().queue(trackID: id)
                    }
                },
                queueLabel: "Add to Spotify Queue"
            )
        case .appleMusic where canUseAppleMusic:
            return ResultActions(
                open: open,
                play: { match in try await shownInBar(match, .appleMusic) { try await logged("apple-play") { try await music.play(match, in: matches) } } },
                playMessage: "Playing in Apple Music",
                queue: { match in try await logged("play-next") { try await music.playNext(match) } },
                queueLabel: "Play Next",
                playAll: { matches in try await logged("play-all") { try await music.playAll(matches) } },
                saveAsPlaylist: { name, matches in try await logged("save-playlist") { try await music.saveAsPlaylist(name: name, matches) } }
            )
        default:
            return ResultActions(open: open)
        }
    }
}

/// Puts a tapped result in the player bar straight away, then has the bar read the music app
/// back as soon as the song has started.
private func shownInBar(_ match: Match, _ source: PlayerController.Source, _ play: () async throws -> Void) async throws {
    let request = await PlayerController.shared.starting(match, in: source)
    do {
        try await play()
        await PlayerController.shared.started(request, ok: true)
    } catch {
        await PlayerController.shared.started(request, ok: false)
        throw error
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
