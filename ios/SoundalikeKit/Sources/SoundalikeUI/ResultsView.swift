import SoundalikeKit
import SwiftUI

public enum ResultsState: Equatable {
    case loading(Seed?)
    case loaded(SoundalikeResults)
    case failed(String)
}

/// What the host (app or share extension) can do with a result.
@MainActor
public struct ResultActions {
    /// The row's trailing action: opens a match in the given music app (the app),
    /// or copies it (the share extension, which iOS doesn't allow to open apps).
    public var open: (Match, MusicSource) async -> Void
    public var openSymbol: String
    public var openLabel: (MusicSource) -> String
    /// Plays the full song in the music app without leaving Soundalike (Spotify
    /// connected, or an Apple Music subscription). Without it, tapping a song opens it.
    public var play: ((Match) async throws -> Void)?
    public var playMessage: String
    /// Adds a song to the music app's queue; offered on a right swipe and in the menu.
    public var queue: ((Match) async throws -> Void)?
    public var queueLabel: String
    /// Apple Music library actions, only with a subscription.
    public var playAll: (([Match]) async throws -> Void)?
    public var saveAsPlaylist: ((String, [Match]) async throws -> Void)?

    public init(
        open: @escaping (Match, MusicSource) async -> Void,
        openSymbol: String = "arrow.up.forward.app.fill",
        openLabel: @escaping (MusicSource) -> String = { "Open in \($0.displayName)" },
        play: ((Match) async throws -> Void)? = nil,
        playMessage: String = "",
        queue: ((Match) async throws -> Void)? = nil,
        queueLabel: String = "Play Next",
        playAll: (([Match]) async throws -> Void)? = nil,
        saveAsPlaylist: ((String, [Match]) async throws -> Void)? = nil
    ) {
        self.open = open
        self.openSymbol = openSymbol
        self.openLabel = openLabel
        self.play = play
        self.playMessage = playMessage
        self.queue = queue
        self.queueLabel = queueLabel
        self.playAll = playAll
        self.saveAsPlaylist = saveAsPlaylist
    }
}

/// The song the music app is playing, so the results list can mark it like Spotify does.
public struct LiveSong: Equatable, Sendable {
    /// Set while the song was just started from a result and the app hasn't reported it yet.
    public var matchID: String?
    public var title: String
    public var artist: String
    public var isPlaying: Bool

    public init(matchID: String?, title: String, artist: String, isPlaying: Bool) {
        self.matchID = matchID
        self.title = title
        self.artist = artist
        self.isPlaying = isPlaying
    }

    /// The result this is, if any: the one just started, else the first with the same title and artist.
    func row(in matches: [Match]) -> String? {
        if let matchID, matches.contains(where: { $0.id == matchID }) { return matchID }
        return matches.first { TextMatch.similar($0.title, title) && TextMatch.similar($0.artist, artist) }?.id
    }
}

public struct ResultsView: View {
    let state: ResultsState
    let actions: ResultActions
    let retry: (() -> Void)?
    let live: LiveSong?
    /// The Spotify / Apple Music switch; hidden when the user has only one app to play in.
    let showsTargetPicker: Bool
    @Binding var target: MusicSource
    @State private var toast: String?

    public init(
        state: ResultsState, target: Binding<MusicSource>, actions: ResultActions, live: LiveSong? = nil,
        showsTargetPicker: Bool = true, retry: (() -> Void)? = nil
    ) {
        self.state = state
        self._target = target
        self.actions = actions
        self.live = live
        self.showsTargetPicker = showsTargetPicker
        self.retry = retry
    }

    public var body: some View {
        List {
            switch state {
            case .loading(let seed):
                if let seed { SeedHeader(seed: seed).listRowSeparator(.hidden) }
                HStack(spacing: 12) {
                    ProgressView()
                    Text(seed == nil ? "Finding the song…" : "Listening for soundalikes…").foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, minHeight: 120)
                .listRowSeparator(.hidden)
            case .failed(let message):
                VStack(spacing: 14) {
                    Image(systemName: "waveform.slash").font(.system(size: 40)).foregroundStyle(.secondary)
                    Text(message).multilineTextAlignment(.center)
                    if let retry { Button("Try again", action: retry).buttonStyle(.borderedProminent) }
                }
                .frame(maxWidth: .infinity, minHeight: 220)
                .listRowSeparator(.hidden)
            case .loaded(let results):
                SeedHeader(seed: results.seed).listRowSeparator(.hidden)
                if showsTargetPicker {
                    targetPicker
                        .listRowSeparator(.hidden)
                }
                appleMusicBar(results.matches)
                let liveRow = live?.row(in: results.matches)
                ForEach(results.matches) { match in
                    ResultRow(
                        match: match, live: liveRow == match.id ? live : nil,
                        symbol: actions.openSymbol, label: actions.openLabel(target),
                        playFull: actions.play.map { play in { run(actions.playMessage) { try await play(match) } } }
                    ) {
                        Task { await actions.open(match, target) }
                    }
                    .contextMenu { menu(for: match) }
                    .swipeActions(edge: .leading) {
                        if let queue = actions.queue {
                            Button { run("Added to your queue") { try await queue(match) } } label: {
                                Label(actions.queueLabel, systemImage: "text.line.first.and.arrowtriangle.forward")
                            }
                            .tint(.green)
                        }
                    }
                }
                Text((actions.play == nil ? "Tap a song to open it in your music app." : "Tap a song to play it.") + (actions.queue == nil ? "" : " Swipe right to queue it."))
                    .font(.footnote).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity).listRowSeparator(.hidden)
            }
        }
        .listStyle(.plain)
        .overlay(alignment: .bottom) {
            if let toast {
                Text(toast).font(.subheadline.weight(.semibold))
                    .padding(.horizontal, 16).padding(.vertical, 10)
                    .background(.thinMaterial, in: Capsule())
                    .padding(.bottom, 24)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .animation(.spring(duration: 0.3), value: toast)
    }

    private var targetPicker: some View {
        Picker("Open results in", selection: $target) {
            Text("Spotify").tag(MusicSource.spotify)
            Text("Apple Music").tag(MusicSource.appleMusic)
        }
        .pickerStyle(.segmented)
        .accessibilityLabel("Open results in")
    }

    @ViewBuilder
    private func appleMusicBar(_ matches: [Match]) -> some View {
        if target == .appleMusic, actions.playAll != nil || actions.saveAsPlaylist != nil {
            HStack {
                if let playAll = actions.playAll {
                    Button { run("Playing in Apple Music") { try await playAll(matches) } } label: {
                        Label("Play all", systemImage: "play.fill").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                }
                if let save = actions.saveAsPlaylist, case .loaded(let results) = state {
                    Button { run("Saved to your library") { try await save("Sounds like \(results.seed.title)", matches) } } label: {
                        Label("Save playlist", systemImage: "plus.rectangle.on.rectangle").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                }
            }
            .listRowSeparator(.hidden)
        }
    }

    @ViewBuilder
    private func menu(for match: Match) -> some View {
        Button { Task { await actions.open(match, .spotify) } } label: { Label(actions.openLabel(.spotify), systemImage: actions.openSymbol) }
        Button { Task { await actions.open(match, .appleMusic) } } label: { Label(actions.openLabel(.appleMusic), systemImage: actions.openSymbol) }
        if let queue = actions.queue {
            Button { run("Added to your queue") { try await queue(match) } } label: {
                Label(actions.queueLabel, systemImage: "text.line.first.and.arrowtriangle.forward")
            }
        }
        ShareLink(item: Destinations.spotify(match)) { Label("Share", systemImage: "square.and.arrow.up") }
    }

    private func run(_ success: String, _ work: @escaping () async throws -> Void) {
        Task {
            do {
                try await work()
                show(success)
            } catch {
                show(error.localizedDescription)
            }
        }
    }

    private func show(_ message: String) {
        toast = message
        Task {
            try? await Task.sleep(for: .seconds(2.2))
            if toast == message { toast = nil }
        }
    }
}

struct SeedHeader: View {
    let seed: Seed

    var body: some View {
        HStack(spacing: 14) {
            Artwork(url: seed.artworkURL, size: 72)
            VStack(alignment: .leading, spacing: 3) {
                Text("Sounds like").font(.caption.weight(.semibold)).textCase(.uppercase).foregroundStyle(.secondary)
                Text(seed.title).font(.title3.weight(.bold)).lineLimit(2)
                Text(seed.artist).font(.subheadline).foregroundStyle(.secondary).lineLimit(1)
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 6)
        .accessibilityElement(children: .combine)
    }
}

struct ResultRow: View {
    let match: Match
    /// Set when this result is the song the music app is playing.
    let live: LiveSong?
    let symbol: String
    let label: String
    /// Plays the whole song in the music app; when nil, tapping opens the song there.
    let playFull: (() -> Void)?
    let open: () -> Void

    private func tap() {
        if let playFull { playFull() } else { open() }
    }

    var body: some View {
        HStack(spacing: 12) {
            Button(action: tap) {
                ZStack {
                    Artwork(url: match.artworkURL, size: 52)
                    if let live {
                        Circle().fill(.black.opacity(0.45)).frame(width: 30, height: 30)
                        Image(systemName: "waveform").font(.system(size: 14, weight: .bold)).foregroundStyle(Color.green)
                            .symbolEffect(.variableColor.iterative.dimInactiveLayers, options: .repeating, isActive: live.isPlaying)
                    } else if playFull != nil {
                        Circle().fill(.black.opacity(0.25)).frame(width: 30, height: 30)
                        Image(systemName: "play.fill").font(.system(size: 13, weight: .bold)).foregroundStyle(.white)
                    }
                }
            }
            .buttonStyle(.plain)
            .accessibilityLabel(playFull != nil ? "Play" : label)

            VStack(alignment: .leading, spacing: 2) {
                Text(match.title).font(.body.weight(.semibold)).lineLimit(1)
                    .foregroundStyle(live != nil ? Color.green : .primary)
                Text(match.artist).font(.subheadline).foregroundStyle(.secondary).lineLimit(1)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
            .onTapGesture(perform: tap)

            MatchBadge(value: match.match)

            Button(action: open) {
                Image(systemName: symbol).font(.title2)
                    .symbolRenderingMode(.hierarchical)
            }
            .buttonStyle(.borderless)
            .accessibilityLabel(label)
        }
        .padding(.vertical, 2)
    }
}

struct MatchBadge: View {
    let value: Int

    var body: some View {
        Text("\(value)%")
            .font(.caption.monospacedDigit().weight(.bold))
            .padding(.horizontal, 7).padding(.vertical, 4)
            .foregroundStyle(color)
            .background(color.opacity(0.15), in: Capsule())
            .accessibilityLabel("\(value) percent match")
    }

    private var color: Color {
        switch value {
        case 75...: .green
        case 50..<75: .teal
        default: .secondary
        }
    }
}

public struct Artwork: View {
    let url: URL?
    let size: CGFloat

    public init(url: URL?, size: CGFloat) {
        self.url = url
        self.size = size
    }

    public var body: some View {
        AsyncImage(url: url, transaction: Transaction(animation: .easeOut(duration: 0.2))) { phase in
            if let image = phase.image {
                image.resizable().scaledToFill()
            } else {
                ZStack {
                    LinearGradient(colors: [.purple.opacity(0.5), .indigo.opacity(0.5)], startPoint: .topLeading, endPoint: .bottomTrailing)
                    Image(systemName: "music.note").foregroundStyle(.white.opacity(0.8))
                }
            }
        }
        .frame(width: size, height: size)
        .clipShape(RoundedRectangle(cornerRadius: size * 0.16, style: .continuous))
    }
}
