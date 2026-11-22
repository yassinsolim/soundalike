import Combine
import MusicKit
import SoundalikeKit
import SoundalikeUI
import SpotifyiOS
import UIKit

/// The song playing in Spotify or the Music app, with play, pause, skip and seek:
/// a remote control for the player bar. Soundalike plays nothing itself here.
@MainActor
final class PlayerController: ObservableObject {
    static let shared = PlayerController()

    enum Source { case spotify, appleMusic }

    enum Cover: Equatable {
        case image(UIImage)
        case url(URL)
        case musicKit(MusicKit.Artwork)
    }

    struct Track: Equatable {
        var id: String
        var title: String
        var artist: String
        var cover: Cover?
        var duration: TimeInterval
        var seed: Seed?
    }

    struct Snapshot: Equatable {
        var track: Track
        var isPlaying: Bool
        var position: TimeInterval
        var at: Date
        var canSkipPrevious = true
        var canSkipNext = true
        var canSeek = true
    }

    @Published private(set) var source: Source?
    @Published private(set) var now: Snapshot?
    /// Spotify is connected in Soundalike but neither App Remote nor the Web API can see
    /// its player (App Remote refused, and the account isn't on the Web API allow list).
    /// One tap on `wakeSpotify()` fixes it.
    @Published private(set) var spotifyNeedsWake = false

    private var spotify: Snapshot?
    private var appleMusic: Snapshot?
    private var observers: Set<AnyCancellable> = []
    private var spotifyImageID: String?
    private var appleRefresh: Task<Void, Never>?
    /// Reads Spotify through the Web API when App Remote can't connect.
    private var spotifyPoll: Task<Void, Never>?
    private var spotifyDevice: String?
    /// When each app's song or play state last changed, to pick between two paused apps.
    private var changedAt: [Source: Date] = [:]
    /// The soundalike list the playing song was started from, so Next and Previous walk it.
    private var list: FollowedList?

    private struct FollowedList {
        var matches: [Match]
        var index: Int
        var play: (Match) async throws -> Void
        /// When Soundalike last started a song from the list; the app's report lags behind.
        var startedAt = Date()
    }
    private var lastSeen: [Source: String] = [:]

    private init() {
        SpotifyRemote.shared.onState = { [weak self] state in self?.update(spotify: state) }
    }

    /// Starts following both apps; call when Soundalike comes to the foreground.
    func activate() {
        if MusicAuthorization.currentStatus == .authorized, observers.isEmpty {
            let player = SystemMusicPlayer.shared
            player.state.objectWillChange.merge(with: player.queue.objectWillChange)
                .sink { [weak self] _ in self?.scheduleAppleRefresh() }
                .store(in: &observers)
        }
        scheduleAppleRefresh()
        Task { await reconnectSpotify() }
    }

    func reconnectSpotify() async {
        let live = await SpotifyRemote.shared.startObserving()
        if live {
            spotifyPoll?.cancel()
            spotifyPoll = nil
            spotifyNeedsWake = false
            return
        }
        guard SpotifyAccount().isConnected else { return }
        // App Remote refused: read through the Web API, which works for allow-listed accounts.
        await refreshSpotifyWebAPI()
        spotifyNeedsWake = spotify == nil
        if spotifyPoll == nil {
            spotifyPoll = Task {
                while !Task.isCancelled {
                    try? await Task.sleep(for: .seconds(4))
                    await refreshSpotifyWebAPI()
                }
            }
        }
    }

    /// Opens Spotify for a moment so App Remote works; Spotify sends the user straight back.
    func wakeSpotify() {
        Task { await SpotifyConnector.shared.wakeSpotifyApp() }
    }

    func deactivate() {
        SpotifyRemote.shared.stopObserving()
        spotifyPoll?.cancel()
        spotifyPoll = nil
    }

    /// Where playback is now, moving on from the last report while playing.
    func position(at date: Date) -> TimeInterval {
        guard let now else { return 0 }
        guard now.isPlaying else { return now.position }
        return min(now.track.duration, now.position + date.timeIntervalSince(now.at))
    }

    func togglePlay() {
        guard var snapshot = now, let source else { return }
        snapshot.position = position(at: Date())
        snapshot.at = Date()
        snapshot.isPlaying.toggle()
        store(snapshot, for: source)
        switch source {
        case .spotify:
            if SpotifyRemote.shared.isConnected {
                SpotifyRemote.shared.setPlaying(snapshot.isPlaying)
            } else {
                sendWebAPI(snapshot.isPlaying ? .play : .pause)
            }
        case .appleMusic:
            Task {
                if snapshot.isPlaying { try? await SystemMusicPlayer.shared.play() } else { SystemMusicPlayer.shared.pause() }
            }
        }
    }

    /// Remembers the list a result was played from. Next and Previous then move through
    /// it for as long as that result is the song playing.
    func follow(_ matches: [Match], at index: Int, play: @escaping (Match) async throws -> Void) {
        list = FollowedList(matches: matches, index: index, play: play)
    }

    func skipToNext() {
        if step(1) { return }
        switch source {
        case .spotify: SpotifyRemote.shared.isConnected ? SpotifyRemote.shared.skipToNext() : sendWebAPI(.next)
        case .appleMusic: Task { try? await SystemMusicPlayer.shared.skipToNextEntry() }
        case nil: break
        }
    }

    /// Like Control Center: restarts the song after its first few seconds, otherwise goes back one
    /// (in the soundalike list when the song came from it).
    func skipToPrevious() {
        if position(at: Date()) > 3 {
            seek(to: 0)
            return
        }
        if step(-1) { return }
        switch source {
        case .spotify: SpotifyRemote.shared.isConnected ? SpotifyRemote.shared.skipToPrevious() : sendWebAPI(.previous)
        case .appleMusic: Task { try? await SystemMusicPlayer.shared.skipToPreviousEntry() }
        case nil: break
        }
    }

    func seek(to seconds: TimeInterval) {
        guard var snapshot = now, let source else { return }
        snapshot.position = seconds
        snapshot.at = Date()
        store(snapshot, for: source)
        switch source {
        case .spotify: SpotifyRemote.shared.isConnected ? SpotifyRemote.shared.seek(to: seconds) : sendWebAPI(.seek(seconds))
        case .appleMusic: SystemMusicPlayer.shared.playbackTime = seconds
        }
    }

    /// Plays the next (or previous) song in the followed list. False when the playing song
    /// isn't from the list or the list ends, so the music app's own skip runs instead.
    private func step(_ offset: Int) -> Bool {
        guard var list, isFromList(list) else {
            list = nil
            return false
        }
        var index = list.index + offset
        guard list.matches.indices.contains(index) else { return false }
        list.index = index
        list.startedAt = Date()
        self.list = list
        let matches = list.matches
        let play = list.play
        show(matches[index])
        Task {
            // Songs missing from the music app's catalog are skipped over.
            while matches.indices.contains(index) {
                do {
                    try await play(matches[index])
                    EventLog.record("player-list", ["index": index, "ok": true])
                    break
                } catch {
                    EventLog.record("player-list", ["index": index, "ok": false, "error": "\(error)"])
                    index += offset
                    self.list?.index = index
                }
            }
            if source == .spotify, !SpotifyRemote.shared.isConnected {
                for delay in [250, 900] {
                    try? await Task.sleep(for: .milliseconds(delay))
                    await refreshSpotifyWebAPI()
                }
            }
        }
        return true
    }

    /// Shows a list song in the bar straight away, before the music app reports it.
    private func show(_ match: Match) {
        guard let source, var snapshot = now else { return }
        snapshot.track = Track(
            id: "pending:\(match.id)", title: match.title, artist: match.artist,
            cover: match.artworkURL.map { .url($0) }, duration: 0, seed: nil
        )
        snapshot.isPlaying = true
        snapshot.position = 0
        snapshot.at = Date()
        store(snapshot, for: source)
    }

    private func isFromList(_ list: FollowedList) -> Bool {
        if Date().timeIntervalSince(list.startedAt) < 8 { return true }
        guard let track = now?.track else { return false }
        let match = list.matches[list.index]
        // Lists are full of same-title covers, so the artist has to agree too.
        return Self.similar(track.title, match.title) && Self.similar(track.artist, match.artist)
    }

    /// Equal after normalising, or one starts with the other ("Song" / "Song - Remastered",
    /// "Artist" / "Artist, Guest").
    private static func similar(_ a: String, _ b: String) -> Bool {
        let a = TextMatch.normalize(a), b = TextMatch.normalize(b)
        return !a.isEmpty && !b.isEmpty && (a == b || a.hasPrefix(b) || b.hasPrefix(a))
    }

    // MARK: - Spotify

    private func update(spotify state: SPTAppRemotePlayerState) {
        let track = state.track
        // Between songs App Remote reports a placeholder ("--", no length); skip it.
        guard track.duration > 0, !track.uri.isEmpty else { return }
        let id = track.uri
        let cover = spotify?.track.id == id ? spotify?.track.cover : nil
        spotify = Snapshot(
            track: Track(
                id: id, title: track.name, artist: track.artist.name, cover: cover,
                duration: TimeInterval(track.duration) / 1000, seed: SpotifyRemote.seed(for: track)
            ),
            isPlaying: !state.isPaused,
            position: TimeInterval(state.playbackPosition) / 1000,
            at: Date(),
            canSkipPrevious: state.playbackRestrictions.canSkipPrevious,
            canSkipNext: state.playbackRestrictions.canSkipNext,
            canSeek: state.playbackRestrictions.canSeek
        )
        choose()
        if cover == nil, spotifyImageID != id {
            spotifyImageID = id
            Task {
                guard let image = await SpotifyRemote.shared.image(for: track), spotify?.track.id == id else { return }
                spotify?.track.cover = .image(image)
                choose()
            }
        }
    }

    private func refreshSpotifyWebAPI() async {
        guard let state = try? await SpotifyAccount().playbackState() else { return }
        spotifyNeedsWake = false
        spotifyDevice = state.deviceID
        let id = state.seed.spotifyID ?? state.seed.title
        spotify = Snapshot(
            track: Track(
                id: id, title: state.seed.title, artist: state.seed.artist,
                cover: state.seed.artworkURL.map { .url($0) }, duration: state.duration, seed: state.seed
            ),
            isPlaying: state.isPlaying, position: state.position, at: Date()
        )
        choose()
    }

    /// Sends a command through the Web API, then reads the player back.
    private func sendWebAPI(_ command: SpotifyAccount.Command) {
        Task {
            do {
                try await SpotifyAccount().send(command, deviceID: spotifyDevice)
            } catch {
                EventLog.record("player-command", ["ok": false, "error": "\(error)"])
                // A song started on its own has nothing before it, and Spotify answers 403: restart it instead.
                if case .previous = command { try? await SpotifyAccount().send(.seek(0), deviceID: spotifyDevice) }
            }
            try? await Task.sleep(for: .milliseconds(500))
            await refreshSpotifyWebAPI()
        }
    }

    // MARK: - Music app

    /// MusicKit reports a change before it happens, so read the player a moment later.
    private func scheduleAppleRefresh() {
        appleRefresh?.cancel()
        appleRefresh = Task {
            try? await Task.sleep(for: .milliseconds(150))
            guard !Task.isCancelled else { return }
            refreshAppleMusic()
        }
    }

    private func refreshAppleMusic() {
        guard MusicAuthorization.currentStatus == .authorized else { return }
        let player = SystemMusicPlayer.shared
        guard let entry = player.queue.currentEntry else {
            appleMusic = nil
            choose()
            return
        }
        var title = entry.title
        var artist = entry.subtitle ?? ""
        var duration: TimeInterval = 0
        var seed: Seed?
        if case .song(let song)? = entry.item {
            title = song.title
            artist = song.artistName
            duration = song.duration ?? 0
            seed = Seed(title: song.title, artist: song.artistName, isrc: song.isrc, source: .appleMusic, appleMusicID: song.id.rawValue)
        }
        appleMusic = Snapshot(
            track: Track(
                id: entry.id, title: title, artist: artist, cover: entry.artwork.map { .musicKit($0) },
                duration: duration, seed: seed ?? Seed(title: title, artist: artist, source: .appleMusic)
            ),
            isPlaying: player.state.playbackStatus == .playing,
            position: player.playbackTime,
            at: Date(),
            canSeek: duration > 0
        )
        choose()
    }

    // MARK: -

    private func store(_ snapshot: Snapshot, for source: Source) {
        switch source {
        case .spotify: spotify = snapshot
        case .appleMusic: appleMusic = snapshot
        }
        choose()
    }

    /// Shows whichever app is playing; when neither is, the one that changed most recently.
    private func choose() {
        for candidate in [Source.spotify, .appleMusic] {
            let seen = snapshot(for: candidate).map { "\($0.track.id)|\($0.isPlaying)" }
            if seen != lastSeen[candidate] {
                lastSeen[candidate] = seen
                changedAt[candidate] = Date()
            }
        }
        let pick: Source?
        if spotify?.isPlaying == true {
            pick = .spotify
        } else if appleMusic?.isPlaying == true {
            pick = .appleMusic
        } else {
            pick = [Source.spotify, .appleMusic]
                .filter { snapshot(for: $0) != nil }
                .max { (changedAt[$0] ?? .distantPast) < (changedAt[$1] ?? .distantPast) }
        }
        let snapshot = pick.flatMap(snapshot(for:))
        if source != pick || now?.track.id != snapshot?.track.id || now?.isPlaying != snapshot?.isPlaying {
            EventLog.record("player", [
                "source": pick.map { "\($0)" } ?? "none", "playing": snapshot?.isPlaying ?? false,
                "track": snapshot.map { "\($0.track.title) — \($0.track.artist)" } ?? "",
                "duration": Int(snapshot?.track.duration ?? 0), "position": Int(snapshot?.position ?? 0),
            ])
        }
        if source != pick { source = pick }
        if now != snapshot { now = snapshot }
    }

    private func snapshot(for source: Source) -> Snapshot? {
        switch source {
        case .spotify: spotify
        case .appleMusic: appleMusic
        }
    }
}
