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
    private var lastSeen: [Source: String] = [:]
    /// A song Soundalike just asked a music app to play, shown until that app reports it.
    private var pending: (source: Source, title: String, artist: String, until: Date)?

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

    func skipToNext() {
        switch source {
        case .spotify: SpotifyRemote.shared.isConnected ? SpotifyRemote.shared.skipToNext() : sendWebAPI(.next)
        case .appleMusic: Task { try? await SystemMusicPlayer.shared.skipToNextEntry() }
        case nil: break
        }
    }

    /// Like Control Center: restarts the song after its first few seconds, otherwise goes back one.
    /// A soundalike list started from Soundalike is what the music app is playing, so both
    /// directions walk it there too.
    func skipToPrevious() {
        if position(at: Date()) > 3 {
            seek(to: 0)
            return
        }
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

    /// Shows a tapped result in the bar at once; the music app takes a moment to report it.
    func starting(_ match: Match, in source: Source) {
        pending = (source, match.title, match.artist, Date().addingTimeInterval(4))
        let cover = (source == .spotify ? spotify : appleMusic)?.track.cover
        store(Snapshot(
            track: Track(
                id: "pending:\(match.id)", title: match.title, artist: match.artist,
                cover: match.artworkURL.map { .url($0) } ?? cover, duration: 0, seed: nil
            ),
            isPlaying: true, position: 0, at: Date()
        ), for: source)
    }

    /// After the play request: reads the app back right away instead of at the next poll,
    /// and drops the placeholder when the request failed.
    func started(in source: Source, ok: Bool) {
        if !ok { pending = nil }
        switch source {
        case .appleMusic: scheduleAppleRefresh()
        case .spotify:
            guard !SpotifyRemote.shared.isConnected else { return }
            Task {
                for delay in [150, 350, 700, 1200] {
                    try? await Task.sleep(for: .milliseconds(delay))
                    await refreshSpotifyWebAPI()
                    if pending == nil { break }
                }
            }
        }
    }

    /// False for a report of the song that was playing before the tapped one.
    private func accepts(title: String, artist: String, from source: Source) -> Bool {
        guard let pending, pending.source == source else { return true }
        guard Date() > pending.until || (TextMatch.similar(title, pending.title) && TextMatch.similar(artist, pending.artist)) else { return false }
        self.pending = nil
        return true
    }

    /// The song in the bar, for marking it in a results list.
    var live: LiveSong? {
        guard let track = now?.track else { return nil }
        let pendingID = track.id.hasPrefix("pending:") ? String(track.id.dropFirst("pending:".count)) : nil
        return LiveSong(matchID: pendingID, title: track.title, artist: track.artist, isPlaying: now?.isPlaying ?? false)
    }

    // MARK: - Spotify

    private func update(spotify state: SPTAppRemotePlayerState) {
        let track = state.track
        // Between songs App Remote reports a placeholder ("--", no length); skip it.
        guard track.duration > 0, !track.uri.isEmpty,
              accepts(title: track.name, artist: track.artist.name, from: .spotify) else { return }
        let id = track.uri
        // Keep the tapped result's cover until Spotify's own image arrives.
        let previous = spotify?.track
        let cover = previous?.id == id || previous?.id.hasPrefix("pending:") == true ? previous?.cover : nil
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
        if spotifyImageID != id {
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
        guard accepts(title: state.seed.title, artist: state.seed.artist, from: .spotify) else { return }
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
        guard accepts(title: title, artist: artist, from: .appleMusic) else { return }
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
