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

    private var spotify: Snapshot?
    private var appleMusic: Snapshot?
    private var observers: Set<AnyCancellable> = []
    private var spotifyImageID: String?
    private var appleRefresh: Task<Void, Never>?
    /// Reads Spotify through the Web API when App Remote can't connect.
    private var spotifyPoll: Task<Void, Never>?
    private var spotifyDevice: String?

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
        Task {
            let live = await SpotifyRemote.shared.startObserving()
            if !live, SpotifyAccount().isConnected, spotifyPoll == nil {
                spotifyPoll = Task {
                    while !Task.isCancelled {
                        await refreshSpotifyWebAPI()
                        try? await Task.sleep(for: .seconds(4))
                    }
                }
            }
        }
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
    func skipToPrevious() {
        switch source {
        case .spotify: SpotifyRemote.shared.isConnected ? SpotifyRemote.shared.skipToPrevious() : sendWebAPI(.previous)
        case .appleMusic:
            if position(at: Date()) > 3 {
                seek(to: 0)
            } else {
                Task { try? await SystemMusicPlayer.shared.skipToPreviousEntry() }
            }
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

    // MARK: - Spotify

    private func update(spotify state: SPTAppRemotePlayerState) {
        let track = state.track
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

    /// Shows whichever app is playing; when neither is, keeps the one shown last.
    private func choose() {
        let pick: Source?
        if spotify?.isPlaying == true {
            pick = .spotify
        } else if appleMusic?.isPlaying == true {
            pick = .appleMusic
        } else if let source, snapshot(for: source) != nil {
            pick = source
        } else {
            pick = spotify != nil ? .spotify : (appleMusic != nil ? .appleMusic : nil)
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
