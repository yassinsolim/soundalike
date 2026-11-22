import AVFoundation
import Foundation
import SoundalikeKit

/// Plays 30-second previews one at a time. Streams only; nothing is saved.
///
/// Starting a preview pauses whatever else is playing (Spotify, Music). Ending or
/// stopping it releases the audio session with `notifyOthersOnDeactivation`, so the
/// other app may resume instead of staying silenced.
@MainActor
public final class PreviewPlayer: ObservableObject {
    @Published public private(set) var playingID: String?
    @Published public private(set) var loadingID: String?
    @Published public private(set) var progress: Double = 0

    private var player: AVPlayer?
    private var timeObserver: Any?
    private var endObserver: NSObjectProtocol?
    private let deezer = DeezerCatalog()

    public init() {}

    public func toggle(_ match: Match) {
        if playingID == match.id || loadingID == match.id {
            stop()
        } else {
            Task { await play(match) }
        }
    }

    public func play(_ match: Match) async {
        stop()
        loadingID = match.id
        var url = match.previewURL
        // Deezer preview URLs are signed and expire; fetch a fresh one when we can.
        if let id = match.deezerID, let fresh = await deezer.track(id)?.preview { url = fresh }
        guard loadingID == match.id, let url else {
            if loadingID == match.id { loadingID = nil }
            return
        }
        #if os(iOS)
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
        try? AVAudioSession.sharedInstance().setActive(true)
        #endif
        let item = AVPlayerItem(url: url)
        let player = AVPlayer(playerItem: item)
        self.player = player
        timeObserver = player.addPeriodicTimeObserver(forInterval: CMTime(seconds: 0.25, preferredTimescale: 600), queue: .main) { [weak self] time in
            MainActor.assumeIsolated {
                guard let self, let duration = self.player?.currentItem?.duration.seconds, duration.isFinite, duration > 0 else { return }
                self.progress = min(time.seconds / duration, 1)
                if self.loadingID == match.id, time.seconds > 0 {
                    self.loadingID = nil
                    self.playingID = match.id
                }
            }
        }
        endObserver = NotificationCenter.default.addObserver(forName: .AVPlayerItemDidPlayToEndTime, object: item, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.stop() }
        }
        player.play()
    }

    public func stop() {
        player?.pause()
        if let timeObserver { player?.removeTimeObserver(timeObserver) }
        if let endObserver { NotificationCenter.default.removeObserver(endObserver) }
        let wasActive = player != nil
        timeObserver = nil
        endObserver = nil
        player = nil
        playingID = nil
        loadingID = nil
        progress = 0
        #if os(iOS)
        if wasActive {
            try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        }
        #endif
    }
}
