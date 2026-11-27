import AVFoundation
import Foundation
import SoundalikeKit

/// Plays Deezer's 30-second previews (or a span of one, or of a file the listener picked) one at a time.
/// Streams only; nothing is saved.
///
/// Starting pauses whatever else is playing (Spotify, Music). Ending or stopping releases the audio session
/// with `notifyOthersOnDeactivation`, so the other app may resume instead of staying silenced. A preview
/// address expires about 15 minutes after the answer that carried it; when playback fails, a new one is
/// fetched from `/api/preview` and tried once.
@MainActor
public final class PreviewPlayer: ObservableObject {
    @Published public private(set) var playingID: String?
    @Published public private(set) var loadingID: String?
    /// 0...1 through the preview, or through the span when one plays.
    @Published public private(set) var progress: Double = 0
    /// Seconds into the preview or file.
    @Published public private(set) var position: Double = 0
    /// Why the last preview couldn't play, fit to show.
    @Published public var failure: String?

    private let client: SoundalikeClient
    private var player: AVPlayer?
    private var timeObserver: Any?
    private var endObserver: NSObjectProtocol?
    private var statusObserver: NSKeyValueObservation?
    private var request: Request?

    private struct Request {
        var id: String
        var deezerID: Int?
        var span: ClosedRange<Double>?
        var refreshed: Bool
    }

    public init(client: SoundalikeClient = SoundalikeClient()) {
        self.client = client
    }

    public func isActive(_ id: String) -> Bool { playingID == id || loadingID == id }

    /// Plays or stops a result's preview.
    public func toggle(_ match: Match) {
        if isActive(match.id) {
            stop()
        } else {
            Task { await play(id: match.id, url: match.previewURL, deezerID: match.deezerID) }
        }
    }

    /// `url`: the preview (or a local file); without one it is fetched by `deezerID`.
    /// `span`: seconds to play, else all of it.
    public func play(id: String, url: URL?, deezerID: Int?, span: ClosedRange<Double>? = nil) async {
        stop()
        failure = nil
        loadingID = id
        request = Request(id: id, deezerID: deezerID, span: span, refreshed: url == nil)
        var url = url
        if url == nil, let deezerID { url = try? await client.previewURL(deezerID: deezerID) }
        guard loadingID == id else { return }
        guard let url else {
            fail(id, "There's no preview of this song.")
            return
        }
        start(url)
    }

    private func start(_ url: URL) {
        guard let request else { return }
        let id = request.id
        clearPlayer()
        #if os(iOS)
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
        try? AVAudioSession.sharedInstance().setActive(true)
        #endif
        let item = AVPlayerItem(url: url)
        if let span = request.span { item.forwardPlaybackEndTime = CMTime(seconds: span.upperBound, preferredTimescale: 600) }
        let player = AVPlayer(playerItem: item)
        self.player = player
        statusObserver = item.observe(\.status) { [weak self] item, _ in
            guard item.status == .failed else { return }
            Task { @MainActor in self?.retry(id) }
        }
        timeObserver = player.addPeriodicTimeObserver(forInterval: CMTime(seconds: 0.1, preferredTimescale: 600), queue: .main) { [weak self] time in
            MainActor.assumeIsolated { self?.tick(id, time.seconds) }
        }
        endObserver = NotificationCenter.default.addObserver(forName: .AVPlayerItemDidPlayToEndTime, object: item, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { if self?.request?.id == id { self?.stop() } }
        }
        if let span = request.span, span.lowerBound > 0 {
            player.seek(to: CMTime(seconds: span.lowerBound, preferredTimescale: 600), toleranceBefore: .zero, toleranceAfter: .zero)
        }
        player.play()
    }

    private func tick(_ id: String, _ seconds: Double) {
        guard request?.id == id, seconds.isFinite else { return }
        position = seconds
        if let span = request?.span {
            progress = min(max((seconds - span.lowerBound) / max(span.upperBound - span.lowerBound, 0.1), 0), 1)
        } else if let duration = player?.currentItem?.duration.seconds, duration.isFinite, duration > 0 {
            progress = min(seconds / duration, 1)
        }
        if loadingID == id, seconds > (request?.span?.lowerBound ?? 0) {
            loadingID = nil
            playingID = id
        }
    }

    /// The address expired (Deezer answers 403): get a new one once.
    private func retry(_ id: String) {
        guard var request, request.id == id else { return }
        guard !request.refreshed, let deezerID = request.deezerID else {
            fail(id, "The preview didn't play. Try again in a moment.")
            return
        }
        request.refreshed = true
        self.request = request
        clearPlayer()
        loadingID = id
        playingID = nil
        Task {
            let fresh = try? await client.previewURL(deezerID: deezerID)
            guard self.request?.id == id else { return }
            if let fresh { start(fresh) } else { fail(id, "The preview didn't play. Try again in a moment.") }
        }
    }

    private func fail(_ id: String, _ message: String) {
        guard request?.id == id else { return }
        stop()
        failure = message
    }

    public func stop() {
        let wasActive = player != nil
        clearPlayer()
        request = nil
        playingID = nil
        loadingID = nil
        progress = 0
        position = 0
        #if os(iOS)
        if wasActive {
            try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        }
        #endif
    }

    private func clearPlayer() {
        player?.pause()
        if let timeObserver { player?.removeTimeObserver(timeObserver) }
        if let endObserver { NotificationCenter.default.removeObserver(endObserver) }
        statusObserver?.invalidate()
        timeObserver = nil
        endObserver = nil
        statusObserver = nil
        player = nil
    }
}
