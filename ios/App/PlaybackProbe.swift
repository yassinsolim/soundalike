import AVFoundation
import MusicKit

/// Works out where the music the user is hearing comes from, without asking them.
enum PlaybackProbe {
    enum Finding: String {
        /// The Music app is playing: read the song directly, no microphone needed.
        case musicApp
        /// Another app (Spotify, YouTube, …) is playing through the phone's speaker.
        case speaker
        /// Another app is playing through headphones, Bluetooth, AirPlay or a car.
        /// ShazamKit still tries (Apple shows it matching music in AirPods), but no
        /// third-party app can read another app's now-playing song, so it may miss.
        case external
        case silence
    }

    @MainActor
    static func detect() async -> Finding {
        let session = AVAudioSession.sharedInstance()
        let otherAudio = session.isOtherAudioPlaying
        if otherAudio, MusicAuthorization.currentStatus == .notDetermined {
            _ = await MusicAuthorization.request()
        }
        if MusicAuthorization.currentStatus == .authorized,
           SystemMusicPlayer.shared.state.playbackStatus == .playing,
           SystemMusicPlayer.shared.queue.currentEntry != nil {
            return .musicApp
        }
        guard otherAudio else { return .silence }
        let outputs = Set(session.currentRoute.outputs.map(\.portType))
        if outputs.isSubset(of: [.builtInSpeaker, .builtInReceiver]) { return .speaker }
        return .external
    }
}
