import AVFoundation
import MusicKit

/// Works out where the music the user is hearing comes from, without asking them.
enum PlaybackProbe {
    enum Finding: String {
        /// The Music app is playing: read the song directly, no microphone needed.
        case musicApp
        /// Another app (Spotify, YouTube, …) is playing through the phone's speaker.
        case speaker
        /// Another app is playing through Bluetooth, AirPlay or a car. The microphone
        /// hears it only when that is a speaker in the room, not headphones.
        case external
        /// Another app is playing into wired headphones. No third-party app can hear or read it.
        case headphones
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
        if outputs.contains(.headphones) { return .headphones }
        if outputs.isSubset(of: [.builtInSpeaker, .builtInReceiver]) { return .speaker }
        return .external
    }
}
