import AppIntents
import SwiftUI
import WidgetKit

/// Control Center, Lock Screen and Action Button controls.
@main
struct SoundalikeControls: WidgetBundle {
    var body: some Widget {
        RecogniseControl()
        NowPlayingControl()
    }
}

struct RecogniseControl: ControlWidget {
    var body: some ControlWidgetConfiguration {
        StaticControlConfiguration(kind: "app.yassin.soundalike.control.recognise") {
            ControlWidgetButton(action: ListenAndFindIntent()) {
                Label("Recognise Song", systemImage: "waveform")
            }
        }
        .displayName("Recognise Song")
        .description("Recognise the song playing around you and find soundalikes.")
    }
}

struct NowPlayingControl: ControlWidget {
    var body: some ControlWidgetConfiguration {
        StaticControlConfiguration(kind: "app.yassin.soundalike.control.nowplaying") {
            ControlWidgetButton(action: FindForCurrentSongIntent()) {
                Label("Soundalikes for Now Playing", systemImage: "music.note")
            }
        }
        .displayName("Soundalikes for Now Playing")
        .description("Find songs that sound like what's playing in the Music app.")
    }
}
