import AppIntents
import SwiftUI
import WidgetKit

/// Control Center, Lock Screen and Action Button control.
@main
struct SoundalikeControls: WidgetBundle {
    var body: some Widget {
        FindSoundalikesControl()
    }
}

struct FindSoundalikesControl: ControlWidget {
    var body: some ControlWidgetConfiguration {
        StaticControlConfiguration(kind: "app.yassin.soundalike.control.open") {
            ControlWidgetButton(action: FindSoundalikesIntent()) {
                Label("Soundalike", systemImage: "waveform")
            }
        }
        .displayName("Find Soundalikes")
        .description("Opens songs that sound like what you're playing.")
    }
}
