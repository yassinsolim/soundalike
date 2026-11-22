package app.yassin.soundalike.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LifecycleEventEffect
import app.yassin.soundalike.NowPlaying
import app.yassin.soundalike.R

/** First-run guide: the two quickest ways from a playing song to its soundalikes. */
@Composable
fun GuideScreen(modifier: Modifier, onDone: () -> Unit) {
    val context = LocalContext.current
    var access by remember { mutableStateOf(NowPlaying.hasAccess(context)) }
    LifecycleEventEffect(Lifecycle.Event.ON_RESUME) { access = NowPlaying.hasAccess(context) }

    Column(
        modifier.verticalScroll(rememberScrollState()).padding(horizontal = 16.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Section("Share a song", "Works straight away, for any song you can see.") {
            Step(R.drawable.ic_share, "In Spotify, YouTube Music or Apple Music, open a song's ⋮ menu and tap Share.")
            Step(R.drawable.ic_mark, "Pick Find soundalikes. In Spotify it is behind More. After a few uses Android keeps it in the first row.")
            Step(R.drawable.ic_open, "The list opens over your music app. Tap a cover to hear 30 seconds, tap a song to open it there.")
        }
        Section("One tap for what's playing", "Needs notification access: that is how Android lets an app see the playing song. Soundalike reads the title and artist, never your notifications.") {
            if (access) {
                Step(R.drawable.ic_note, "Notification access is on.")
            } else {
                Button(onClick = { context.startActivity(NowPlaying.accessSettingsIntent(context)) }) { Text("Allow notification access") }
            }
            Step(R.drawable.ic_settings, "Quick Settings: swipe down twice, tap the pencil, drag Soundalikes into your tiles. Then it is swipe down, tap.")
            Step(R.drawable.ic_mark, "Home screen: hold an empty spot, Widgets, Soundalike. Or hold the app icon and tap Find soundalikes.")
        }
        Button(onClick = onDone) { Text("Done") }
        Spacer(Modifier.size(24.dp))
    }
}

@Composable
private fun Section(title: String, subtitle: String, content: @Composable () -> Unit) {
    Column(
        Modifier.fillMaxWidth().background(MaterialTheme.colorScheme.surface, RoundedCornerShape(18.dp)).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
        Text(subtitle, color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodyMedium)
        content()
    }
}

@Composable
private fun Step(icon: Int, text: String) {
    Row {
        Icon(painterResource(icon), null, Modifier.size(22.dp), tint = MaterialTheme.colorScheme.primary)
        Spacer(Modifier.width(12.dp))
        Text(text)
    }
}
