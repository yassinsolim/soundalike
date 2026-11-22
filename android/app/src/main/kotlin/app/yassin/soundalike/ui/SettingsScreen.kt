package app.yassin.soundalike.ui

import android.Manifest
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.provider.Settings as SystemSettings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LifecycleEventEffect
import app.yassin.soundalike.BuildConfig
import app.yassin.soundalike.FloatingButton
import app.yassin.soundalike.ForegroundApp
import app.yassin.soundalike.QuickTile
import app.yassin.soundalike.NowPlaying
import app.yassin.soundalike.Surfaces
import app.yassin.soundalike.app
import app.yassin.soundalike.core.MusicSource

@Composable
fun SettingsScreen(modifier: Modifier, onGuide: () -> Unit) {
    val context = LocalContext.current
    val settings = context.app.settings
    var target by remember { mutableStateOf(settings.defaultTarget) }
    var sameLanguage by remember { mutableStateOf(settings.language == "same") }
    var notification by remember { mutableStateOf(settings.playingNotification) }
    var floating by remember { mutableStateOf(settings.floatingButton && FloatingButton.canDraw(context)) }
    var access by remember { mutableStateOf(NowPlaying.hasAccess(context)) }
    var usage by remember { mutableStateOf(ForegroundApp.hasAccess(context)) }
    var report by remember { mutableStateOf<String?>(null) }
    var server by remember { mutableStateOf(settings.baseUrl) }

    LifecycleEventEffect(Lifecycle.Event.ON_RESUME) {
        access = NowPlaying.hasAccess(context)
        usage = ForegroundApp.hasAccess(context)
        // Back from "Display over other apps": the switch follows what was granted.
        floating = settings.floatingButton && FloatingButton.canDraw(context)
        Surfaces.update(context)
    }
    val askToNotify = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        notification = granted
        settings.playingNotification = granted
        Surfaces.update(context)
    }

    Column(
        modifier.verticalScroll(rememberScrollState()).padding(horizontal = 16.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Card("Open songs in", "Used when a request didn't come from a music app. Shared songs open in the app they came from.") {
            for (source in MusicSource.targets) {
                Row(
                    Modifier.fillMaxWidth().clickable { target = source; settings.defaultTarget = source },
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    RadioButton(selected = target == source, onClick = null, Modifier.padding(vertical = 10.dp, horizontal = 4.dp))
                    Text(source.displayName, Modifier.weight(1f).padding(start = 8.dp))
                    if (!settings.isInstalled(source)) {
                        Text("not installed", color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
        }
        Card("Matches") {
            Toggle("Same language only", "Off: soundalikes in any language.", sameLanguage) {
                sameLanguage = it
                settings.language = if (it) "same" else "all"
            }
        }
        Card("What's playing", "Android shows the playing song to apps with notification access. Soundalike uses the title and artist only.") {
            Text(if (access) "Notification access is on." else "Notification access is off.", fontWeight = FontWeight.SemiBold)
            OutlinedButton(onClick = { context.startActivity(NowPlaying.accessSettingsIntent(context)) }) {
                Text(if (access) "Manage access" else "Allow notification access")
            }
            Text(
                "Spotify tip: turn on Device Broadcast Status in Spotify's settings and Soundalike gets the exact track, not only its name.",
                color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodyMedium,
            )
        }
        Card("One-tap shortcuts", "A Quick Settings tile and a home-screen widget are always available; see How to use Soundalike.") {
            if (QuickTile.canAskToAdd) {
                OutlinedButton(onClick = { QuickTile.askToAdd(context) }) { Text("Add the Soundalikes tile") }
            }
            Toggle("Notification while music plays", "A quiet notification; tap it for soundalikes.", notification) { on ->
                if (on && Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    askToNotify.launch(Manifest.permission.POST_NOTIFICATIONS)
                } else {
                    notification = on
                    settings.playingNotification = on
                    Surfaces.update(context)
                }
            }
            Toggle("Floating button while music plays", "One tap inside your music app. Drag it anywhere; hold it to hide it until the next song. Needs “Display over other apps”.", floating) { on ->
                settings.floatingButton = on
                if (on && !FloatingButton.canDraw(context)) {
                    context.startActivity(Intent(SystemSettings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:${context.packageName}")))
                } else {
                    floating = on
                    Surfaces.update(context)
                }
            }
        }
        if (floating) {
            Card("Floating button: where it shows") {
                if (usage) {
                    Text("Only over the app that is playing.")
                } else {
                    Text(
                        "Over every app while music plays. Android only tells an app which app is in front with “Usage access”; allow it and the button shows only over your music app. Soundalike uses it for nothing else.",
                        color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodyMedium,
                    )
                    OutlinedButton(onClick = { context.startActivity(ForegroundApp.accessSettingsIntent(context)) }) { Text("Allow usage access") }
                }
            }
        }
        Card("Help") {
            OutlinedButton(onClick = onGuide) { Text("How to use Soundalike") }
            OutlinedButton(onClick = {
                report = NowPlaying.report(context) + "\nLast answer: " + (context.app.lastAnswer ?: "none since the app started")
            }) { Text("Show what my player reports") }
            report?.let { Text(it, fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall) }
        }
        if (BuildConfig.DEBUG) {
            Card("Debug build", "Engine address. From the emulator, the PC is http://10.0.2.2:8792.") {
                OutlinedTextField(
                    value = server,
                    onValueChange = { server = it; settings.baseUrl = it },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true,
                )
            }
        }
        Text(
            "Soundalike ${BuildConfig.VERSION_NAME}",
            Modifier.padding(horizontal = 4.dp),
            color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodySmall,
        )
        Spacer(Modifier.size(24.dp))
    }
}

@Composable
private fun Card(title: String, subtitle: String? = null, content: @Composable () -> Unit) {
    Column(
        Modifier.fillMaxWidth().background(MaterialTheme.colorScheme.surface, RoundedCornerShape(18.dp)).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
        subtitle?.let { Text(it, color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodyMedium) }
        content()
    }
}

@Composable
private fun Toggle(title: String, subtitle: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth().clickable { onChange(!checked) }, verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f).padding(end = 12.dp)) {
            Text(title)
            Text(subtitle, color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodySmall)
        }
        Switch(checked = checked, onCheckedChange = null)
    }
}
