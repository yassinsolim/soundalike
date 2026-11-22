package app.yassin.soundalike

import android.content.Context
import android.content.pm.PackageManager
import app.yassin.soundalike.core.MusicSource
import app.yassin.soundalike.core.SoundalikeClient

class Settings(private val context: Context) {
    private val prefs = context.getSharedPreferences("settings", Context.MODE_PRIVATE)

    /** The app results open in when the request didn't come from one. Defaults to the first one installed. */
    var defaultTarget: MusicSource
        get() = MusicSource.fromWire(prefs.getString("target", null))?.takeIf { it in MusicSource.targets }
            ?: MusicSource.targets.firstOrNull(::isInstalled)
            ?: MusicSource.SPOTIFY
        set(value) = prefs.edit().putString("target", value.wire).apply()

    /** `same`: only songs in the seed's language. `all`: any language. */
    var language: String
        get() = prefs.getString("language", "same") ?: "same"
        set(value) = prefs.edit().putString("language", value).apply()

    var guideSeen: Boolean
        get() = prefs.getBoolean("guide_seen", false)
        set(value) = prefs.edit().putBoolean("guide_seen", value).apply()

    var playingNotification: Boolean
        get() = prefs.getBoolean("playing_notification", false)
        set(value) = prefs.edit().putBoolean("playing_notification", value).apply()

    var floatingButton: Boolean
        get() = prefs.getBoolean("floating_button", false)
        set(value) = prefs.edit().putBoolean("floating_button", value).apply()

    /** Where the user dragged the floating button: offset from the right edge and from the vertical centre, in pixels. */
    var floatingPosition: Pair<Int, Int>?
        get() = if (prefs.contains("floating_x")) prefs.getInt("floating_x", 0) to prefs.getInt("floating_y", 0) else null
        set(value) = prefs.edit().putInt("floating_x", value?.first ?: 0).putInt("floating_y", value?.second ?: 0).apply()

    /** Debug builds can point at another engine, for example http://10.0.2.2:8792 from the emulator. */
    var baseUrl: String
        get() = prefs.getString("base_url", null)?.takeIf { BuildConfig.DEBUG && it.isNotBlank() } ?: SoundalikeClient.DEFAULT_BASE
        set(value) = prefs.edit().putString("base_url", value.trim().trimEnd('/')).apply()

    fun isInstalled(source: MusicSource): Boolean {
        val name = source.androidPackage ?: return false
        return try {
            context.packageManager.getPackageInfo(name, 0)
            true
        } catch (missing: PackageManager.NameNotFoundException) {
            false
        }
    }
}
