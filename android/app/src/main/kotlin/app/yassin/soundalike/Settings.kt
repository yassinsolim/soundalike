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
