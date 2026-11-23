package app.yassin.soundalike

import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.net.Uri
import app.yassin.soundalike.core.Destinations
import app.yassin.soundalike.core.Match
import app.yassin.soundalike.core.MusicSource

object Opener {
    /**
     * Opens the song in the target music app; when that app isn't installed, the same
     * address goes to whatever handles it (the browser's web player).
     */
    enum class Outcome { IN_APP, IN_BROWSER, APP_MISSING, FAILED }

    suspend fun open(context: Context, match: Match, target: MusicSource, storefront: String): Outcome {
        val addresses = when (target) {
            MusicSource.SPOTIFY -> Destinations.spotify(match)
            MusicSource.APPLE_MUSIC -> listOf(Destinations.appleMusic(context.app.http, match, storefront))
            else -> listOf(Destinations.youTubeMusic(match))
        }
        val installed = target.androidPackage?.takeIf { context.app.settings.isInstalled(target) }
        if (installed != null && addresses.any { start(context, it, installed) }) return Outcome.IN_APP
        // Spotify's web player drops a search link on phones (it lands on an empty search page),
        // so without the app only a song's own page is worth opening in the browser.
        val web = addresses.filter { it.startsWith("https://") && !(target == MusicSource.SPOTIFY && "/search/" in it) }
        if (web.isEmpty() && installed == null) return Outcome.APP_MISSING
        return if (web.any { start(context, it, null) }) Outcome.IN_BROWSER else Outcome.FAILED
    }

    /** The app's Play Store page, in the Play Store app when there is one. */
    fun openStore(context: Context, target: MusicSource) {
        val name = target.androidPackage ?: return
        if (!start(context, "market://details?id=$name", null)) start(context, "https://play.google.com/store/apps/details?id=$name", null)
    }

    private fun start(context: Context, address: String, packageName: String?): Boolean = try {
        context.startActivity(
            Intent(Intent.ACTION_VIEW, Uri.parse(address)).setPackage(packageName).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        )
        true
    } catch (missing: ActivityNotFoundException) {
        false
    }
}
