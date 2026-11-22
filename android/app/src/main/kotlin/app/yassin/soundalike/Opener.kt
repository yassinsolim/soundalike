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
    suspend fun open(context: Context, match: Match, target: MusicSource, storefront: String): Boolean {
        val addresses = when (target) {
            MusicSource.SPOTIFY -> Destinations.spotify(match)
            MusicSource.APPLE_MUSIC -> listOf(Destinations.appleMusic(context.app.http, match, storefront))
            else -> listOf(Destinations.youTubeMusic(match))
        }
        val installed = target.androidPackage?.takeIf { context.app.settings.isInstalled(target) }
        if (installed != null && addresses.any { start(context, it, installed) }) return true
        return addresses.any { it.startsWith("https://") && start(context, it, null) }
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
