package app.yassin.soundalike

import android.app.AppOpsManager
import android.app.usage.UsageEvents
import android.app.usage.UsageStatsManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Process
import android.provider.Settings as SystemSettings

/**
 * Which app is in front, for showing the floating button only over the music app. Android tells
 * an ordinary app this only through "Usage access", a special access the user grants in system
 * settings. It is optional: without it the button shows whenever music plays.
 */
object ForegroundApp {
    private var lastSeen: String? = null
    private var queriedUntil = 0L

    fun hasAccess(context: Context): Boolean {
        val ops = context.getSystemService(AppOpsManager::class.java)
        @Suppress("DEPRECATION")
        return ops.checkOpNoThrow(AppOpsManager.OPSTR_GET_USAGE_STATS, Process.myUid(), context.packageName) == AppOpsManager.MODE_ALLOWED
    }

    fun accessSettingsIntent(context: Context): Intent =
        Intent(SystemSettings.ACTION_USAGE_ACCESS_SETTINGS, Uri.parse("package:${context.packageName}")).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)

    /** The package of the activity resumed last; null when unknown. Reads only the events since the previous call. */
    fun current(context: Context): String? {
        val now = System.currentTimeMillis()
        val from = if (queriedUntil == 0L) now - FIRST_WINDOW_MS else queriedUntil
        val events = context.getSystemService(UsageStatsManager::class.java).queryEvents(from, now) ?: return lastSeen
        val event = UsageEvents.Event()
        while (events.getNextEvent(event)) {
            if (event.eventType == UsageEvents.Event.ACTIVITY_RESUMED) lastSeen = event.packageName
        }
        queriedUntil = now
        return lastSeen
    }

    private const val FIRST_WINDOW_MS = 24 * 60 * 60 * 1000L
}
