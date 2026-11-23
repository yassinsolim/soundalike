package app.yassin.soundalike

import android.content.Context
import androidx.core.content.pm.ShortcutInfoCompat
import androidx.core.content.pm.ShortcutManagerCompat
import androidx.core.graphics.drawable.IconCompat

/**
 * One long-lived dynamic shortcut does two jobs: long-pressing the app icon offers
 * "Find soundalikes" for the playing song, and the share sheet may rank it as a direct-share
 * target (see `share-target` in res/xml/shortcuts.xml), one tap closer than the app list.
 */
object Shortcuts {
    private const val ID = "find"
    private const val SHARE_CATEGORY = "app.yassin.soundalike.category.SHARE_TARGET"

    /** Pushing also reports a use, which is what raises the shortcut's rank in the share sheet. */
    fun publish(context: Context) {
        val shortcut = ShortcutInfoCompat.Builder(context, ID)
            .setShortLabel(context.getString(R.string.find_soundalikes))
            .setIcon(IconCompat.createWithResource(context, R.drawable.brand_circle))
            .setIntent(Surfaces.sheetIntent(context, "shortcut"))
            .setCategories(setOf(SHARE_CATEGORY))
            .setLongLived(true)
            .setRank(0)
            .build()
        runCatching { ShortcutManagerCompat.pushDynamicShortcut(context, shortcut) }
    }
}
