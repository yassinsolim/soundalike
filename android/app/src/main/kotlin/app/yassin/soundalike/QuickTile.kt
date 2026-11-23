package app.yassin.soundalike

import android.annotation.SuppressLint
import android.app.StatusBarManager
import android.graphics.drawable.Icon
import android.content.ComponentName
import android.content.Context
import android.os.Build
import android.service.quicksettings.Tile
import android.service.quicksettings.TileService

/** Quick Settings tile: swipe down, tap, soundalikes for the playing song. */
class QuickTile : TileService() {
    override fun onStartListening() {
        val tile = qsTile ?: return
        val song = NowPlaying.read(this)
        tile.state = if (song != null) Tile.STATE_ACTIVE else Tile.STATE_INACTIVE
        tile.label = getString(R.string.tile_label)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) tile.subtitle = song?.title
        tile.updateTile()
    }

    @SuppressLint("StartActivityAndCollapseDeprecated")
    override fun onClick() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startActivityAndCollapse(Surfaces.sheetPendingIntent(this, "tile"))
        } else {
            @Suppress("DEPRECATION")
            startActivityAndCollapse(Surfaces.sheetIntent(this, "tile"))
        }
    }

    companion object {
        /** Android 13 and later can offer the tile in a system dialog; before that the user drags it in by hand. */
        val canAskToAdd = Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU

        fun askToAdd(context: Context) {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return
            context.getSystemService(StatusBarManager::class.java).requestAddTileService(
                ComponentName(context, QuickTile::class.java), context.getString(R.string.tile_label),
                Icon.createWithResource(context, R.drawable.ic_mark), context.mainExecutor,
            ) {}
        }

        fun requestUpdate(context: Context) {
            // Throws when the tile has not been added to the panel; there is nothing to update then.
            runCatching { requestListeningState(context, ComponentName(context, QuickTile::class.java)) }
        }
    }
}
