package app.yassin.soundalike

import android.app.PendingIntent
import android.content.Context
import android.content.Intent

/** The one-tap entry points that mirror the playing song. */
object Surfaces {
    fun update(context: Context) {
        val song = NowPlaying.read(context)
        PlayingWidget.update(context, song)
        QuickTile.requestUpdate(context)
        PlayingNotification.update(context, song)
        FloatingButton.update(context, song)
    }

    /** Opens the results sheet for the playing song. `from` names the surface in the intent, for telling them apart. */
    fun sheetIntent(context: Context, from: String): Intent =
        Intent(context, SheetActivity::class.java)
            .setAction(SheetActivity.ACTION_NOW_PLAYING)
            .putExtra(SheetActivity.EXTRA_FROM, from)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)

    fun sheetPendingIntent(context: Context, from: String): PendingIntent =
        PendingIntent.getActivity(
            context, from.hashCode(), sheetIntent(context, from),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
}
