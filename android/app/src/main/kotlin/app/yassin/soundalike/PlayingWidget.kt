package app.yassin.soundalike

import android.appwidget.AppWidgetManager
import android.appwidget.AppWidgetProvider
import android.content.ComponentName
import android.content.Context
import android.widget.RemoteViews

/** Home-screen widget: shows the playing song; one tap opens its soundalikes. */
class PlayingWidget : AppWidgetProvider() {
    override fun onUpdate(context: Context, manager: AppWidgetManager, ids: IntArray) {
        update(context, NowPlaying.read(context))
    }

    companion object {
        fun update(context: Context, song: PlayingSong?) {
            val views = RemoteViews(context.packageName, R.layout.widget_playing)
            views.setTextViewText(
                R.id.widget_song,
                song?.let { "${it.title} · ${it.artist}" } ?: context.getString(R.string.widget_hint),
            )
            views.setOnClickPendingIntent(R.id.widget_root, Surfaces.sheetPendingIntent(context, "widget"))
            AppWidgetManager.getInstance(context).updateAppWidget(ComponentName(context, PlayingWidget::class.java), views)
        }
    }
}
