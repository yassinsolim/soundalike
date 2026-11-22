package app.yassin.soundalike

import android.annotation.SuppressLint
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat

/** Optional quiet notification shown while music plays; tapping it opens the soundalikes. */
object PlayingNotification {
    private const val CHANNEL = "playing"
    private const val ID = 1

    fun createChannel(context: Context) {
        val channel = NotificationChannel(CHANNEL, context.getString(R.string.channel_playing), NotificationManager.IMPORTANCE_LOW)
        channel.description = context.getString(R.string.channel_playing_description)
        channel.setShowBadge(false)
        context.getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
    }

    // areNotificationsEnabled() below is the permission check; lint only recognises checkSelfPermission.
    @SuppressLint("MissingPermission")
    fun update(context: Context, song: PlayingSong?) {
        val manager = NotificationManagerCompat.from(context)
        if (song == null || !song.playing || !context.app.settings.playingNotification || !manager.areNotificationsEnabled()) {
            manager.cancel(ID)
            return
        }
        val notification = NotificationCompat.Builder(context, CHANNEL)
            .setSmallIcon(R.drawable.ic_mark)
            .setContentTitle(context.getString(R.string.find_soundalikes))
            .setContentText("${song.title} · ${song.artist}")
            .setContentIntent(Surfaces.sheetPendingIntent(context, "notification"))
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setShowWhen(false)
            .setCategory(NotificationCompat.CATEGORY_RECOMMENDATION)
            .build()
        manager.notify(ID, notification)
    }
}
