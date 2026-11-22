package app.yassin.soundalike

import android.app.Application
import android.content.Context
import app.yassin.soundalike.core.DeezerCatalog
import app.yassin.soundalike.core.Http
import app.yassin.soundalike.core.SoundalikeClient
import app.yassin.soundalike.core.UrlConnectionHttp

class SoundalikeApp : Application() {
    val http: Http = UrlConnectionHttp("Soundalike-Android/${BuildConfig.VERSION_NAME}")
    val deezer = DeezerCatalog(http)
    lateinit var settings: Settings
        private set
    lateinit var history: History
        private set

    /** Which server route answered the last request and how long it took; shown in Settings and logged. */
    @Volatile var lastAnswer: String? = null

    /** Built per request so a changed language or server applies at once. */
    fun client() = SoundalikeClient(http, settings.baseUrl, settings.language)

    override fun onCreate() {
        super.onCreate()
        settings = Settings(this)
        history = History(this)
        PlayingNotification.createChannel(this)
        Shortcuts.publish(this)
    }
}

val Context.app: SoundalikeApp get() = applicationContext as SoundalikeApp
