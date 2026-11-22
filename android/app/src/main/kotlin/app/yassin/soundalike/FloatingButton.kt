package app.yassin.soundalike

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.PixelFormat
import android.os.Handler
import android.os.Looper
import android.provider.Settings as SystemSettings
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewConfiguration
import android.view.WindowManager
import android.widget.ImageView
import android.widget.Toast
import kotlin.math.abs

/**
 * Optional: a small draggable button drawn over the music app while it plays. One tap opens the
 * soundalikes for the playing song without leaving the app; a long press hides it until the next song.
 *
 * Needs "Display over other apps". With "Usage access" as well it shows only while the app that
 * is playing is in front; without it Android can't say which app that is, so it shows over every app.
 */
object FloatingButton {
    private const val SIZE_DP = 44
    private const val POLL_MS = 1000L

    private val main = Handler(Looper.getMainLooper())
    private var view: View? = null
    private var song: PlayingSong? = null
    private var hiddenFor: String? = null
    private var appContext: Context? = null

    /** How many Soundalike screens are showing. Over its own screens the button would only be in the way. */
    var ownScreens = 0

    /** The app in front can change without the player saying anything, so it is looked up again while the button may show. */
    private val poll = Runnable { appContext?.let { update(it, song) } }

    fun canDraw(context: Context): Boolean = SystemSettings.canDrawOverlays(context)

    fun update(context: Context, song: PlayingSong?) {
        this.song = song
        appContext = context.applicationContext
        main.removeCallbacks(poll)
        val wanted = song != null && song.playing && context.app.settings.floatingButton && canDraw(context)
        val overMusicOnly = wanted && ForegroundApp.hasAccess(context)
        if (overMusicOnly) main.postDelayed(poll, POLL_MS)
        val show = wanted && ownScreens == 0 && hiddenFor != song?.key &&
            (!overMusicOnly || ForegroundApp.current(context) == song?.packageName)
        if (show && view == null) show(context.applicationContext)
        if (!show) hide(context)
    }

    private fun hide(context: Context) {
        view?.let { context.getSystemService(WindowManager::class.java).removeView(it) }
        view = null
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun show(context: Context) {
        val windows = context.getSystemService(WindowManager::class.java)
        val density = context.resources.displayMetrics.density
        val size = (SIZE_DP * density).toInt()
        val saved = context.app.settings.floatingPosition
        val params = WindowManager.LayoutParams(
            size, size,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
            PixelFormat.TRANSLUCENT,
        ).apply {
            gravity = Gravity.END or Gravity.CENTER_VERTICAL
            x = saved?.first ?: (8 * density).toInt()
            y = saved?.second ?: 0
        }
        val button = ImageView(context).apply {
            setImageResource(R.drawable.brand_circle)
            contentDescription = context.getString(R.string.find_soundalikes)
            alpha = 0.92f
        }
        val slop = ViewConfiguration.get(context).scaledTouchSlop
        var downX = 0f
        var downY = 0f
        var startX = 0
        var startY = 0
        var dragged = false
        val longPress = Runnable {
            if (!dragged) {
                dragged = true // the finger coming up must not count as a tap
                hiddenFor = song?.key
                hide(context)
                Toast.makeText(context, "Hidden until the next song.", Toast.LENGTH_SHORT).show()
            }
        }
        button.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    downX = event.rawX; downY = event.rawY
                    startX = params.x; startY = params.y
                    dragged = false
                    main.postDelayed(longPress, ViewConfiguration.getLongPressTimeout().toLong())
                }
                MotionEvent.ACTION_MOVE -> {
                    val dx = event.rawX - downX
                    val dy = event.rawY - downY
                    if (dragged || abs(dx) > slop || abs(dy) > slop) {
                        if (view == null) return@setOnTouchListener true // hidden by the long press
                        dragged = true
                        main.removeCallbacks(longPress)
                        params.x = startX - dx.toInt() // gravity END: x grows leftwards
                        params.y = startY + dy.toInt()
                        windows.updateViewLayout(button, params)
                    }
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    main.removeCallbacks(longPress)
                    if (dragged) {
                        if (view != null) context.app.settings.floatingPosition = params.x to params.y
                    } else if (event.action == MotionEvent.ACTION_UP) {
                        context.startActivity(Surfaces.sheetIntent(context, "floating"))
                    }
                }
            }
            true
        }
        windows.addView(button, params)
        view = button
    }
}
