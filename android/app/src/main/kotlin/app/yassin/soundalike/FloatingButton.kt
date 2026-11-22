package app.yassin.soundalike

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.PixelFormat
import android.provider.Settings as SystemSettings
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewConfiguration
import android.view.WindowManager
import android.widget.ImageView
import kotlin.math.abs

/**
 * Experimental: a small draggable button drawn over other apps while music plays.
 * Needs "Display over other apps". Android can't tell us which app is in front without
 * accessibility or usage access, so it shows over every app, not only the music app.
 */
object FloatingButton {
    private var view: View? = null

    fun canDraw(context: Context): Boolean = SystemSettings.canDrawOverlays(context)

    fun update(context: Context, song: PlayingSong?) {
        val show = song?.playing == true && context.app.settings.floatingButton && canDraw(context)
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
        val size = (52 * context.resources.displayMetrics.density).toInt()
        val params = WindowManager.LayoutParams(
            size, size,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
            PixelFormat.TRANSLUCENT,
        ).apply {
            gravity = Gravity.END or Gravity.CENTER_VERTICAL
            x = size / 4
        }
        val button = ImageView(context).apply {
            setImageResource(R.drawable.brand_circle)
            contentDescription = context.getString(R.string.find_soundalikes)
            elevation = 8 * context.resources.displayMetrics.density
        }
        val slop = ViewConfiguration.get(context).scaledTouchSlop
        var downX = 0f
        var downY = 0f
        var startX = 0
        var startY = 0
        var dragged = false
        button.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    downX = event.rawX; downY = event.rawY
                    startX = params.x; startY = params.y
                    dragged = false
                }
                MotionEvent.ACTION_MOVE -> {
                    val dx = event.rawX - downX
                    val dy = event.rawY - downY
                    if (dragged || abs(dx) > slop || abs(dy) > slop) {
                        dragged = true
                        params.x = startX - dx.toInt() // gravity END: x grows leftwards
                        params.y = startY + dy.toInt()
                        windows.updateViewLayout(button, params)
                    }
                }
                MotionEvent.ACTION_UP -> if (!dragged) context.startActivity(Surfaces.sheetIntent(context, "floating"))
            }
            true
        }
        windows.addView(button, params)
        view = button
    }
}
