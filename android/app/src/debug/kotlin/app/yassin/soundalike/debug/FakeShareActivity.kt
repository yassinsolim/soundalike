package app.yassin.soundalike.debug

import android.app.Activity
import android.content.Intent
import android.os.Bundle

/**
 * Debug builds only. Opens Android's share sheet with a song link, the way a music app's
 * Share › More does, to see where "Find soundalikes" ranks:
 *
 *     adb shell am start -n app.yassin.soundalike/.debug.FakeShareActivity \
 *         --es text "https://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b?si=x"
 */
class FakeShareActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val send = Intent(Intent.ACTION_SEND).setType("text/plain").putExtra(Intent.EXTRA_TEXT, intent.getStringExtra("text"))
        startActivity(Intent.createChooser(send, null))
        finish()
    }
}
