package app.yassin.soundalike.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.Dp
import app.yassin.soundalike.R

/** The site's dark palette with the app icon's green-to-violet gradient as the brand. */
object Palette {
    val background = Color(0xFF0B0B0D)
    val surface = Color(0xFF141419)
    val surfaceHigh = Color(0xFF1D1D24)
    val border = Color(0xFF2A2A33)
    val text = Color(0xFFF5F5F7)
    val muted = Color(0xFFA1A1AA)
    val accent = Color(0xFF8B78F5)
    val green = Color(0xFF34D07B)
    val teal = Color(0xFF4FB8C4)
    val danger = Color(0xFFF87171)
    val brand = Brush.linearGradient(listOf(green, Color(0xFF44A7A6), Color(0xFF6D51E6)))
}

private val colors = darkColorScheme(
    primary = Palette.accent,
    onPrimary = Color.White,
    primaryContainer = Color(0xFF2E2666),
    onPrimaryContainer = Palette.text,
    secondary = Palette.green,
    onSecondary = Color(0xFF04140A),
    secondaryContainer = Color(0xFF2E2666),
    onSecondaryContainer = Palette.text,
    background = Palette.background,
    onBackground = Palette.text,
    surface = Palette.surface,
    onSurface = Palette.text,
    surfaceVariant = Palette.surfaceHigh,
    onSurfaceVariant = Palette.muted,
    surfaceContainerLowest = Palette.background,
    surfaceContainerLow = Palette.surface,
    surfaceContainer = Palette.surface,
    surfaceContainerHigh = Palette.surfaceHigh,
    surfaceContainerHighest = Palette.surfaceHigh,
    outline = Palette.border,
    outlineVariant = Palette.border,
    error = Palette.danger,
)

/** The app icon as a small round mark. */
@Composable
fun BrandMark(size: Dp) {
    Box(Modifier.size(size).background(Palette.brand, CircleShape), contentAlignment = Alignment.Center) {
        Icon(painterResource(R.drawable.ic_mark), null, Modifier.size(size * 0.62f), tint = Color(0xFF0A0C14))
    }
}

@Composable
fun SoundalikeTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = colors, content = content)
}
