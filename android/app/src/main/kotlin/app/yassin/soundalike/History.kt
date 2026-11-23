package app.yassin.soundalike

import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import app.yassin.soundalike.core.MusicSource
import app.yassin.soundalike.core.Seed
import app.yassin.soundalike.core.TextMatch
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/** The songs soundalikes were found for, newest first. Kept on the phone only. */
class History(context: Context) {
    private val prefs = context.getSharedPreferences("history", Context.MODE_PRIVATE)

    var items: List<Seed> by mutableStateOf(load())
        private set

    fun add(seed: Seed) {
        if (seed.title.isBlank()) return
        val key = key(seed)
        items = (listOf(seed) + items.filter { key(it) != key }).take(LIMIT)
        save()
    }

    fun clear() {
        items = emptyList()
        save()
    }

    private fun key(seed: Seed) = TextMatch.normalize(seed.title) + "|" + TextMatch.normalize(seed.artist)

    private fun save() {
        val json = buildJsonArray {
            for (seed in items) add(buildJsonObject {
                put("title", seed.title)
                put("artist", seed.artist)
                put("isrc", seed.isrc)
                put("artwork", seed.artworkUrl)
                put("source", seed.source.wire)
                put("spotify_id", seed.spotifyId)
                put("apple_id", seed.appleMusicId)
                put("storefront", seed.storefront)
            })
        }
        prefs.edit().putString("items", json.toString()).apply()
    }

    private fun load(): List<Seed> {
        val stored = prefs.getString("items", null) ?: return emptyList()
        val rows = runCatching { Json.parseToJsonElement(stored) as? JsonArray }.getOrNull() ?: return emptyList()
        return rows.filterIsInstance<JsonObject>().mapNotNull { row ->
            fun text(name: String) = (row[name] as? JsonPrimitive)?.takeIf { it.isString }?.content
            Seed(
                title = text("title") ?: return@mapNotNull null,
                artist = text("artist") ?: return@mapNotNull null,
                isrc = text("isrc"), artworkUrl = text("artwork"),
                source = MusicSource.fromWire(text("source")) ?: MusicSource.SEARCH,
                spotifyId = text("spotify_id"), appleMusicId = text("apple_id"), storefront = text("storefront"),
            )
        }
    }

    private companion object {
        const val LIMIT = 30
    }
}
