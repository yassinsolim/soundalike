package app.yassin.soundalike.core

import kotlin.math.roundToLong
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

internal fun parseJsonObject(text: String): JsonObject? =
    runCatching { Json.parseToJsonElement(text) as? JsonObject }.getOrNull()

internal fun JsonObject.obj(key: String): JsonObject? = this[key] as? JsonObject

internal fun JsonObject.objects(key: String): List<JsonObject> =
    (this[key] as? JsonArray)?.filterIsInstance<JsonObject>().orEmpty()

/** A non-empty string value; null for anything else. */
internal fun JsonObject.string(key: String): String? =
    (this[key] as? JsonPrimitive)?.takeIf { it.isString }?.content?.takeIf { it.isNotEmpty() }

/** A whole number given as a number or as digits in a string. */
internal fun JsonObject.long(key: String): Long? = (this[key] as? JsonPrimitive)?.asLong()

internal fun JsonElement.asLong(): Long? {
    val content = (this as? JsonPrimitive)?.content ?: return null
    return content.toLongOrNull() ?: content.toDoubleOrNull()?.takeIf { it.isFinite() }?.roundToLong()
}
