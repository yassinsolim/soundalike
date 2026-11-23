package app.yassin.soundalike.core

import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLDecoder
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope

class HttpResponse(
    val status: Int,
    val body: String,
    /** The address that answered, after redirects. */
    val finalUrl: String,
    /** `Retry-After` in seconds, when the server sent one. */
    val retryAfterSeconds: Int? = null,
)

/** Every request has a timeout; transport failures surface as [SoundalikeError.Offline]. */
fun interface Http {
    suspend fun get(url: String, timeoutMs: Int, headers: Map<String, String>): HttpResponse
}

suspend fun Http.get(url: String, timeoutMs: Int = 15_000): HttpResponse = get(url, timeoutMs, emptyMap())

class UrlConnectionHttp(private val userAgent: String = "Soundalike-Android") : Http {
    override suspend fun get(url: String, timeoutMs: Int, headers: Map<String, String>): HttpResponse = coroutineScope {
        val connection = try {
            URL(url).openConnection() as HttpURLConnection
        } catch (error: IOException) {
            throw SoundalikeError.Offline()
        }
        val request = async(Dispatchers.IO) {
            try {
                connection.connectTimeout = minOf(timeoutMs, 10_000)
                connection.readTimeout = timeoutMs
                connection.setRequestProperty("User-Agent", userAgent)
                headers.forEach { (name, value) -> connection.setRequestProperty(name, value) }
                val status = connection.responseCode
                val stream = if (status >= 400) connection.errorStream else connection.inputStream
                HttpResponse(
                    status = status,
                    body = stream?.use { it.readBytes().toString(Charsets.UTF_8) }.orEmpty(),
                    finalUrl = connection.url.toString(),
                    retryAfterSeconds = connection.getHeaderField("Retry-After")?.trim()?.toIntOrNull(),
                )
            } catch (error: IOException) {
                throw SoundalikeError.Offline()
            } finally {
                connection.disconnect()
            }
        }
        try {
            request.await()
        } catch (cancelled: CancellationException) {
            // Unblocks the socket read so the IO thread is released at once.
            connection.disconnect()
            throw cancelled
        }
    }
}

private const val UNRESERVED = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"

/** Percent-encodes everything but RFC 3986 unreserved characters, so a space is `%20` and `+` is `%2B`. */
fun percentEncode(text: String): String = buildString {
    for (byte in text.toByteArray(Charsets.UTF_8)) {
        val char = byte.toInt().toChar()
        if (byte >= 0 && char in UNRESERVED) append(char) else append('%').append("%02X".format(byte.toInt() and 0xFF))
    }
}

fun percentDecode(text: String): String =
    runCatching { URLDecoder.decode(text.replace("+", "%2B"), "UTF-8") }.getOrDefault(text)

/** Appends the non-empty values as a query string, in name order. */
fun urlWithQuery(base: String, query: Map<String, String?>): String {
    val items = query.entries
        .filter { !it.value.isNullOrEmpty() }
        .sortedBy { it.key }
        .joinToString("&") { "${percentEncode(it.key)}=${percentEncode(it.value!!)}" }
    return if (items.isEmpty()) base else base + (if ('?' in base) "&" else "?") + items
}
