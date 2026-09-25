package com.example.engine

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.util.concurrent.TimeUnit

/**
 * Remote-inference bridge modelled on Ollama's client/server split and
 * llama.cpp `llama-server` OpenAI-compatible API.
 *
 * Why this exists: pure-Kotlin matmul cannot produce real LLM output for a
 * multi-GB imported GGUF (llama.cpp does the forward pass in C++/ggml with
 * mmap + dequant-on-the-fly + KV cache; Ollama wraps that runner in a Go
 * server exposing `/api/chat`). Until an NDK/JNI backend lands in this app,
 * pointing the chat at a local `llama-server` or `ollama serve` endpoint is
 * the only way an imported model "chats well" with its own weights.
 *
 * Supports, in order:
 *  1. Ollama native  `/api/chat`  (POST {model, messages[], stream, options})
 *  2. OpenAI-compat `/v1/chat/completions` (llama-server, LM Studio, etc.)
 *
 * All calls are opt-in: if no server is configured/reachable the caller falls
 * back to the on-device simulation composer. No new Gradle dependencies
 * (OkHttp only, already in the app).
 */
class LlamaServerBridge(
    var baseUrl: String = "http://127.0.0.1:11434", // Ollama default; llama-server default is :8080
    var model: String = "",
    var apiKey: String? = null
) {
    companion object {
        private const val TAG = "LlamaServerBridge"
        fun normalizeBaseUrl(raw: String): String = raw.trim().trimEnd('/')
    }

    private val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS) // infinite for SSE streaming
        .writeTimeout(30, TimeUnit.SECONDS)
        .retryOnConnectionFailure(true)
        .build()

    var enabled: Boolean = false

    fun configure(baseUrl: String, model: String, apiKey: String? = null, enabled: Boolean = true) {
        this.baseUrl = normalizeBaseUrl(baseUrl)
        this.model = model
        this.apiKey = apiKey?.ifBlank { null }
        this.enabled = enabled
    }

    private fun jsonEscape(s: String): String = buildString {
        for (c in s) when (c) {
            '\\' -> append("\\\\")
            '"' -> append("\\\"")
            '\n' -> append("\\n")
            '\r' -> append("\\r")
            '\t' -> append("\\t")
            else -> if (c < ' ') append("\\u%04x".format(c.code)) else append(c)
        }
    }

    private fun messagesJson(systemPrompt: String, history: List<ChatTurn>, userPrompt: String): String {
        val sb = StringBuilder("[")
        if (systemPrompt.isNotBlank()) {
            sb.append("""{"role":"system","content":"${jsonEscape(systemPrompt)}"},""")
        }
        for (t in history) {
            val role = when (t.role.lowercase()) {
                "human" -> "user"; "ai", "model" -> "assistant"; else -> t.role.lowercase()
            }.let { if (it != "user" && it != "assistant" && it != "system") "user" else it }
            sb.append("""{"role":"$role","content":"${jsonEscape(t.content)}"},""")
        }
        sb.append("""{"role":"user","content":"${jsonEscape(userPrompt)}"}]""")
        return sb.toString()
    }

    private fun optionsJson(sampling: SamplingConfig, maxTokens: Int): String {
        // Ollama `options` + OpenAI params merged; unknown keys are ignored server-side.
        return """"temperature":${sampling.temperature},"top_p":${sampling.topP},"top_k":${sampling.topK},"min_p":${sampling.minP},"repeat_penalty":${sampling.repeatPenalty},"num_predict":$maxTokens,"stream":true"""
    }

    /** Quick probe: true if either Ollama or OpenAI-compat endpoint answers. */
    suspend fun probe(): Boolean = try {
        val req = Request.Builder().url("$baseUrl/api/tags").get().build()
        client.newCall(req).execute().use { it.isSuccessful }
    } catch (_: Exception) {
        try {
            val req = Request.Builder().url("$baseUrl/v1/models").apply {
                apiKey?.let { header("Authorization", "Bearer $it") }
            }.get().build()
            client.newCall(req).execute().use { it.isSuccessful }
        } catch (_: Exception) { false }
    }

    /**
     * Stream completion text via Ollama `/api/chat` first, then OpenAI-compat SSE.
     * Emits incremental `fullText` snapshots so the UI collector needs no changes.
     */
    fun streamChat(
        systemPrompt: String,
        history: List<ChatTurn>,
        userPrompt: String,
        sampling: SamplingConfig,
        maxTokens: Int = 1024
    ): Flow<String> = flow {
        // Attempt 1: Ollama native API (newline-delimited JSON).
        var ok = false
        try {
            emitAllOllama(systemPrompt, history, userPrompt, sampling, maxTokens) { emit(it) }
            ok = true
        } catch (e: Exception) {
            Log.w(TAG, "Ollama /api/chat failed, trying OpenAI-compat: ${e.message}")
        }
        if (!ok) {
            emitAllOpenAi(systemPrompt, history, userPrompt, sampling, maxTokens) { emit(it) }
        }
    }.flowOn(Dispatchers.IO)

    private suspend fun emitAllOllama(
        systemPrompt: String, history: List<ChatTurn>, userPrompt: String,
        sampling: SamplingConfig, maxTokens: Int, emit: suspend (String) -> Unit
    ) {
        val body = """{"model":"${jsonEscape(model)}","messages":${messagesJson(systemPrompt, history, userPrompt)},"stream":true,"options":{${optionsJson(sampling, maxTokens)}}}"""
        val req = Request.Builder().url("$baseUrl/api/chat")
            .post(body.toRequestBody("application/json".toMediaType())).build()
        client.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) throw IllegalStateException("Ollama HTTP ${resp.code}")
            val source = resp.body?.source() ?: throw IllegalStateException("empty body")
            val acc = StringBuilder()
            while (!source.exhausted()) {
                val line = source.readUtf8Line() ?: break
                if (line.isBlank()) continue
                // {"message":{"content":"..."},"done":false}
                val content = extractJsonStringField(line, "content") ?: continue
                acc.append(unescapeJson(content))
                emit(acc.toString())
                if (line.contains("\"done\":true")) break
            }
            if (acc.isEmpty()) throw IllegalStateException("no content from Ollama endpoint")
        }
    }

    private suspend fun emitAllOpenAi(
        systemPrompt: String, history: List<ChatTurn>, userPrompt: String,
        sampling: SamplingConfig, maxTokens: Int, emit: suspend (String) -> Unit
    ) {
        val body = """{"model":"${jsonEscape(model.ifBlank { "default" })}","messages":${messagesJson(systemPrompt, history, userPrompt)},"stream":true,${optionsJson(sampling, maxTokens)},"max_tokens":$maxTokens}"""
        val req = Request.Builder().url("$baseUrl/v1/chat/completions").apply {
            apiKey?.let { header("Authorization", "Bearer $it") }
        }.post(body.toRequestBody("application/json".toMediaType())).build()
        client.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) throw IllegalStateException("chat/completions HTTP ${resp.code}")
            val source = resp.body?.source() ?: throw IllegalStateException("empty body")
            val acc = StringBuilder()
            while (!source.exhausted()) {
                val line = source.readUtf8Line() ?: break
                val data = line.trim()
                if (!data.startsWith("data:")) continue
                val payload = data.removePrefix("data:").trim()
                if (payload == "[DONE]") break
                // {"choices":[{"delta":{"content":"..."}}]}
                val content = extractJsonStringField(payload, "content") ?: continue
                acc.append(unescapeJson(content))
                emit(acc.toString())
            }
            if (acc.isEmpty()) throw IllegalStateException("no content from OpenAI-compat endpoint")
        }
    }

    /** Minimal string-field extractor tolerant to nesting (finds "key":"value" with escapes). */
    internal fun extractJsonStringField(json: String, key: String): String? {
        val needle = "\"$key\""
        var idx = json.indexOf(needle)
        while (idx >= 0) {
            var p = idx + needle.length
            while (p < json.length && json[p].isWhitespace()) p++
            if (p < json.length && json[p] == ':') {
                p++
                while (p < json.length && json[p].isWhitespace()) p++
                if (p < json.length && json[p] == '"') {
                    val sb = StringBuilder()
                    p++
                    while (p < json.length) {
                        val c = json[p]
                        if (c == '\\' && p + 1 < json.length) { sb.append(c).append(json[p + 1]); p += 2; continue }
                        if (c == '"') return sb.toString()
                        sb.append(c); p++
                    }
                    return null
                }
            }
            idx = json.indexOf(needle, idx + 1)
        }
        return null
    }

    internal fun unescapeJson(s: String): String = buildString {
        var i = 0
        while (i < s.length) {
            val c = s[i]
            if (c == '\\' && i + 1 < s.length) {
                when (s[i + 1]) {
                    'n' -> append('\n'); 'r' -> append('\r'); 't' -> append('\t')
                    '"' -> append('"'); '\\' -> append('\\'); '/' -> append('/')
                    'u' -> {
                        val hex = s.substring(i + 2, minOf(i + 6, s.length))
                        append(hex.toIntOrNull(16)?.toChar() ?: '?'); i += 4
                    }
                    else -> append(s[i + 1])
                }
                i += 2
            } else { append(c); i++ }
        }
    }
}
