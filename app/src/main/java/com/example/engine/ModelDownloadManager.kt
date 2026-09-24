package com.example.engine

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Protocol
import okhttp3.Request
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.util.concurrent.TimeUnit

sealed class DownloadState {
    data object Idle : DownloadState()
    data class Downloading(
        val modelId: String,
        val modelName: String,
        val bytesDownloaded: Long,
        val totalBytes: Long,
        val progressPercent: Float,
        val speedMbPerSec: Double,
        val etaSeconds: Long
    ) : DownloadState()
    data class Success(val modelName: String, val file: File) : DownloadState()
    data class Error(val modelName: String, val message: String) : DownloadState()
}

data class RemoteGgufModel(
    val id: String,
    val name: String,
    val architecture: String,
    val parameterCount: String,
    val quantType: String,
    val estimatedSizeFormatted: String,
    val downloadUrl: String,
    val fileName: String,
    val description: String,
    val isQwen: Boolean = true
)

class ModelDownloadManager(private val context: Context) {
    companion object {
        private const val TAG = "ModelDownloadManager"
        private const val MAX_RETRIES = 5

        val POPULAR_MODELS = listOf(
            RemoteGgufModel(
                id = "qwen_2_5_coder_3b",
                name = "Qwen 2.5 Coder 3B",
                architecture = "qwen2",
                parameterCount = "3.2B",
                quantType = "Q4_K_M",
                estimatedSizeFormatted = "2.1 GB",
                downloadUrl = "https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q4_k_m.gguf",
                fileName = "qwen2_5_coder_3b_q4_k_m.gguf",
                description = "High-accuracy coding & reasoning model by Qwen with 32k context and ChatML support.",
                isQwen = true
            ),
            RemoteGgufModel(
                id = "qwen_3_5_4b_instruct",
                name = "Qwen 3.5 4B Instruct",
                architecture = "qwen2",
                parameterCount = "4.02B",
                quantType = "Q4_K_M",
                estimatedSizeFormatted = "2.6 GB",
                downloadUrl = "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
                fileName = "qwen3_5_4b_q4_k_m.gguf",
                description = "State-of-the-art multilingual conversation & instruction follower with GQA attention.",
                isQwen = true
            ),
            RemoteGgufModel(
                id = "qwen_2_5_0_5b",
                name = "Qwen 2.5 0.5B Fast",
                architecture = "qwen2",
                parameterCount = "0.49B",
                quantType = "Q4_K_M",
                estimatedSizeFormatted = "390 MB",
                downloadUrl = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
                fileName = "qwen2_5_0_5b_q4_k_m.gguf",
                description = "Ultra-fast low-bandwidth Qwen model. Instantaneous download and minimal flash storage.",
                isQwen = true
            ),
            RemoteGgufModel(
                id = "deepseek_r1_1_5b_remote",
                name = "DeepSeek R1 Distill 1.5B",
                architecture = "qwen2",
                parameterCount = "1.54B",
                quantType = "Q4_K_M",
                estimatedSizeFormatted = "1.1 GB",
                downloadUrl = "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
                fileName = "deepseek_r1_distill_1_5b_q4_k_m.gguf",
                description = "Mathematical chain-of-thought reasoning distilled into Qwen architecture.",
                isQwen = true
            )
        )
    }

    // HTTP/1.1 client configured specifically to prevent HTTP/2 StreamResetException / stream CANCEL from CDNs
    private val httpClient = OkHttpClient.Builder()
        .protocols(listOf(Protocol.HTTP_1_1))
        .connectTimeout(60, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS) // No timeout during long active multi-gigabyte downloads
        .writeTimeout(60, TimeUnit.SECONDS)
        .followRedirects(true)
        .followSslRedirects(true)
        .retryOnConnectionFailure(true)
        .build()

    private val _downloadState = MutableStateFlow<DownloadState>(DownloadState.Idle)
    val downloadState: StateFlow<DownloadState> = _downloadState.asStateFlow()

    @Volatile
    private var activeCall: okhttp3.Call? = null

    @Volatile
    private var isCancelledByUser: Boolean = false

    suspend fun downloadModel(model: RemoteGgufModel, onCompleted: (File) -> Unit) = withContext(Dispatchers.IO) {
        isCancelledByUser = false
        val modelsDir = File(context.filesDir, "models")
        if (!modelsDir.exists()) modelsDir.mkdirs()

        val targetFile = File(modelsDir, model.fileName)
        if (targetFile.exists() && targetFile.length() > 50_000) {
            _downloadState.value = DownloadState.Success(model.name, targetFile)
            onCompleted(targetFile)
            return@withContext
        }

        val partFile = File(modelsDir, "${model.fileName}.part")
        var attempt = 0
        var totalBytesExpected = 0L

        while (attempt < MAX_RETRIES && !isCancelledByUser) {
            var downloadedOffset = if (partFile.exists()) partFile.length() else 0L

            try {
                val requestBuilder = Request.Builder()
                    .url(model.downloadUrl)
                    .header("User-Agent", "Mozilla/5.0 (Android; Mobile) MmapLLM/1.0")

                // Resume download if partial file exists
                if (downloadedOffset > 0L) {
                    requestBuilder.header("Range", "bytes=$downloadedOffset-")
                    Log.i(TAG, "Resuming download of ${model.name} from byte offset $downloadedOffset")
                }

                val request = requestBuilder.build()
                val call = httpClient.newCall(request)
                activeCall = call

                val response = call.execute()
                val code = response.code

                if (code != 200 && code != 206) {
                    if (code == 416) {
                        // Range Not Satisfiable: file might already be complete
                        Log.w(TAG, "HTTP 416 Range Not Satisfiable for $downloadedOffset bytes. Verifying file.")
                    } else {
                        response.close()
                        _downloadState.value = DownloadState.Error(model.name, "Server error HTTP $code: ${response.message}")
                        return@withContext
                    }
                }

                val body = response.body
                if (body == null) {
                    response.close()
                    _downloadState.value = DownloadState.Error(model.name, "Empty response body from server")
                    return@withContext
                }

                val contentLength = body.contentLength()
                if (code == 206) {
                    // Partial content: total is existing + remainder
                    totalBytesExpected = downloadedOffset + contentLength
                } else if (code == 200) {
                    // Full response: reset existing offset if server didn't honor range
                    if (downloadedOffset > 0L) {
                        downloadedOffset = 0L
                    }
                    totalBytesExpected = contentLength
                }

                val appendMode = (downloadedOffset > 0L)
                val outputStream = FileOutputStream(partFile, appendMode)
                val inputStream = body.byteStream()
                val buffer = ByteArray(64 * 1024)

                val startTime = System.currentTimeMillis()
                var lastUpdate = startTime
                var downloadedSinceStart = 0L

                inputStream.use { input ->
                    outputStream.use { output ->
                        var read: Int
                        while (input.read(buffer).also { read = it } != -1) {
                            if (isCancelledByUser) {
                                break
                            }
                            output.write(buffer, 0, read)
                            downloadedOffset += read
                            downloadedSinceStart += read

                            val now = System.currentTimeMillis()
                            if (now - lastUpdate >= 300) {
                                val elapsedSec = maxOf(0.001, (now - startTime) / 1000.0)
                                val speedBytesPerSec = downloadedSinceStart / elapsedSec
                                val speedMb = speedBytesPerSec / (1024.0 * 1024.0)
                                val progress = if (totalBytesExpected > 0) {
                                    (downloadedOffset.toFloat() / totalBytesExpected.toFloat()) * 100f
                                } else 0f
                                val remainingBytes = maxOf(0L, totalBytesExpected - downloadedOffset)
                                val eta = if (speedBytesPerSec > 0) (remainingBytes / speedBytesPerSec).toLong() else 0L

                                _downloadState.value = DownloadState.Downloading(
                                    modelId = model.id,
                                    modelName = model.name,
                                    bytesDownloaded = downloadedOffset,
                                    totalBytes = totalBytesExpected,
                                    progressPercent = progress,
                                    speedMbPerSec = speedMb,
                                    etaSeconds = eta
                                )
                                lastUpdate = now
                            }
                        }
                        output.flush()
                    }
                }

                if (isCancelledByUser) {
                    Log.i(TAG, "Download cancelled by user for ${model.name}")
                    _downloadState.value = DownloadState.Idle
                    return@withContext
                }

                // Check if download completed (or within tolerance)
                val downloadedLength = partFile.length()
                if (totalBytesExpected > 0 && downloadedLength >= totalBytesExpected - 1024) {
                    // Validate GGUF magic bytes before finalizing
                    if (isValidGgufFile(partFile)) {
                        if (partFile.renameTo(targetFile)) {
                            Log.i(TAG, "Successfully downloaded & verified GGUF: ${targetFile.absolutePath} (${targetFile.length()} bytes)")
                            _downloadState.value = DownloadState.Success(model.name, targetFile)
                            onCompleted(targetFile)
                            return@withContext
                        } else {
                            _downloadState.value = DownloadState.Error(model.name, "Failed moving downloaded file into model repository")
                            return@withContext
                        }
                    } else {
                        _downloadState.value = DownloadState.Error(model.name, "Downloaded file failed GGUF binary magic validation")
                        return@withContext
                    }
                }

            } catch (e: Exception) {
                if (isCancelledByUser) {
                    Log.i(TAG, "Download cancelled: ${e.message}")
                    _downloadState.value = DownloadState.Idle
                    return@withContext
                }

                attempt++
                Log.w(TAG, "Download attempt $attempt/$MAX_RETRIES encountered network hiccup: ${e.javaClass.simpleName} (${e.message}). Resuming in 1.5s...")

                if (attempt < MAX_RETRIES) {
                    delay(1500L)
                } else {
                    Log.e(TAG, "Exceeded maximum retry attempts for ${model.name}", e)
                    _downloadState.value = DownloadState.Error(
                        model.name,
                        "Network error: ${e.localizedMessage ?: "Stream reset by server"}. Tap to retry download."
                    )
                    return@withContext
                }
            } finally {
                activeCall = null
            }
        }
    }

    private fun isValidGgufFile(file: File): Boolean {
        if (!file.exists() || file.length() < 16) return false
        try {
            FileInputStream(file).use { stream ->
                val magic = ByteArray(4)
                val read = stream.read(magic)
                if (read == 4) {
                    val magicStr = String(magic, Charsets.US_ASCII)
                    return magicStr == "GGUF"
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error checking GGUF magic", e)
        }
        return false
    }

    fun cancelDownload() {
        isCancelledByUser = true
        activeCall?.cancel()
        activeCall = null
        _downloadState.value = DownloadState.Idle
    }

    fun dismissState() {
        _downloadState.value = DownloadState.Idle
    }
}
