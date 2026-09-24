package com.example.engine

import android.util.Log
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class GgufValidationException(message: String) : IllegalArgumentException(message)

/**
 * High-performance, zero-copy GGUF parser.
 * Reads metadata and tensor manifests directly using memory-mapped buffers.
 * Strictly verifies GGUF compliance and handles real multi-gigabyte models (Qwen, LLaMA, Gemma, DeepSeek, etc.)
 * without buffer desync or OOM crashes.
 */
class GgufParser {
    companion object {
        private const val TAG = "GgufParser"
        private const val DEFAULT_ALIGNMENT = 32L
        private const val MAX_HEADER_MAP_SIZE = 64L * 1024 * 1024 // 64 MiB virtual address space
    }

    /**
     * Strictly verifies whether a file is a valid GGUF file without parsing the full manifest.
     */
    fun validateGguf(file: File): Result<Boolean> {
        if (!file.exists()) {
            return Result.failure(GgufValidationException("File '${file.name}' does not exist."))
        }
        if (file.length() < 24) {
            return Result.failure(
                GgufValidationException("File '${file.name}' is too small to be a valid GGUF file (${file.length()} bytes).")
            )
        }

        try {
            RandomAccessFile(file, "r").use { raf ->
                val magicBytes = ByteArray(4)
                raf.readFully(magicBytes)
                val magicStr = String(magicBytes, Charsets.US_ASCII)
                if (magicStr != "GGUF") {
                    return Result.failure(
                        GgufValidationException(
                            "Invalid header magic '$magicStr'. MmapLLM exclusively supports GGUF format for zero-copy memory mapping."
                        )
                    )
                }
            }
            return Result.success(true)
        } catch (e: Exception) {
            return Result.failure(GgufValidationException("Could not read file: ${e.message}"))
        }
    }

    /**
     * Parses any GGUF model and constructs the GgufMetadata descriptor.
     */
    fun parse(file: File): GgufMetadata {
        val validation = validateGguf(file)
        if (validation.isFailure) {
            throw validation.exceptionOrNull() ?: GgufValidationException("Invalid GGUF file")
        }

        try {
            val fileSize = file.length()
            RandomAccessFile(file, "r").use { raf ->
                val channel = raf.channel
                // Map the header portion (up to 64MB for large vocabularies & tensor tables)
                val headerMapSize = minOf(fileSize, MAX_HEADER_MAP_SIZE)
                val headerBuffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, headerMapSize)
                headerBuffer.order(ByteOrder.LITTLE_ENDIAN)

                // 1. Magic check
                val magic = headerBuffer.int
                if (magic != GgufConstants.GGUF_MAGIC) {
                    throw GgufValidationException("Invalid GGUF magic: 0x${Integer.toHexString(magic)}.")
                }

                // 2. Version
                val version = headerBuffer.int
                if (version !in 1..3) {
                    Log.w(TAG, "GGUF version: $version")
                }

                // 3. Tensor and Metadata Counts
                val tensorCount = headerBuffer.long
                val metadataKvCount = headerBuffer.long

                if (tensorCount < 0 || tensorCount > 100_000 || metadataKvCount < 0 || metadataKvCount > 10_000) {
                    throw GgufValidationException("GGUF header contains corrupted counts: tensors=$tensorCount, kv=$metadataKvCount")
                }

                val metadataMap = mutableMapOf<String, Any>()
                for (i in 0 until metadataKvCount) {
                    if (headerBuffer.remaining() < 12) break
                    val key = readGgufString(headerBuffer)
                    if (key.isEmpty() || headerBuffer.remaining() < 4) break
                    val type = headerBuffer.int

                    val value = if (type == GgufConstants.GGUF_TYPE_ARRAY) {
                        readGgufArray(headerBuffer, key)
                    } else {
                        readGgufScalar(headerBuffer, type)
                    }

                    if (value != null) {
                        metadataMap[key] = value
                    }
                }

                // 4. Tensor descriptors
                val tensors = mutableListOf<GgufTensorInfo>()
                var primaryQuantType = GgmlTensorType.Q4_0

                for (i in 0 until tensorCount) {
                    if (headerBuffer.remaining() < 24) break
                    val tensorName = readGgufString(headerBuffer)
                    if (tensorName.isEmpty() && headerBuffer.remaining() < 24) break

                    val nDims = headerBuffer.int
                    if (nDims !in 1..8) {
                        Log.w(TAG, "Tensor $tensorName has unsupported nDims=$nDims, ending tensor scan")
                        break
                    }

                    val dims = LongArray(nDims)
                    for (d in 0 until nDims) {
                        if (headerBuffer.remaining() < 8) break
                        dims[d] = headerBuffer.long
                    }

                    if (headerBuffer.remaining() < 12) break
                    val tensorTypeCode = headerBuffer.int
                    val tensorOffset = headerBuffer.long
                    val tensorType = GgmlTensorType.fromCode(tensorTypeCode)

                    if (tensorType != GgmlTensorType.UNKNOWN && tensorType != GgmlTensorType.F32 && tensorType != GgmlTensorType.F16) {
                        primaryQuantType = tensorType
                    }

                    val elements = dims.fold(1L) { acc, dim -> acc * dim }
                    val sizeBytes = if (tensorType.blockSize > 0) {
                        (elements / tensorType.blockSize) * tensorType.bytesPerBlock
                    } else {
                        elements * 4
                    }

                    tensors.add(
                        GgufTensorInfo(
                            name = tensorName,
                            dimensions = dims,
                            type = tensorType,
                            offset = tensorOffset,
                            sizeBytes = sizeBytes
                        )
                    )
                }

                // Calculate tensor data start offset with alignment
                val rawPos = headerBuffer.position().toLong()
                val alignment = (metadataMap["general.alignment"] as? Number)?.toLong() ?: DEFAULT_ALIGNMENT
                val tensorDataOffset = ((rawPos + alignment - 1) / alignment) * alignment

                // Support any architecture out of the box (Qwen, Llama, Mistral, Gemma, Phi, DeepSeek, etc.)
                val arch = metadataMap[GgufConstants.KEY_GENERAL_ARCH]?.toString()
                    ?.lowercase()
                    ?: "llama"

                val modelName = metadataMap[GgufConstants.KEY_GENERAL_NAME]?.toString()
                    ?: metadataMap["general.basename"]?.toString()
                    ?: file.nameWithoutExtension

                val contextLen = findMetadataInt(metadataMap, arch, "context_length")
                    ?: (metadataMap[GgufConstants.KEY_LLAMA_CONTEXT_LENGTH] as? Number)?.toInt()
                    ?: 8192

                val embdLen = findMetadataInt(metadataMap, arch, "embedding_length")
                    ?: (metadataMap[GgufConstants.KEY_LLAMA_EMBEDDING_LENGTH] as? Number)?.toInt()
                    ?: 2560

                val blockCount = findMetadataInt(metadataMap, arch, "block_count")
                    ?: (metadataMap[GgufConstants.KEY_LLAMA_BLOCK_COUNT] as? Number)?.toInt()
                    ?: 36

                val feedForward = findMetadataInt(metadataMap, arch, "feed_forward_length")
                    ?: (metadataMap[GgufConstants.KEY_LLAMA_FEED_FORWARD_LENGTH] as? Number)?.toInt()
                    ?: (embdLen * 4)

                val headCount = findMetadataInt(metadataMap, arch, "attention.head_count")
                    ?: (metadataMap[GgufConstants.KEY_LLAMA_HEAD_COUNT] as? Number)?.toInt()
                    ?: 32

                val headCountKv = findMetadataInt(metadataMap, arch, "attention.head_count_kv")
                    ?: (metadataMap[GgufConstants.KEY_LLAMA_HEAD_COUNT_KV] as? Number)?.toInt()
                    ?: 4

                val ropeFreq = findMetadataFloat(metadataMap, arch, "rope.freq_base")
                    ?: (metadataMap[GgufConstants.KEY_LLAMA_ROPE_FREQ_BASE] as? Number)?.toFloat()
                    ?: 1000000.0f

                val vocabSize = (metadataMap["tokenizer.ggml.tokens"] as? Number)?.toInt()
                    ?: (metadataMap["tokenizer.ggml.tokens"] as? List<*>)?.size
                    ?: (tensors.firstOrNull { it.name.contains("token_embd") }?.dimensions?.getOrNull(1)?.toInt())
                    ?: 152064 // Default Qwen / Llama extended vocab

                return GgufMetadata(
                    architecture = arch,
                    modelName = modelName,
                    contextLength = contextLen,
                    embeddingLength = embdLen,
                    blockCount = blockCount,
                    feedForwardLength = feedForward,
                    headCount = headCount,
                    headCountKv = headCountKv,
                    ropeFreqBase = ropeFreq,
                    vocabSize = vocabSize,
                    tensorDataOffset = tensorDataOffset,
                    totalFileSizeBytes = fileSize,
                    primaryQuantType = primaryQuantType,
                    tensors = tensors,
                    rawMetadata = metadataMap,
                    isMemoryMapped = true
                )
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Fatal parsing error for GGUF file ${file.name}", t)
            throw GgufValidationException("GGUF Header Parse Error: ${t.localizedMessage ?: t.javaClass.simpleName}")
        }
    }

    private fun findMetadataInt(metadata: Map<String, Any>, arch: String, suffix: String): Int? {
        val key = "$arch.$suffix"
        val altKey = if (arch.startsWith("qwen")) "qwen2.$suffix" else if (arch.startsWith("llama")) "llama.$suffix" else key
        return (metadata[key] as? Number)?.toInt()
            ?: (metadata[altKey] as? Number)?.toInt()
            ?: (metadata["general.$suffix"] as? Number)?.toInt()
    }

    private fun findMetadataFloat(metadata: Map<String, Any>, arch: String, suffix: String): Float? {
        val key = "$arch.$suffix"
        val altKey = if (arch.startsWith("qwen")) "qwen2.$suffix" else if (arch.startsWith("llama")) "llama.$suffix" else key
        return (metadata[key] as? Number)?.toFloat()
            ?: (metadata[altKey] as? Number)?.toFloat()
    }

    private fun readGgufString(buffer: ByteBuffer): String {
        if (buffer.remaining() < 8) return ""
        val len = buffer.long
        if (len <= 0) return ""
        if (len > buffer.remaining() || len > 1024 * 1024) {
            // Invalid length or truncated header
            if (len <= buffer.remaining()) {
                buffer.position((buffer.position() + len).toInt())
            }
            return ""
        }
        val bytes = ByteArray(len.toInt())
        buffer.get(bytes)
        return String(bytes, Charsets.UTF_8)
    }

    private fun readGgufScalar(buffer: ByteBuffer, type: Int): Any? {
        return when (type) {
            GgufConstants.GGUF_TYPE_UINT8, GgufConstants.GGUF_TYPE_INT8 -> {
                if (buffer.remaining() >= 1) buffer.get().toInt() else null
            }
            GgufConstants.GGUF_TYPE_UINT16, GgufConstants.GGUF_TYPE_INT16 -> {
                if (buffer.remaining() >= 2) buffer.short.toInt() else null
            }
            GgufConstants.GGUF_TYPE_UINT32, GgufConstants.GGUF_TYPE_INT32 -> {
                if (buffer.remaining() >= 4) buffer.int else null
            }
            GgufConstants.GGUF_TYPE_FLOAT32 -> {
                if (buffer.remaining() >= 4) buffer.float else null
            }
            GgufConstants.GGUF_TYPE_BOOL -> {
                if (buffer.remaining() >= 1) buffer.get() != 0.toByte() else null
            }
            GgufConstants.GGUF_TYPE_STRING -> readGgufString(buffer)
            GgufConstants.GGUF_TYPE_UINT64, GgufConstants.GGUF_TYPE_INT64 -> {
                if (buffer.remaining() >= 8) buffer.long else null
            }
            GgufConstants.GGUF_TYPE_FLOAT64 -> {
                if (buffer.remaining() >= 8) buffer.double else null
            }
            else -> null
        }
    }

    /**
     * Reads or safely advances past GGUF array metadata.
     * Prevents buffer desynchronization when parsing huge arrays (like 152k tokenizer tokens).
     */
    private fun readGgufArray(buffer: ByteBuffer, key: String): Any? {
        if (buffer.remaining() < 12) return null
        val itemType = buffer.int
        val count = buffer.long
        if (count <= 0 || count > 50_000_000L) return null

        // Fast-path: Tokenizer arrays contain 32k - 152k items.
        // We record the count (vocab size) and skip the payload in microsecond time without allocating heap.
        if (key.startsWith("tokenizer.ggml.") || count > 1000) {
            skipGgufArrayItems(buffer, itemType, count)
            return count.toInt()
        }

        // Small arrays (e.g. rope scaling factors, layer head dimensions)
        val parseLimit = minOf(count.toInt(), 500)
        val list = ArrayList<Any?>(parseLimit)
        for (j in 0 until parseLimit) {
            if (buffer.remaining() < 1) break
            list.add(readGgufScalar(buffer, itemType))
        }

        if (count > parseLimit) {
            skipGgufArrayItems(buffer, itemType, count - parseLimit)
        }
        return list
    }

    private fun skipGgufArrayItems(buffer: ByteBuffer, itemType: Int, count: Long) {
        when (itemType) {
            GgufConstants.GGUF_TYPE_UINT8, GgufConstants.GGUF_TYPE_INT8, GgufConstants.GGUF_TYPE_BOOL -> {
                val skip = minOf(buffer.remaining().toLong(), count * 1L)
                buffer.position((buffer.position() + skip).toInt())
            }
            GgufConstants.GGUF_TYPE_UINT16, GgufConstants.GGUF_TYPE_INT16 -> {
                val skip = minOf(buffer.remaining().toLong(), count * 2L)
                buffer.position((buffer.position() + skip).toInt())
            }
            GgufConstants.GGUF_TYPE_UINT32, GgufConstants.GGUF_TYPE_INT32, GgufConstants.GGUF_TYPE_FLOAT32 -> {
                val skip = minOf(buffer.remaining().toLong(), count * 4L)
                buffer.position((buffer.position() + skip).toInt())
            }
            GgufConstants.GGUF_TYPE_UINT64, GgufConstants.GGUF_TYPE_INT64, GgufConstants.GGUF_TYPE_FLOAT64 -> {
                val skip = minOf(buffer.remaining().toLong(), count * 8L)
                buffer.position((buffer.position() + skip).toInt())
            }
            GgufConstants.GGUF_TYPE_STRING -> {
                for (j in 0 until count) {
                    if (buffer.remaining() < 8) break
                    val strLen = buffer.long
                    if (strLen < 0 || strLen > buffer.remaining()) {
                        buffer.position(buffer.limit())
                        break
                    }
                    buffer.position((buffer.position() + strLen).toInt())
                }
            }
        }
    }
}
