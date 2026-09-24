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
 * Strictly verifies GGUF compliance and rejects any other file formats.
 * Supports any model architecture: Qwen (2.5/3.5), Llama (3/3.2), Gemma 2, Phi-3/4, DeepSeek, etc.
 */
class GgufParser {
    companion object {
        private const val TAG = "GgufParser"
        private const val DEFAULT_ALIGNMENT = 32L
    }

    /**
     * Strictly verifies whether a file is a valid GGUF file without parsing the full manifest.
     */
    fun validateGguf(file: File): Result<Boolean> {
        if (!file.name.endsWith(".gguf", ignoreCase = true)) {
            return Result.failure(
                GgufValidationException("Only .gguf files are supported. File '${file.name}' was rejected.")
            )
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

        val fileSize = file.length()
        RandomAccessFile(file, "r").use { raf ->
            val channel = raf.channel
            // Map the header portion (up to 8MB for large tensor lists like 4B/8B/14B models)
            val headerMapSize = minOf(fileSize, 8L * 1024 * 1024)
            val headerBuffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, headerMapSize)
            headerBuffer.order(ByteOrder.LITTLE_ENDIAN)

            // 1. Magic check
            val magic = headerBuffer.int
            if (magic != GgufConstants.GGUF_MAGIC) {
                throw GgufValidationException("Invalid GGUF magic: 0x${Integer.toHexString(magic)}. Non-GGUF files cannot be memory-mapped.")
            }

            // 2. Version
            val version = headerBuffer.int
            if (version !in 1..3) {
                Log.w(TAG, "Uncommon GGUF version: $version")
            }

            // 3. Tensor and Metadata Counts
            val tensorCount = headerBuffer.long
            val metadataKvCount = headerBuffer.long

            val metadataMap = mutableMapOf<String, Any>()
            for (i in 0 until metadataKvCount) {
                if (headerBuffer.remaining() < 16) break
                val key = readGgufString(headerBuffer)
                val type = headerBuffer.int
                val value = readGgufValue(headerBuffer, type)
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
                val nDims = headerBuffer.int
                val dims = LongArray(nDims)
                for (d in 0 until nDims) {
                    dims[d] = headerBuffer.long
                }
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

            val vocabSize = (metadataMap["tokenizer.ggml.tokens"] as? List<*>)?.size
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
        val len = buffer.long.toInt()
        if (len <= 0 || len > 1024 * 1024 || buffer.remaining() < len) {
            return ""
        }
        val bytes = ByteArray(len)
        buffer.get(bytes)
        return String(bytes, Charsets.UTF_8)
    }

    private fun readGgufValue(buffer: ByteBuffer, type: Int): Any? {
        return when (type) {
            GgufConstants.GGUF_TYPE_UINT8, GgufConstants.GGUF_TYPE_INT8 -> buffer.get().toInt()
            GgufConstants.GGUF_TYPE_UINT16, GgufConstants.GGUF_TYPE_INT16 -> buffer.short.toInt()
            GgufConstants.GGUF_TYPE_UINT32, GgufConstants.GGUF_TYPE_INT32 -> buffer.int
            GgufConstants.GGUF_TYPE_FLOAT32 -> buffer.float
            GgufConstants.GGUF_TYPE_BOOL -> buffer.get() != 0.toByte()
            GgufConstants.GGUF_TYPE_STRING -> readGgufString(buffer)
            GgufConstants.GGUF_TYPE_ARRAY -> {
                val itemType = buffer.int
                val count = buffer.long.toInt()
                val list = ArrayList<Any?>(minOf(count, 1000))
                for (j in 0 until minOf(count, 1000)) {
                    if (buffer.remaining() < 1) break
                    list.add(readGgufValue(buffer, itemType))
                }
                list
            }
            GgufConstants.GGUF_TYPE_UINT64, GgufConstants.GGUF_TYPE_INT64 -> buffer.long
            GgufConstants.GGUF_TYPE_FLOAT64 -> buffer.double
            else -> null
        }
    }
}
