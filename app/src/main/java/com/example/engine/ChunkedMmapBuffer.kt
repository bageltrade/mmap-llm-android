package com.example.engine

import android.util.Log
import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * 64-bit Multi-Segment Zero-Copy Memory Mapper.
 *
 * In standard Java/Android, `FileChannel.map()` fails for files or regions > 2 GB
 * (`IllegalArgumentException: size exceeds Integer.MAX_VALUE`).
 *
 * Multi-gigabyte LLMs like Qwen 3.5 4B (3.2 GB), Llama-3 8B (4.8 GB), or 14B models
 * require multi-segment chunked mapping.
 *
 * ChunkedMmapBuffer maps the entire model file in 1 GB (1,073,741,824 bytes) segments
 * directly from flash storage. ZERO bytes of weights are copied into JVM heap.
 */
class ChunkedMmapBuffer(
    val file: File,
    val baseOffset: Long = 0L,
    val totalLength: Long = file.length() - baseOffset
) : Closeable {

    companion object {
        private const val TAG = "ChunkedMmapBuffer"
        const val CHUNK_SIZE = 1073741824L // 1 GiB per segment
    }

    private var raf: RandomAccessFile? = null
    private var channel: FileChannel? = null
    private val chunks = mutableListOf<MappedByteBuffer>()
    val numSegments: Int

    init {
        raf = RandomAccessFile(file, "r")
        val ch = raf!!.channel
        channel = ch

        val segments = if (totalLength <= 0) 0 else ((totalLength + CHUNK_SIZE - 1) / CHUNK_SIZE).toInt()
        numSegments = segments

        var mappedBytes = 0L
        for (i in 0 until segments) {
            val segmentStart = baseOffset + (i.toLong() * CHUNK_SIZE)
            val segmentLen = minOf(CHUNK_SIZE, totalLength - mappedBytes)
            try {
                val mapped = ch.map(FileChannel.MapMode.READ_ONLY, segmentStart, segmentLen)
                mapped.order(ByteOrder.LITTLE_ENDIAN)
                chunks.add(mapped)
                mappedBytes += segmentLen
            } catch (e: Exception) {
                Log.e(TAG, "Failed to map segment $i at $segmentStart (len $segmentLen)", e)
                throw e
            }
        }
        Log.i(TAG, "Successfully mapped ${file.name} in $segments segments (${totalLength / (1024 * 1024)} MB virtual)")
    }

    /**
     * Read byte at global 64-bit offset within the mapped range.
     */
    fun getByte(globalOffset: Long): Byte {
        val segmentIdx = (globalOffset / CHUNK_SIZE).toInt()
        val localOffset = (globalOffset % CHUNK_SIZE).toInt()
        return chunks[segmentIdx].get(localOffset)
    }

    /**
     * Read 16-bit short at global 64-bit offset.
     */
    fun getShort(globalOffset: Long): Short {
        val segmentIdx = (globalOffset / CHUNK_SIZE).toInt()
        val localOffset = (globalOffset % CHUNK_SIZE).toInt()
        return chunks[segmentIdx].getShort(localOffset)
    }

    /**
     * Read 32-bit int at global 64-bit offset.
     */
    fun getInt(globalOffset: Long): Int {
        val segmentIdx = (globalOffset / CHUNK_SIZE).toInt()
        val localOffset = (globalOffset % CHUNK_SIZE).toInt()
        return chunks[segmentIdx].getInt(localOffset)
    }

    /**
     * Read 32-bit float at global 64-bit offset.
     */
    fun getFloat(globalOffset: Long): Float {
        val segmentIdx = (globalOffset / CHUNK_SIZE).toInt()
        val localOffset = (globalOffset % CHUNK_SIZE).toInt()
        return chunks[segmentIdx].getFloat(localOffset)
    }

    /**
     * Copy bytes into target destination array with zero heap amplification.
     */
    fun getBytes(globalOffset: Long, dst: ByteArray, dstOffset: Int, length: Int) {
        var remaining = length
        var curGlobal = globalOffset
        var curDst = dstOffset

        while (remaining > 0) {
            val segmentIdx = (curGlobal / CHUNK_SIZE).toInt()
            val localOffset = (curGlobal % CHUNK_SIZE).toInt()
            val chunk = chunks[segmentIdx]
            val canRead = minOf(remaining.toLong(), CHUNK_SIZE - localOffset).toInt()

            val duplicate = chunk.duplicate()
            duplicate.position(localOffset)
            duplicate.get(dst, curDst, canRead)

            remaining -= canRead
            curGlobal += canRead
            curDst += canRead
        }
    }

    override fun close() {
        chunks.clear()
        try {
            channel?.close()
            raf?.close()
        } catch (e: Exception) {
            Log.w(TAG, "Error closing chunked mmap channel", e)
        }
    }
}
