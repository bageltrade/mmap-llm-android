package com.example.engine

import java.nio.ByteBuffer
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * High-speed, allocation-free mathematical operations for quantized LLM inference.
 * Operates directly on byte buffers without creating intermediate float arrays.
 */
object QuantizedMath {

    /**
     * Compute dot product between a Q4_0 quantized row and a float vector x.
     * Q4_0 structure: 32 elements per block.
     * Block layout: 2 bytes half-float delta (scale), followed by 16 bytes (each byte holds two 4-bit nibbles).
     * Total = 18 bytes per 32 elements.
     */
    fun dotProductQ4_0(
        weightBuffer: ByteBuffer,
        weightOffset: Int,
        x: FloatArray,
        xOffset: Int,
        numElements: Int
    ): Float {
        var sum = 0.0f
        val numBlocks = numElements / 32
        var wPos = weightOffset
        var xPos = xOffset

        for (b in 0 until numBlocks) {
            // Read 16-bit half precision float scale (delta)
            val dHalf = weightBuffer.getShort(wPos).toInt() and 0xFFFF
            val delta = halfToFloat(dHalf)
            wPos += 2

            // Process 16 bytes = 32 nibbles
            for (j in 0 until 16) {
                val byteVal = weightBuffer.get(wPos + j).toInt()
                val nibble0 = (byteVal and 0x0F) - 8
                val nibble1 = ((byteVal ushr 4) and 0x0F) - 8

                val w0 = nibble0 * delta
                val w1 = nibble1 * delta

                sum += w0 * x[xPos + j]
                sum += w1 * x[xPos + j + 16]
            }
            wPos += 16
            xPos += 32
        }
        return sum
    }

    /**
     * RMSNorm: out = x * weight / sqrt(mean(x^2) + eps)
     */
    fun rmsNorm(
        x: FloatArray,
        weight: FloatArray?,
        out: FloatArray,
        dim: Int,
        eps: Float = 1e-5f
    ) {
        var sumSq = 0.0f
        for (i in 0 until dim) {
            sumSq += x[i] * x[i]
        }
        val meanSq = sumSq / dim
        val scale = 1.0f / sqrt(meanSq + eps)

        if (weight != null) {
            for (i in 0 until dim) {
                out[i] = x[i] * scale * weight[i]
            }
        } else {
            for (i in 0 until dim) {
                out[i] = x[i] * scale
            }
        }
    }

    /**
     * Rotary Positional Embedding (RoPE) applied to query or key vectors.
     */
    fun applyRoPE(
        vec: FloatArray,
        headDim: Int,
        numHeads: Int,
        position: Int,
        freqBase: Float = 10000.0f
    ) {
        for (h in 0 until numHeads) {
            val offset = h * headDim
            for (i in 0 until headDim / 2) {
                val theta = 1.0f / Math.pow(freqBase.toDouble(), (2 * i).toDouble() / headDim).toFloat()
                val angle = position * theta
                val cosA = cos(angle)
                val sinA = sin(angle)

                val v0 = vec[offset + i]
                val v1 = vec[offset + i + headDim / 2]

                vec[offset + i] = v0 * cosA - v1 * sinA
                vec[offset + i + headDim / 2] = v0 * sinA + v1 * cosA
            }
        }
    }

    /**
     * Temperature-scaled Softmax in-place.
     */
    fun softmax(logits: FloatArray, count: Int, temperature: Float = 1.0f) {
        val temp = if (temperature <= 0.01f) 0.01f else temperature
        var maxLogit = Float.NEGATIVE_INFINITY
        for (i in 0 until count) {
            if (logits[i] > maxLogit) maxLogit = logits[i]
        }

        var sumExp = 0.0f
        for (i in 0 until count) {
            logits[i] = exp((logits[i] - maxLogit) / temp)
            sumExp += logits[i]
        }

        val invSum = 1.0f / (sumExp + 1e-12f)
        for (i in 0 until count) {
            logits[i] *= invSum
        }
    }

    /**
     * Sample next token from probability distribution with Top-P (nucleus) filtering.
     * Mirrors llama.cpp sampler order: input probs are sorted descending first
     * (llama.cpp sorts candidates before applying top_p/min_p), then nucleus
     * cutoff, then single draw from the truncated mass.
     */
    fun sampleTopP(
        probs: FloatArray,
        indices: IntArray,
        count: Int,
        topP: Float = 0.95f,
        rngFloat: Float = Math.random().toFloat()
    ): Int {
        if (count <= 0) return 0
        val n = count.coerceAtMost(minOf(probs.size, indices.size))
        // Sort candidate indices by probability descending (llama.cpp behaviour).
        val order = (0 until n).sortedByDescending { probs[it] }
        if (topP >= 1.0f) {
            // Pure random draw from full distribution.
            var r = rngFloat.coerceIn(0f, 1f)
            for (pos in order) {
                r -= probs[pos]
                if (r <= 0f) return indices[pos]
            }
            return indices[order.first()]
        }
        var cumulative = 0.0f
        for (pos in order) {
            cumulative += probs[pos]
            if (cumulative >= topP) return indices[pos]
        }
        return indices[order.first()]
    }

    /**
     * Top-K pre-filter + Top-P + Min-P chain on raw logits.
     * Matches llama.cpp defaults (top_k=40, top_p=0.95, min_p=0.05).
     * Returns the chosen index into [logits].
     */
    fun sampleLogits(
        logits: FloatArray,
        topK: Int = 40,
        topP: Float = 0.95f,
        minP: Float = 0.05f,
        temperature: Float = 0.8f,
        rngFloat: Float = Math.random().toFloat()
    ): Int {
        if (logits.isEmpty()) return 0
        val temp = if (temperature <= 0.01f) 0.01f else temperature
        val k = if (topK <= 0) logits.size else topK.coerceAtMost(logits.size)
        val topIdx = logits.indices.sortedByDescending { logits[it] }.take(k)
        var max = Float.NEGATIVE_INFINITY
        for (i in topIdx) if (logits[i] > max) max = logits[i]
        val expVals = topIdx.map { i -> i to exp((logits[i] - max) / temp) }
        val sum = expVals.sumOf { it.second.toDouble() }.toFloat().coerceAtLeast(1e-12f)
        val probsDesc = expVals.map { (i, e) -> i to e / sum }.sortedByDescending { it.second }
        // Min-P floor relative to top candidate.
        val floor = (probsDesc.firstOrNull()?.second ?: 0f) * minP
        val minFiltered = probsDesc.filter { it.second >= floor }.ifEmpty { probsDesc.take(1) }
        // Nucleus cutoff then weighted draw inside the kept set.
        var cumulative = 0f
        val nucleus = mutableListOf<Pair<Int, Float>>()
        for ((idx, p) in minFiltered) {
            nucleus.add(idx to p)
            cumulative += p
            if (cumulative >= topP) break
        }
        val kept = nucleus.ifEmpty { minFiltered.take(1) }
        val total = kept.sumOf { it.second.toDouble() }.toFloat().coerceAtLeast(1e-12f)
        var rr = rngFloat.coerceIn(0f, 1f) * total
        for ((idx, p) in kept) {
            rr -= p
            if (rr <= 0f) return idx
        }
        return kept.first().first
    }

    /** Repetition penalty in-place (llama.cpp `penalties` sampler). */
    fun applyRepeatPenaltyInPlace(
        logits: FloatArray,
        recentIds: IntArray,
        penalty: Float = 1.0f,
        lastN: Int = 64
    ) {
        if (penalty == 1.0f || recentIds.isEmpty()) return
        for (id in recentIds.takeLast(maxOf(0, lastN)).toSet()) {
            if (id in logits.indices) {
                logits[id] = if (logits[id] >= 0) logits[id] / penalty else logits[id] * penalty
            }
        }
    }

    /**
     * Converts IEEE 754 half-precision float to 32-bit single-precision float.
     */
    private fun halfToFloat(halfBits: Int): Float {
        val sign = (halfBits ushr 15) and 0x0001
        var exp = (halfBits ushr 10) and 0x001F
        var mant = halfBits and 0x03FF

        if (exp == 0) {
            if (mant == 0) {
                return Float.fromBits(sign shl 31)
            }
            while ((mant and 0x0400) == 0) {
                mant = mant shl 1
                exp--
            }
            exp++
            mant = mant and 0x03FF.inv()
        } else if (exp == 31) {
            return if (mant == 0) {
                Float.fromBits((sign shl 31) or 0x7F800000)
            } else {
                Float.NaN
            }
        }

        exp = exp + (127 - 15)
        mant = mant shl 13
        return Float.fromBits((sign shl 31) or (exp shl 23) or mant)
    }
}
