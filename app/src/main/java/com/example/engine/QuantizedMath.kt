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
     * Sample next token from probability distribution with Top-P (nucleus) and Top-K filtering.
     */
    fun sampleTopP(
        probs: FloatArray,
        indices: IntArray,
        count: Int,
        topP: Float = 0.9f,
        rngFloat: Float = Math.random().toFloat()
    ): Int {
        // Quick sort/selection for top-p
        var cumulative = 0.0f
        for (i in 0 until count) {
            cumulative += probs[i]
            if (cumulative >= (topP * rngFloat)) {
                return indices[i]
            }
        }
        return indices[0]
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
