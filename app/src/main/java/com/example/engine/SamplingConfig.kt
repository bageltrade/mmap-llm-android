package com.example.engine

import kotlin.math.exp
import kotlin.random.Random

/**
 * Sampling hyper-parameters mirroring llama.cpp `--samplers` chain and Ollama
 * Modelfile PARAMETER / `/api/chat` options block.
 *
 * llama.cpp default chain (server README):
 *   penalties; dry; top_n_sigma; top_k; typ_p; top_p; min_p; xtc; temperature
 * defaults: temp 0.8, top_k 40, top_p 0.95, min_p 0.05, repeat_penalty 1.0.
 *
 * Ollama equivalents: temperature, top_k, top_p, min_p, repeat_penalty,
 * repeat_last_n, presence_penalty, frequency_penalty, seed, num_predict.
 */
data class SamplingConfig(
    val temperature: Float = 0.8f,
    val topK: Int = 40,
    val topP: Float = 0.95f,
    val minP: Float = 0.05f,
    val repeatPenalty: Float = 1.0f,
    val repeatLastN: Int = 64,
    val presencePenalty: Float = 0.0f,
    val frequencyPenalty: Float = 0.0f,
    val seed: Long? = null
) {
    companion object {
        val BALANCED = SamplingConfig()
        val PRECISE = SamplingConfig(temperature = 0.3f, topK = 20, topP = 0.85f, minP = 0.1f)
        val CREATIVE = SamplingConfig(temperature = 1.0f, topK = 80, topP = 0.98f, minP = 0.02f)

        fun fromLegacy(temperature: Float, topP: Float): SamplingConfig {
            // Old engine only exposed temp/topP; fill rest with llama.cpp defaults.
            return SamplingConfig(
                temperature = temperature.coerceIn(0.05f, 2.0f),
                topK = 40,
                topP = topP.coerceIn(0.05f, 1.0f),
                minP = 0.05f
            )
        }
    }
}

/**
 * Pure-Kotlin sampler implementing the llama.cpp order that matters for text quality:
 * repeat penalty -> top-K -> top-P (nucleus) -> min-P -> temperature.
 * Used both by the offline fallback composer (to vary phrasing/length) and,
 * once native logits exist, directly on model logits.
 */
object LlamaStyleSampler {

    fun applyRepeatPenalty(
        logits: FloatArray,
        recentTokenIds: IntArray,
        penalty: Float,
        lastN: Int
    ) {
        if (penalty == 1.0f || recentTokenIds.isEmpty()) return
        val window = recentTokenIds.takeLast(maxOf(0, lastN)).toSet()
        for (id in window) {
            if (id in logits.indices) {
                logits[id] = if (logits[id] >= 0) logits[id] / penalty else logits[id] * penalty
            }
        }
    }

    /** Returns sorted (prob desc) candidate ids after top-K + softmax. */
    fun topKFilter(logits: FloatArray, k: Int): List<Int> {
        if (k <= 0 || k >= logits.size) return logits.indices.toList()
        return logits.indices.sortedByDescending { logits[it] }.take(k)
    }

    fun topPFilter(probsSorted: List<Pair<Int, Float>>, topP: Float): List<Pair<Int, Float>> {
        if (topP >= 1.0f) return probsSorted
        var cum = 0f
        val out = mutableListOf<Pair<Int, Float>>()
        for (p in probsSorted) {
            out.add(p)
            cum += p.second
            if (cum >= topP) break
        }
        return out.ifEmpty { probsSorted.take(1) }
    }

    fun minPFilter(probsSorted: List<Pair<Int, Float>>, minP: Float): List<Pair<Int, Float>> {
        if (minP <= 0f || probsSorted.isEmpty()) return probsSorted
        val top = probsSorted.maxOf { it.second }
        val floor = top * minP
        return probsSorted.filter { it.second >= floor }.ifEmpty { probsSorted.take(1) }
    }

    fun sampleIndex(probs: List<Pair<Int, Float>>, rng: Random): Int {
        if (probs.isEmpty()) return 0
        if (probs.size == 1) return probs[0].first
        var sum = probs.sumOf { it.second.toDouble() }.toFloat()
        if (sum <= 0f) return probs[0].first
        var r = rng.nextFloat() * sum
        for ((id, p) in probs) {
            r -= p
            if (r <= 0) return id
        }
        return probs.last().first
    }

    /** Full chain on raw logits -> chosen token id. */
    fun sampleToken(
        logitsIn: FloatArray,
        recentTokenIds: IntArray = intArrayOf(),
        config: SamplingConfig = SamplingConfig.BALANCED,
        rng: Random = Random(config.seed ?: System.nanoTime())
    ): Int {
        val logits = logitsIn.copyOf()
        applyRepeatPenalty(logits, recentTokenIds, config.repeatPenalty, config.repeatLastN)
        val candidates = topKFilter(logits, config.topK)
        // Temperature + softmax over candidates.
        val temp = maxOf(0.01f, config.temperature)
        var max = Float.NEGATIVE_INFINITY
        for (id in candidates) if (logits[id] > max) max = logits[id]
        val probs = candidates.map { id ->
            id to exp((logits[id] - max) / temp)
        }
        val sum = probs.sumOf { it.second.toDouble() }.toFloat().coerceAtLeast(1e-12f)
        val norm = probs.map { (id, p) -> id to p / sum }.sortedByDescending { it.second }
        val nucleus = topPFilter(norm, config.topP)
        val final = minPFilter(nucleus, config.minP)
        return sampleIndex(final, rng)
    }
}
