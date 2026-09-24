package com.example

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import com.example.engine.AnyModelPresets
import com.example.engine.BundledModelProvider
import com.example.engine.DynamicContextCache
import com.example.engine.GgufParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ExampleRobolectricTest {

    @Test
    fun `read string from context`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val appName = context.getString(R.string.app_name)
        assertEquals("MmapLLM", appName)
    }

    @Test
    fun `test dynamic context caching and eviction`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val cache = DynamicContextCache(
            context = context,
            embeddingDim = 256,
            numLayers = 2,
            numKvHeads = 2,
            headDim = 32,
            maxRamTokens = 32,
            sinkTokenCount = 4
        )

        val dummyK = FloatArray(16) { 0.1f }
        val dummyV = FloatArray(16) { 0.2f }

        // Append 50 tokens (exceeding maxRamTokens=32)
        for (i in 0 until 50) {
            cache.appendTokenKv(i, dummyK, dummyV)
        }

        val stats = cache.getStats()
        assertEquals(50, stats.totalContextTokens)
        assertEquals(4, stats.attentionSinksRetained)
        assertTrue(stats.evictedTokensToDisk > 0)
        assertTrue(stats.activeTokensInRam <= 32)
        cache.close()
    }

    @Test
    fun `test bundled model creation and parsing`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val file = BundledModelProvider.getOrCreateModelFile(context)
        assertTrue(file.exists())
        assertTrue(file.length() > 1000)

        val parser = GgufParser()
        val metadata = parser.parse(file)
        assertEquals("llama", metadata.architecture)
        assertTrue(metadata.tensors.isNotEmpty())
    }

    @Test
    fun `test qwen 3_5 4b preset generation and parse`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val qwenPreset = AnyModelPresets.PRESETS.first { it.id == "qwen_3_5_4b" }
        val file = AnyModelPresets.getOrCreateModelForPreset(context, qwenPreset)
        assertTrue(file.exists())

        val parser = GgufParser()
        val meta = parser.parse(file)
        assertEquals("qwen2", meta.architecture)
        assertEquals(36, meta.blockCount)
        assertEquals(2560, meta.embeddingLength)
    }

    @Test
    fun `test bundled qwen 4b model creation and parse`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val file = BundledModelProvider.getOrCreateQwen4bModelFile(context)
        assertTrue(file.exists())
        assertTrue(file.length() > 50_000)

        val parser = GgufParser()
        val meta = parser.parse(file)
        assertEquals("qwen2", meta.architecture)
        assertEquals(BundledModelProvider.QWEN_4B_MODEL_NAME, meta.modelName)
        assertEquals(36, meta.blockCount)
        assertEquals(2560, meta.embeddingLength)
        assertEquals(32768, meta.contextLength)
    }
}
