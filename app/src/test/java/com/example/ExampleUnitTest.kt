package com.example

import com.example.engine.AnyModelPresets
import com.example.engine.GgmlTensorType
import com.example.engine.GgufConstants
import com.example.engine.GgufParser
import com.example.engine.QuantizedMath
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class ExampleUnitTest {
    @Test
    fun testGgufConstantsAndTypes() {
        assertEquals(0x46554747, GgufConstants.GGUF_MAGIC)
        assertEquals(GgmlTensorType.Q4_0, GgmlTensorType.fromCode(2))
        assertEquals(18, GgmlTensorType.Q4_0.bytesPerBlock)
        assertEquals(32, GgmlTensorType.Q4_0.blockSize)
    }

    @Test
    fun testRejectNonGgufFiles() {
        val parser = GgufParser()
        val dummyBin = File.createTempFile("model", ".bin")
        dummyBin.writeBytes(byteArrayOf(1, 2, 3, 4, 5))
        try {
            val result = parser.validateGguf(dummyBin)
            assertFalse("Non-GGUF file (.bin) must be rejected", result.isSuccess)
        } finally {
            dummyBin.delete()
        }
    }

    @Test
    fun testArchitecturePresetsAvailability() {
        val presets = AnyModelPresets.PRESETS
        assertTrue(presets.any { it.id == "qwen_3_5_4b" })
        assertTrue(presets.any { it.id == "llama_3_2_3b" })
        assertTrue(presets.any { it.id == "deepseek_r1_1_5b" })

        val qwen = presets.first { it.id == "qwen_3_5_4b" }
        assertEquals(36, qwen.layers)
        assertEquals(32768, qwen.contextLimit)
    }

    @Test
    fun testRMSNorm() {
        val x = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
        val out = FloatArray(4)
        QuantizedMath.rmsNorm(x, null, out, 4)
        assertTrue(out[0] > 0.35f && out[0] < 0.38f)
    }

    @Test
    fun testQ4_0DotProduct() {
        val buf = ByteBuffer.allocate(18).order(ByteOrder.LITTLE_ENDIAN)
        buf.putShort(0x3C00.toShort()) // 1.0f in half float
        for (i in 0 until 16) {
            buf.put(0x98.toByte())
        }

        val x = FloatArray(32) { 1.0f }
        val result = QuantizedMath.dotProductQ4_0(buf, 0, x, 0, 32)
        assertEquals(16.0f, result, 0.05f)
    }

    @Test
    fun testProperChatEngineGeneration() {
        val chatEngine = com.example.engine.ProperChatEngine()
        val codingResponse = chatEngine.generateResponseText(
            prompt = "Write a Kotlin Flow example",
            history = emptyList(),
            modelMeta = null,
            systemPrompt = com.example.engine.ProperChatEngine.SYSTEM_PROMPT_DEFAULT
        )
        assertTrue(codingResponse.contains("Kotlin"))
        assertTrue(codingResponse.contains("Flow"))

        val qwenResponse = chatEngine.generateResponseText(
            prompt = "Explain Qwen 3.5 4B parameters and GQA",
            history = emptyList(),
            modelMeta = null,
            systemPrompt = com.example.engine.ProperChatEngine.SYSTEM_PROMPT_DEFAULT
        )
        assertTrue(qwenResponse.contains("Qwen"))
        assertTrue(qwenResponse.contains("GQA"))

        val mathResponse = chatEngine.generateResponseText(
            prompt = "Calculate tip on bill",
            history = emptyList(),
            modelMeta = null,
            systemPrompt = com.example.engine.ProperChatEngine.SYSTEM_PROMPT_DEFAULT
        )
        assertTrue(mathResponse.contains("Mathematical Reasoning"))
    }

    @Test
    fun testProperChatEngineChatMlFormat() {
        val chatEngine = com.example.engine.ProperChatEngine()
        val formatted = chatEngine.formatChatMl(
            systemPrompt = "You are a helpful assistant.",
            history = listOf(com.example.engine.ChatTurn("user", "Hi")),
            currentUserPrompt = "How are you?"
        )
        assertTrue(formatted.startsWith("<|im_start|>system"))
        assertTrue(formatted.contains("<|im_start|>user\nHi<|im_end|>"))
        assertTrue(formatted.contains("<|im_start|>user\nHow are you?<|im_end|>"))
        assertTrue(formatted.endsWith("<|im_start|>assistant\n"))
    }
}
