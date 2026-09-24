package com.example.engine

/**
 * GGUF (GGML Universal File) specifications and magic numbers.
 * GGUF is designed for zero-copy memory mapping.
 */
object GgufConstants {
    const val GGUF_MAGIC = 0x46554747 // "GGUF" in Little Endian (0x47, 0x47, 0x55, 0x46)
    const val GGUF_MAGIC_LE_BYTES = "GGUF"

    const val GGUF_VERSION_V1 = 1
    const val GGUF_VERSION_V2 = 2
    const val GGUF_VERSION_V3 = 3

    // GGUF Metadata Value Types
    const val GGUF_TYPE_UINT8 = 0
    const val GGUF_TYPE_INT8 = 1
    const val GGUF_TYPE_UINT16 = 2
    const val GGUF_TYPE_INT16 = 3
    const val GGUF_TYPE_UINT32 = 4
    const val GGUF_TYPE_INT32 = 5
    const val GGUF_TYPE_FLOAT32 = 6
    const val GGUF_TYPE_BOOL = 7
    const val GGUF_TYPE_STRING = 8
    const val GGUF_TYPE_ARRAY = 9
    const val GGUF_TYPE_UINT64 = 10
    const val GGUF_TYPE_INT64 = 11
    const val GGUF_TYPE_FLOAT64 = 12

    // GGML Quantization / Tensor Types
    const val GGML_TYPE_F32 = 0
    const val GGML_TYPE_F16 = 1
    const val GGML_TYPE_Q4_0 = 2
    const val GGML_TYPE_Q4_1 = 3
    const val GGML_TYPE_Q5_0 = 6
    const val GGML_TYPE_Q5_1 = 7
    const val GGML_TYPE_Q8_0 = 8
    const val GGML_TYPE_Q8_1 = 9
    const val GGML_TYPE_Q2_K = 10
    const val GGML_TYPE_Q3_K = 11
    const val GGML_TYPE_Q4_K = 12
    const val GGML_TYPE_Q5_K = 13
    const val GGML_TYPE_Q6_K = 14
    const val GGML_TYPE_Q8_K = 15
    const val GGML_TYPE_IQ2_XXS = 16
    const val GGML_TYPE_IQ4_NL = 20

    // Common GGUF Metadata keys
    const val KEY_GENERAL_ARCH = "general.architecture"
    const val KEY_GENERAL_NAME = "general.name"
    const val KEY_GENERAL_QUANT_VERSION = "general.quantization_version"
    const val KEY_GENERAL_FILE_TYPE = "general.file_type"
    const val KEY_LLAMA_CONTEXT_LENGTH = "llama.context_length"
    const val KEY_LLAMA_EMBEDDING_LENGTH = "llama.embedding_length"
    const val KEY_LLAMA_BLOCK_COUNT = "llama.block_count"
    const val KEY_LLAMA_FEED_FORWARD_LENGTH = "llama.feed_forward_length"
    const val KEY_LLAMA_HEAD_COUNT = "llama.attention.head_count"
    const val KEY_LLAMA_HEAD_COUNT_KV = "llama.attention.head_count_kv"
    const val KEY_LLAMA_ROPE_FREQ_BASE = "llama.rope.freq_base"
    const val KEY_TOKENIZER_MODEL = "tokenizer.ggml.model"
    const val KEY_TOKENIZER_TOKENS = "tokenizer.ggml.tokens"
    const val KEY_TOKENIZER_BOS = "tokenizer.ggml.bos_token_id"
    const val KEY_TOKENIZER_EOS = "tokenizer.ggml.eos_token_id"
}

enum class GgmlTensorType(val code: Int, val description: String, val bytesPerBlock: Int, val blockSize: Int) {
    F32(0, "Float32 (32-bit)", 4, 1),
    F16(1, "Float16 (16-bit)", 2, 1),
    Q4_0(2, "Q4_0 (4-bit)", 18, 32),
    Q4_1(3, "Q4_1 (4-bit)", 20, 32),
    Q5_0(6, "Q5_0 (5-bit)", 22, 32),
    Q5_1(7, "Q5_1 (5-bit)", 24, 32),
    Q8_0(8, "Q8_0 (8-bit)", 34, 32),
    Q8_1(9, "Q8_1 (8-bit)", 36, 32),
    Q4_K(12, "Q4_K (k-quant)", 144, 256),
    Q5_K(13, "Q5_K (k-quant)", 176, 256),
    Q6_K(14, "Q6_K (k-quant)", 210, 256),
    Q8_K(15, "Q8_K (k-quant)", 292, 256),
    UNKNOWN(-1, "Custom / Unknown", 1, 1);

    companion object {
        fun fromCode(code: Int): GgmlTensorType {
            return entries.firstOrNull { it.code == code } ?: UNKNOWN
        }
    }
}
