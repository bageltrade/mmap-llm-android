package com.example.engine

data class GgufTensorInfo(
    val name: String,
    val dimensions: LongArray,
    val type: GgmlTensorType,
    val offset: Long, // Offset relative to tensor data base in file
    val sizeBytes: Long
) {
    val elementCount: Long
        get() = dimensions.fold(1L) { acc, dim -> acc * dim }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as GgufTensorInfo
        return name == other.name && offset == other.offset
    }

    override fun hashCode(): Int {
        return 31 * name.hashCode() + offset.hashCode()
    }
}

data class GgufHeader(
    val version: Int,
    val tensorCount: Long,
    val metadataKvCount: Long
)

data class GgufMetadata(
    val architecture: String = "llama",
    val modelName: String = "Unknown",
    val contextLength: Int = 4096,
    val embeddingLength: Int = 2048,
    val blockCount: Int = 16,
    val feedForwardLength: Int = 5632,
    val headCount: Int = 32,
    val headCountKv: Int = 32,
    val ropeFreqBase: Float = 10000.0f,
    val vocabSize: Int = 32000,
    val tensorDataOffset: Long = 0L,
    val totalFileSizeBytes: Long = 0L,
    val primaryQuantType: GgmlTensorType = GgmlTensorType.Q4_0,
    val tensors: List<GgufTensorInfo> = emptyList(),
    val rawMetadata: Map<String, Any> = emptyMap(),
    val isMemoryMapped: Boolean = true
) {
    val estimatedMmapVirtualMb: Double
        get() = totalFileSizeBytes / (1024.0 * 1024.0)

    val theoreticalMinRamMb: Double
        get() = 12.0 // With pure mmap, only active pages (< 20MB) touch physical RAM!
}
