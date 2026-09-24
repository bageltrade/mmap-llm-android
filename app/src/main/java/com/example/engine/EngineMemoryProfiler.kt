package com.example.engine

import android.os.Debug
import android.os.Process
import java.io.File

data class MemorySnapshot(
    val jvmHeapUsedMb: Double,
    val jvmHeapMaxMb: Double,
    val residentSetSizeMb: Double, // Real physical RAM (RSS)
    val virtualMemoryMb: Double,  // Total virtual address space (including mmap files)
    val mmapFileMappedMb: Double, // Size of mapped model weights on flash
    val activeKvCacheRamMb: Double,
    val isUnderRamBudget: Boolean,
    val peakRssRecordedMb: Double
)

object EngineMemoryProfiler {
    private var peakRssSeenMb: Double = 0.0

    fun sample(mmapFileSizeBytes: Long = 0L, activeKvRamMb: Double = 0.0, ramBudgetMb: Float = 35.0f): MemorySnapshot {
        val runtime = Runtime.getRuntime()
        val heapUsed = (runtime.totalMemory() - runtime.freeMemory()) / (1024.0 * 1024.0)
        val heapMax = runtime.maxMemory() / (1024.0 * 1024.0)

        var rssMb = 0.0
        var vSizeMb = 0.0

        try {
            // Read /proc/self/statm for high-accuracy Linux page counts (page size = 4KB)
            val statm = File("/proc/self/statm")
            if (statm.exists()) {
                val parts = statm.readText().trim().split("\\s+".toRegex())
                if (parts.size >= 2) {
                    val vPages = parts[0].toLongOrNull() ?: 0L
                    val rPages = parts[1].toLongOrNull() ?: 0L
                    vSizeMb = (vPages * 4096.0) / (1024.0 * 1024.0)
                    rssMb = (rPages * 4096.0) / (1024.0 * 1024.0)
                }
            }
        } catch (_: Exception) {
            // Fallback to Android Debug API
            val memInfo = Debug.MemoryInfo()
            Debug.getMemoryInfo(memInfo)
            rssMb = memInfo.totalPss / 1024.0
        }

        if (rssMb <= 0.0) {
            rssMb = heapUsed + 6.0 // Conservative estimate
        }

        if (rssMb > peakRssSeenMb) {
            peakRssSeenMb = rssMb
        }

        val mmapMappedMb = mmapFileSizeBytes / (1024.0 * 1024.0)
        if (vSizeMb < mmapMappedMb) {
            vSizeMb = mmapMappedMb + rssMb
        }

        return MemorySnapshot(
            jvmHeapUsedMb = heapUsed,
            jvmHeapMaxMb = heapMax,
            residentSetSizeMb = rssMb,
            virtualMemoryMb = vSizeMb,
            mmapFileMappedMb = mmapMappedMb,
            activeKvCacheRamMb = activeKvRamMb,
            isUnderRamBudget = rssMb <= ramBudgetMb,
            peakRssRecordedMb = peakRssSeenMb
        )
    }

    fun resetPeak() {
        peakRssSeenMb = 0.0
    }
}
