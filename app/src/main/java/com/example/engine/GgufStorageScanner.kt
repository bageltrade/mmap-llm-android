package com.example.engine

import android.content.Context
import android.os.Environment
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

data class DiscoveredGgufModel(
    val name: String,
    val file: File,
    val sizeBytes: Long,
    val lastModified: Long,
    val isBundled: Boolean = false
) {
    val formattedSize: String
        get() {
            val mb = sizeBytes / (1024.0 * 1024.0)
            return if (mb >= 1024.0) {
                String.format("%.2f GB", mb / 1024.0)
            } else {
                String.format("%.1f MB", mb)
            }
        }
}

/**
 * Storage Scanner that searches storage directories specifically for .gguf model files.
 * Rejects all other non-GGUF file extensions.
 */
class GgufStorageScanner(private val context: Context) {
    companion object {
        private const val TAG = "GgufStorageScanner"
    }

    private val parser = GgufParser()

    suspend fun scanStorageForGgufFiles(): List<DiscoveredGgufModel> = withContext(Dispatchers.IO) {
        val results = mutableListOf<DiscoveredGgufModel>()
        val searchDirs = mutableListOf<File>()

        // 1. App internal models directory
        val internalModels = File(context.filesDir, "models")
        if (internalModels.exists()) searchDirs.add(internalModels)

        // 2. Standard Downloads and Documents
        val publicDownloads = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        if (publicDownloads != null && publicDownloads.exists() && publicDownloads.canRead()) {
            searchDirs.add(publicDownloads)
        }

        val publicDocs = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
        if (publicDocs != null && publicDocs.exists() && publicDocs.canRead()) {
            searchDirs.add(publicDocs)
        }

        // 3. External root folders if accessible
        try {
            val extRoot = Environment.getExternalStorageDirectory()
            if (extRoot != null && extRoot.exists() && extRoot.canRead()) {
                val candidateDirs = listOf("Models", "LLM", "llama", "gguf", "ollama")
                for (name in candidateDirs) {
                    val sub = File(extRoot, name)
                    if (sub.exists() && sub.canRead()) searchDirs.add(sub)
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "External storage directory scan restricted: ${e.message}")
        }

        val scannedPaths = mutableSetOf<String>()

        for (dir in searchDirs) {
            scanDirectoryRecursively(dir, results, scannedPaths, maxDepth = 2)
        }

        results.sortedByDescending { it.lastModified }
    }

    private fun scanDirectoryRecursively(
        dir: File,
        results: MutableList<DiscoveredGgufModel>,
        scannedPaths: MutableSet<String>,
        maxDepth: Int
    ) {
        if (maxDepth < 0 || !dir.canRead()) return

        val files = dir.listFiles() ?: return
        for (f in files) {
            if (f.isDirectory) {
                if (!f.name.startsWith(".")) {
                    scanDirectoryRecursively(f, results, scannedPaths, maxDepth - 1)
                }
            } else if (f.isFile) {
                // Strictly only .gguf files
                if (f.name.endsWith(".gguf", ignoreCase = true)) {
                    if (scannedPaths.add(f.absolutePath)) {
                        val validation = parser.validateGguf(f)
                        if (validation.isSuccess) {
                            results.add(
                                DiscoveredGgufModel(
                                    name = f.nameWithoutExtension,
                                    file = f,
                                    sizeBytes = f.length(),
                                    lastModified = f.lastModified(),
                                    isBundled = f.parentFile?.name == "models"
                                )
                            )
                        }
                    }
                }
            }
        }
    }
}
