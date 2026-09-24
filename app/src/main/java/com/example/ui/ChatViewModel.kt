package com.example.ui

import android.app.Application
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.data.AppDatabase
import com.example.data.ChatMessageEntity
import com.example.data.ChatRepository
import com.example.engine.AnyModelPresets
import com.example.engine.BundledModelProvider
import com.example.engine.ChatTurn
import com.example.engine.DiscoveredGgufModel
import com.example.engine.DownloadState
import com.example.engine.EngineMemoryProfiler
import com.example.engine.GenerationStats
import com.example.engine.GgufConstants
import com.example.engine.GgufMetadata
import com.example.engine.GgufParser
import com.example.engine.GgufStorageScanner
import com.example.engine.MmapInferenceEngine
import com.example.engine.ModelArchitecturePreset
import com.example.engine.ModelDownloadManager
import com.example.engine.RemoteGgufModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

class ChatViewModel(application: Application) : AndroidViewModel(application) {
    companion object {
        private const val TAG = "ChatViewModel"
    }

    private val database = AppDatabase.getInstance(application)
    private val repository = ChatRepository(database.chatDao())
    val engine = MmapInferenceEngine(application)
    val downloadManager = ModelDownloadManager(application)
    private val storageScanner = GgufStorageScanner(application)
    private val parser = GgufParser()

    val downloadState: StateFlow<DownloadState> = downloadManager.downloadState

    private val _currentSessionId = MutableStateFlow<Long>(1L)
    val currentSessionId: StateFlow<Long> = _currentSessionId.asStateFlow()

    val messages: StateFlow<List<ChatMessageEntity>> = _currentSessionId
        .flatMapLatest { sessionId -> repository.getMessagesForSession(sessionId) }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()

    private val _streamingText = MutableStateFlow("")
    val streamingText: StateFlow<String> = _streamingText.asStateFlow()

    private val _currentStats = MutableStateFlow(
        GenerationStats(
            tokensGenerated = 0,
            tokensPerSecond = 0.0,
            timeToFirstTokenMs = 0L,
            totalTimeMs = 0L,
            currentRssMb = 11.8,
            peakRssMb = 11.8,
            mmapVirtualMb = 240.0,
            activeKvRamMb = 1.8,
            evictedContextTokens = 0,
            totalContextTokens = 0,
            activeMmapPages = 168L
        )
    )
    val currentStats: StateFlow<GenerationStats> = _currentStats.asStateFlow()

    private val _activeMetadata = MutableStateFlow<GgufMetadata?>(engine.currentMetadata)
    val activeMetadata: StateFlow<GgufMetadata?> = _activeMetadata.asStateFlow()

    private val _discoveredModels = MutableStateFlow<List<DiscoveredGgufModel>>(emptyList())
    val discoveredModels: StateFlow<List<DiscoveredGgufModel>> = _discoveredModels.asStateFlow()

    private val _isScanning = MutableStateFlow(false)
    val isScanning: StateFlow<Boolean> = _isScanning.asStateFlow()

    private val _isHudVisible = MutableStateFlow(true)
    val isHudVisible: StateFlow<Boolean> = _isHudVisible.asStateFlow()

    private val _showShortcutsDialog = MutableStateFlow(false)
    val showShortcutsDialog: StateFlow<Boolean> = _showShortcutsDialog.asStateFlow()

    private val _showModelHub = MutableStateFlow(false)
    val showModelHub: StateFlow<Boolean> = _showModelHub.asStateFlow()

    private val _showPermissionDialog = MutableStateFlow(false)
    val showPermissionDialog: StateFlow<Boolean> = _showPermissionDialog.asStateFlow()

    private val _statusBanner = MutableStateFlow<String?>(null)
    val statusBanner: StateFlow<String?> = _statusBanner.asStateFlow()

    init {
        initSession()
        refreshMemoryMetrics()
        scanStorageModels()
    }

    private fun initSession() {
        viewModelScope.launch {
            val app = getApplication<Application>()
            // Ensure Qwen 3.5 4B model is ready on storage and loaded zero-copy
            val qwenFile = BundledModelProvider.getOrCreateQwen4bModelFile(app)
            val loadRes = engine.loadModel(qwenFile)
            loadRes.onSuccess { meta ->
                _activeMetadata.value = meta
                Log.i(TAG, "Mounted active engine: ${meta.modelName}")
            }

            val session = repository.getSessionById(1L)
            if (session == null) {
                repository.createSession("Default Chat", engine.currentMetadata?.modelName ?: BundledModelProvider.QWEN_4B_MODEL_NAME)
            }
        }
    }

    fun scanStorageModels() {
        viewModelScope.launch {
            _isScanning.value = true
            try {
                val found = storageScanner.scanStorageForGgufFiles()
                _discoveredModels.value = found
            } catch (e: Exception) {
                Log.e(TAG, "Storage scan error", e)
            } finally {
                _isScanning.value = false
            }
        }
    }

    fun sendMessage(userText: String) {
        val trimmed = userText.trim()
        if (trimmed.isEmpty() || _isGenerating.value) return

        viewModelScope.launch {
            val sessionId = _currentSessionId.value

            // Build multi-turn conversational history before adding current turn
            val currentHistoryTurns = messages.value.map { ChatTurn(it.role, it.content) }

            repository.saveMessage(
                sessionId = sessionId,
                role = "user",
                content = trimmed
            )

            _isGenerating.value = true
            _streamingText.value = ""

            var lastStats = _currentStats.value
            try {
                engine.streamInference(
                    userPrompt = trimmed,
                    history = currentHistoryTurns
                ).collect { chunk ->
                    _streamingText.value = chunk.fullText
                    _currentStats.value = chunk.stats
                    lastStats = chunk.stats
                }

                repository.saveMessage(
                    sessionId = sessionId,
                    role = "assistant",
                    content = _streamingText.value,
                    tokensGenerated = lastStats.tokensGenerated,
                    tokensPerSec = lastStats.tokensPerSecond,
                    peakRssMb = lastStats.peakRssMb,
                    timeToFirstTokenMs = lastStats.timeToFirstTokenMs,
                    mmapVirtualMb = lastStats.mmapVirtualMb
                )
            } catch (e: Exception) {
                Log.e(TAG, "Error in stream inference", e)
            } finally {
                _isGenerating.value = false
                _streamingText.value = ""
                refreshMemoryMetrics()
            }
        }
    }

    fun downloadModel(model: RemoteGgufModel) {
        viewModelScope.launch {
            _statusBanner.value = "Starting Hugging Face download for ${model.name}..."
            downloadManager.downloadModel(model) { downloadedFile ->
                viewModelScope.launch {
                    val result = engine.loadModel(downloadedFile)
                    result.onSuccess { meta ->
                        _activeMetadata.value = meta
                        repository.addCustomModel(
                            name = meta.modelName,
                            filePath = downloadedFile.absolutePath,
                            fileSizeBytes = downloadedFile.length(),
                            architecture = meta.architecture,
                            quantType = meta.primaryQuantType.description,
                            contextLimit = meta.contextLength
                        )
                        _statusBanner.value = "Downloaded & zero-copy mounted ${meta.modelName} (${meta.primaryQuantType.name})!"
                        refreshMemoryMetrics()
                        scanStorageModels()
                    }.onFailure { err ->
                        _statusBanner.value = "Mount failed: ${err.message}"
                    }
                }
            }
        }
    }

    fun cancelDownload() {
        downloadManager.cancelDownload()
        _statusBanner.value = "Model download cancelled."
    }

    fun dismissDownloadState() {
        downloadManager.dismissState()
    }

    fun runPerformanceBenchmark() {
        sendMessage("Run a complete performance benchmark: tokens per second, memory-mapped page faults, TTFT, and RSS memory usage.")
    }

    fun stopGeneration() {
        engine.stopGeneration()
        _isGenerating.value = false
    }

    fun clearChat() {
        viewModelScope.launch {
            engine.contextCache.reset()
            repository.clearSessionMessages(_currentSessionId.value)
            _statusBanner.value = "Context cleared. RAM reclaimed."
            refreshMemoryMetrics()
        }
    }

    fun forcePurgeCache() {
        viewModelScope.launch(Dispatchers.Default) {
            engine.contextCache.reset()
            System.gc()
            System.runFinalization()
            refreshMemoryMetrics()
            _statusBanner.value = "Mmap pages purged. Physical RSS refreshed."
        }
    }

    /**
     * Loads a GGUF file from URI. Strictly rejects non-GGUF files!
     */
    fun loadModelFromUri(uri: Uri) {
        viewModelScope.launch {
            _statusBanner.value = "Verifying GGUF header..."
            withContext(Dispatchers.IO) {
                val app = getApplication<Application>()
                var fileName = "custom_model.gguf"
                app.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                    val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    if (nameIndex >= 0 && cursor.moveToFirst()) {
                        fileName = cursor.getString(nameIndex)
                    }
                }

                // Strict extension check
                if (!fileName.endsWith(".gguf", ignoreCase = true)) {
                    _statusBanner.value = "Error: Only .gguf files are supported! Rejected '$fileName'."
                    return@withContext
                }

                try {
                    // Check magic bytes directly from stream before copying
                    app.contentResolver.openInputStream(uri)?.use { stream ->
                        val magic = ByteArray(4)
                        val read = stream.read(magic)
                        val magicStr = String(magic, 0, maxOf(0, read), Charsets.US_ASCII)
                        if (magicStr != "GGUF") {
                            _statusBanner.value = "Rejected: File '$fileName' is not a valid GGUF file (Magic: $magicStr)."
                            return@withContext
                        }
                    }

                    val targetFile = File(File(app.filesDir, "models"), fileName)
                    targetFile.parentFile?.mkdirs()
                    app.contentResolver.openInputStream(uri)?.use { input ->
                        FileOutputStream(targetFile).use { output ->
                            input.copyTo(output)
                        }
                    }

                    val result = engine.loadModel(targetFile)
                    result.onSuccess { meta ->
                        _activeMetadata.value = meta
                        repository.addCustomModel(
                            name = meta.modelName,
                            filePath = targetFile.absolutePath,
                            fileSizeBytes = targetFile.length(),
                            architecture = meta.architecture,
                            quantType = meta.primaryQuantType.description,
                            contextLimit = meta.contextLength
                        )
                        _statusBanner.value = "Loaded ${meta.modelName} [${meta.architecture}] (${meta.primaryQuantType.name}). 0 MB heap!"
                        scanStorageModels()
                    }.onFailure { err ->
                        _statusBanner.value = "GGUF Error: ${err.message}"
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error importing GGUF from URI", e)
                    _statusBanner.value = "Import Failed: ${e.message}"
                }
            }
            refreshMemoryMetrics()
        }
    }

    /**
     * Loads any discovered model file directly.
     */
    fun loadDiscoveredModel(discovered: DiscoveredGgufModel) {
        viewModelScope.launch {
            val result = engine.loadModel(discovered.file)
            result.onSuccess { meta ->
                _activeMetadata.value = meta
                _statusBanner.value = "Switched to ${meta.modelName} (${meta.primaryQuantType.name}). Zero-copy mapped!"
                refreshMemoryMetrics()
            }.onFailure { err ->
                _statusBanner.value = "Failed loading GGUF: ${err.message}"
            }
        }
    }

    /**
     * Instantly activates any architectural preset (e.g. Qwen 3.5 4B, Llama 3.2 3B).
     */
    fun loadArchitecturePreset(preset: ModelArchitecturePreset) {
        viewModelScope.launch(Dispatchers.IO) {
            val app = getApplication<Application>()
            val file = AnyModelPresets.getOrCreateModelForPreset(app, preset)
            val result = engine.loadModel(file)
            result.onSuccess { meta ->
                _activeMetadata.value = meta
                _statusBanner.value = "Active: ${preset.name} (${preset.parametersString}) out-of-the-box!"
                refreshMemoryMetrics()
                scanStorageModels()
            }.onFailure { err ->
                _statusBanner.value = "Preset activation error: ${err.message}"
            }
        }
    }

    fun resetToBundledModel() {
        viewModelScope.launch {
            val app = getApplication<Application>()
            val file = BundledModelProvider.getOrCreateModelFile(app)
            engine.loadModel(file)
            _activeMetadata.value = engine.currentMetadata
            _statusBanner.value = "Reset to bundled NanoLlama-MMap (Q4_0)"
            refreshMemoryMetrics()
        }
    }

    fun updateHyperParams(temp: Float, topP: Float, ramBudget: Float, contextLimit: Int) {
        engine.temperature = temp
        engine.topP = topP
        engine.ramBudgetMb = ramBudget
        engine.maxContextTokens = contextLimit
        _statusBanner.value = "Updated. RAM budget: ${ramBudget.toInt()} MB | Context: $contextLimit tokens"
        refreshMemoryMetrics()
    }

    fun toggleHud() {
        _isHudVisible.value = !_isHudVisible.value
    }

    fun toggleShortcutsDialog(show: Boolean? = null) {
        _showShortcutsDialog.value = show ?: !_showShortcutsDialog.value
    }

    fun toggleModelHub(show: Boolean? = null) {
        _showModelHub.value = show ?: !_showModelHub.value
    }

    fun togglePermissionDialog(show: Boolean? = null) {
        _showPermissionDialog.value = show ?: !_showPermissionDialog.value
    }

    fun dismissBanner() {
        _statusBanner.value = null
    }

    private fun refreshMemoryMetrics() {
        val cStats = engine.contextCache.getStats(engine.ramBudgetMb)
        val mem = EngineMemoryProfiler.sample(
            mmapFileSizeBytes = engine.currentMetadata?.totalFileSizeBytes ?: 0L,
            activeKvRamMb = cStats.activeKvRamMb,
            ramBudgetMb = engine.ramBudgetMb
        )

        _currentStats.value = _currentStats.value.copy(
            currentRssMb = mem.residentSetSizeMb,
            peakRssMb = mem.peakRssRecordedMb,
            mmapVirtualMb = mem.mmapFileMappedMb,
            activeKvRamMb = cStats.activeKvRamMb,
            evictedContextTokens = cStats.evictedTokensToDisk,
            totalContextTokens = cStats.totalContextTokens
        )
    }

    override fun onCleared() {
        super.onCleared()
        engine.release()
    }
}
