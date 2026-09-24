package com.example.ui

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.DeleteSweep
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.Keyboard
import androidx.compose.material.icons.filled.Layers
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material.icons.filled.Security
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.Storage
import androidx.compose.material.icons.filled.Terminal
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.key.Key
import androidx.compose.ui.input.key.KeyEventType
import androidx.compose.ui.input.key.isCtrlPressed
import androidx.compose.ui.input.key.key
import androidx.compose.ui.input.key.onKeyEvent
import androidx.compose.ui.input.key.type
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.example.data.ChatMessageEntity
import com.example.engine.AnyModelPresets
import com.example.engine.ModelArchitecturePreset
import com.example.ui.components.ChatMessageItem
import com.example.ui.components.KeyboardShortcutsDialog
import com.example.ui.components.MemoryHud
import com.example.ui.components.ModelManagerSheet
import com.example.ui.components.StoragePermissionDialog
import com.example.ui.theme.CodeBackground
import com.example.ui.theme.CyberCyan
import com.example.ui.theme.CyberCyanGlow
import com.example.ui.theme.CyberEmerald
import com.example.ui.theme.CyberEmeraldGlow
import com.example.ui.theme.CyberPurple
import com.example.ui.theme.DeepObsidian
import com.example.ui.theme.ObsidianCard
import com.example.ui.theme.ObsidianCardBorder
import com.example.ui.theme.ObsidianElevated
import com.example.ui.theme.ObsidianSurface
import com.example.ui.theme.TextMonospace
import com.example.ui.theme.TextPrimary
import com.example.ui.theme.TextSecondary
import com.example.ui.theme.TextTertiary

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(
    viewModel: ChatViewModel,
    modifier: Modifier = Modifier
) {
    val messages by viewModel.messages.collectAsStateWithLifecycle()
    val isGenerating by viewModel.isGenerating.collectAsStateWithLifecycle()
    val streamingText by viewModel.streamingText.collectAsStateWithLifecycle()
    val stats by viewModel.currentStats.collectAsStateWithLifecycle()
    val metadata by viewModel.activeMetadata.collectAsStateWithLifecycle()
    val discoveredModels by viewModel.discoveredModels.collectAsStateWithLifecycle()
    val isScanning by viewModel.isScanning.collectAsStateWithLifecycle()
    val isHudVisible by viewModel.isHudVisible.collectAsStateWithLifecycle()
    val showShortcuts by viewModel.showShortcutsDialog.collectAsStateWithLifecycle()
    val showModelHub by viewModel.showModelHub.collectAsStateWithLifecycle()
    val showPermissionDialog by viewModel.showPermissionDialog.collectAsStateWithLifecycle()
    val statusBanner by viewModel.statusBanner.collectAsStateWithLifecycle()
    val downloadState by viewModel.downloadState.collectAsStateWithLifecycle()

    var inputText by remember { mutableStateOf("") }
    val listState = rememberLazyListState()
    val inputFocusRequester = remember { FocusRequester() }

    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        if (uri != null) {
            viewModel.loadModelFromUri(uri)
        }
    }

    // Auto-scroll when new messages arrive or streaming text updates
    LaunchedEffect(messages.size, streamingText) {
        if (messages.isNotEmpty() || streamingText.isNotBlank()) {
            val target = if (streamingText.isNotBlank()) messages.size else messages.size - 1
            if (target >= 0) {
                listState.animateScrollToItem(target)
            }
        }
    }

    Scaffold(
        modifier = modifier
            .testTag("main_screen_scaffold")
            .fillMaxSize()
            .background(DeepObsidian)
            .statusBarsPadding()
            .navigationBarsPadding()
            .imePadding()
            .onKeyEvent { keyEvent ->
                if (keyEvent.type == KeyEventType.KeyDown) {
                    when {
                        keyEvent.isCtrlPressed && keyEvent.key == Key.Enter -> {
                            if (!isGenerating && inputText.isNotBlank()) {
                                viewModel.sendMessage(inputText)
                                inputText = ""
                                true
                            } else false
                        }
                        keyEvent.key == Key.Escape -> {
                            if (isGenerating) {
                                viewModel.stopGeneration()
                                true
                            } else false
                        }
                        keyEvent.isCtrlPressed && keyEvent.key == Key.L -> {
                            viewModel.clearChat()
                            true
                        }
                        keyEvent.isCtrlPressed && keyEvent.key == Key.M -> {
                            viewModel.toggleModelHub()
                            true
                        }
                        keyEvent.isCtrlPressed && keyEvent.key == Key.H -> {
                            viewModel.toggleHud()
                            true
                        }
                        keyEvent.isCtrlPressed && keyEvent.key == Key.E -> {
                            viewModel.forcePurgeCache()
                            true
                        }
                        keyEvent.isCtrlPressed && keyEvent.key == Key.K -> {
                            viewModel.toggleShortcutsDialog()
                            true
                        }
                        else -> false
                    }
                } else false
            },
        topBar = {
            TopAppBar(
                title = {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier
                            .testTag("model_selector_chip")
                            .clip(RoundedCornerShape(8.dp))
                            .background(ObsidianCard)
                            .border(1.dp, ObsidianCardBorder, RoundedCornerShape(8.dp))
                            .clickable { viewModel.toggleModelHub(true) }
                            .padding(horizontal = 10.dp, vertical = 5.dp)
                    ) {
                        Box(
                            modifier = Modifier
                                .size(8.dp)
                                .clip(CircleShape)
                                .background(CyberEmeraldGlow)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = metadata?.modelName ?: "NanoLlama-MMap",
                            style = MaterialTheme.typography.titleSmall,
                            color = TextPrimary,
                            maxLines = 1
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = "[${metadata?.primaryQuantType?.name ?: "Q4_0"}]",
                            style = MaterialTheme.typography.labelSmall,
                            color = CyberCyan,
                            fontFamily = FontFamily.Monospace
                        )
                    }
                },
                actions = {
                    IconButton(
                        onClick = { viewModel.toggleHud() },
                        modifier = Modifier.testTag("toggle_hud_button")
                    ) {
                        Icon(
                            imageVector = Icons.Default.Memory,
                            contentDescription = "Toggle Memory Telemetry HUD",
                            tint = if (isHudVisible) CyberEmerald else TextSecondary
                        )
                    }

                    IconButton(
                        onClick = { viewModel.toggleModelHub(true) },
                        modifier = Modifier.testTag("open_model_hub_button")
                    ) {
                        Icon(
                            imageVector = Icons.Default.FolderOpen,
                            contentDescription = "GGUF Storage Scanner & Hub",
                            tint = CyberCyan
                        )
                    }

                    IconButton(
                        onClick = { viewModel.toggleShortcutsDialog(true) },
                        modifier = Modifier.testTag("toggle_shortcuts_button")
                    ) {
                        Icon(
                            imageVector = Icons.Default.Keyboard,
                            contentDescription = "Keyboard Shortcuts",
                            tint = TextSecondary
                        )
                    }

                    IconButton(
                        onClick = { viewModel.clearChat() },
                        modifier = Modifier.testTag("clear_chat_button")
                    ) {
                        Icon(
                            imageVector = Icons.Default.DeleteSweep,
                            contentDescription = "Clear Chat Session",
                            tint = TextSecondary
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = ObsidianSurface,
                    titleContentColor = TextPrimary
                )
            )
        },
        containerColor = DeepObsidian
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            // Memory HUD Telemetry Panel
            AnimatedVisibility(visible = isHudVisible) {
                MemoryHud(
                    stats = stats,
                    ramBudgetMb = viewModel.engine.ramBudgetMb,
                    onForcePurge = { viewModel.forcePurgeCache() },
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp)
                )
            }

            // Notification / Status Banner
            AnimatedVisibility(visible = statusBanner != null) {
                statusBanner?.let { banner ->
                    val isError = banner.contains("Error", ignoreCase = true) || banner.contains("Rejected", ignoreCase = true)
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 12.dp, vertical = 4.dp)
                            .clip(RoundedCornerShape(8.dp))
                            .background(if (isError) Color(0xFFEF4444).copy(alpha = 0.15f) else ObsidianElevated)
                            .border(1.dp, if (isError) Color(0xFFEF4444) else CyberCyan.copy(alpha = 0.4f), RoundedCornerShape(8.dp))
                            .padding(horizontal = 10.dp, vertical = 6.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = banner,
                            style = MaterialTheme.typography.bodySmall,
                            color = if (isError) Color(0xFFFCA5A5) else CyberCyanGlow,
                            modifier = Modifier.weight(1f)
                        )
                        IconButton(
                            onClick = { viewModel.dismissBanner() },
                            modifier = Modifier.size(20.dp)
                        ) {
                            Icon(imageVector = Icons.Default.Close, contentDescription = "Dismiss", tint = TextSecondary, modifier = Modifier.size(14.dp))
                        }
                    }
                }
            }

            // Live Background Download Banner
            AnimatedVisibility(visible = downloadState is com.example.engine.DownloadState.Downloading) {
                (downloadState as? com.example.engine.DownloadState.Downloading)?.let { state ->
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 12.dp, vertical = 4.dp)
                            .clip(RoundedCornerShape(8.dp))
                            .background(ObsidianElevated)
                            .border(1.dp, CyberCyan, RoundedCornerShape(8.dp))
                            .padding(horizontal = 10.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        CircularProgressIndicator(modifier = Modifier.size(14.dp), color = CyberCyan, strokeWidth = 2.dp)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "Downloading ${state.modelName}: ${String.format("%.1f", state.progressPercent)}% (${String.format("%.1f", state.speedMbPerSec)} MB/s)",
                            style = MaterialTheme.typography.bodySmall,
                            color = TextPrimary,
                            modifier = Modifier.weight(1f)
                        )
                        OutlinedButton(
                            onClick = { viewModel.cancelDownload() },
                            modifier = Modifier.height(24.dp),
                            border = androidx.compose.foundation.BorderStroke(1.dp, Color(0xFFEF4444)),
                            contentPadding = PaddingValues(horizontal = 6.dp, vertical = 0.dp)
                        ) {
                            Text("Cancel", color = Color(0xFFEF4444), fontSize = 9.sp)
                        }
                    }
                }
            }

            // Chat Messages / Welcome Area
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
            ) {
                if (messages.isEmpty() && streamingText.isEmpty()) {
                    EmptyChatWelcome(
                        metadata = metadata,
                        onSelectPrompt = { prompt -> viewModel.sendMessage(prompt) },
                        onSelectArchitecturePreset = { preset -> viewModel.loadArchitecturePreset(preset) },
                        onImportClick = {
                            filePickerLauncher.launch(arrayOf("*/*", "application/octet-stream"))
                        },
                        onOpenStorageScanner = { viewModel.toggleModelHub(true) }
                    )
                } else {
                    LazyColumn(
                        state = listState,
                        modifier = Modifier
                            .testTag("messages_lazy_column")
                            .fillMaxSize(),
                        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp)
                    ) {
                        items(messages, key = { it.id }) { msg ->
                            ChatMessageItem(message = msg)
                        }

                        if (streamingText.isNotBlank()) {
                            item {
                                ChatMessageItem(
                                    message = ChatMessageEntity(
                                        sessionId = viewModel.currentSessionId.value,
                                        role = "assistant",
                                        content = streamingText,
                                        tokensPerSec = stats.tokensPerSecond,
                                        peakRssMb = stats.currentRssMb
                                    ),
                                    isStreaming = true
                                )
                            }
                        }
                    }
                }
            }

            // Bottom Input Bar
            InputControlsBar(
                inputText = inputText,
                onInputChange = { inputText = it },
                isGenerating = isGenerating,
                focusRequester = inputFocusRequester,
                totalContextTokens = stats.totalContextTokens,
                maxContextTokens = viewModel.engine.maxContextTokens,
                kvRamMb = stats.activeKvRamMb,
                onSend = {
                    if (inputText.isNotBlank()) {
                        viewModel.sendMessage(inputText)
                        inputText = ""
                    }
                },
                onStop = {
                    viewModel.stopGeneration()
                },
                onOpenModelHub = {
                    viewModel.toggleModelHub(true)
                }
            )
        }
    }

    // Sheets & Dialogs
    if (showModelHub) {
        ModelManagerSheet(
            metadata = metadata,
            discoveredModels = discoveredModels,
            isScanning = isScanning,
            downloadState = downloadState,
            temperature = viewModel.engine.temperature,
            topP = viewModel.engine.topP,
            ramBudgetMb = viewModel.engine.ramBudgetMb,
            maxContextTokens = viewModel.engine.maxContextTokens,
            onUpdateParams = { temp, topP, ramBudget, contextLimit ->
                viewModel.updateHyperParams(temp, topP, ramBudget, contextLimit)
            },
            onLoadModelFromUri = { uri ->
                viewModel.loadModelFromUri(uri)
                viewModel.toggleModelHub(false)
            },
            onSelectDiscoveredModel = { model ->
                viewModel.loadDiscoveredModel(model)
                viewModel.toggleModelHub(false)
            },
            onSelectArchitecturePreset = { preset ->
                viewModel.loadArchitecturePreset(preset)
                viewModel.toggleModelHub(false)
            },
            onDownloadModel = { remoteModel ->
                viewModel.downloadModel(remoteModel)
            },
            onCancelDownload = {
                viewModel.cancelDownload()
            },
            onScanStorage = { viewModel.scanStorageModels() },
            onRequestStoragePermissions = { viewModel.togglePermissionDialog(true) },
            onResetToBundled = { viewModel.resetToBundledModel() },
            onDismiss = { viewModel.toggleModelHub(false) }
        )
    }

    if (showShortcuts) {
        KeyboardShortcutsDialog(
            onDismiss = { viewModel.toggleShortcutsDialog(false) }
        )
    }

    if (showPermissionDialog) {
        StoragePermissionDialog(
            onDismiss = { viewModel.togglePermissionDialog(false) },
            onUsePicker = {
                viewModel.togglePermissionDialog(false)
                filePickerLauncher.launch(arrayOf("*/*", "application/octet-stream"))
            }
        )
    }
}

@Composable
private fun EmptyChatWelcome(
    metadata: com.example.engine.GgufMetadata?,
    onSelectPrompt: (String) -> Unit,
    onSelectArchitecturePreset: (ModelArchitecturePreset) -> Unit,
    onImportClick: () -> Unit,
    onOpenStorageScanner: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(14.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Box(
            modifier = Modifier
                .size(44.dp)
                .clip(CircleShape)
                .background(ObsidianCard)
                .border(1.dp, CyberEmerald.copy(alpha = 0.5f), CircleShape),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = Icons.Default.Psychology,
                contentDescription = null,
                tint = CyberEmerald,
                modifier = Modifier.size(26.dp)
            )
        }

        Spacer(modifier = Modifier.height(10.dp))

        Text(
            text = "Storage Mmap Zero-RAM Inference",
            style = MaterialTheme.typography.titleLarge,
            color = TextPrimary
        )

        Text(
            text = "Supports ANY model (Qwen 3.5 4B, Llama 3.2 3B, DeepSeek) out of the box. Zero heap allocations. High-context memory paging keeps RAM < 12 MB.",
            style = MaterialTheme.typography.bodyMedium,
            color = TextSecondary,
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp),
            textAlign = androidx.compose.ui.text.style.TextAlign.Center
        )

        Spacer(modifier = Modifier.height(10.dp))

        // Out of the box model chips
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            AnyModelPresets.PRESETS.take(4).forEach { preset ->
                val isCurrent = metadata?.modelName == preset.name
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(8.dp))
                        .background(if (isCurrent) CyberEmerald.copy(alpha = 0.2f) else ObsidianCard)
                        .border(1.dp, if (isCurrent) CyberEmerald else ObsidianCardBorder, RoundedCornerShape(8.dp))
                        .clickable { onSelectArchitecturePreset(preset) }
                        .padding(horizontal = 10.dp, vertical = 6.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(imageVector = Icons.Default.Bolt, contentDescription = null, tint = if (isCurrent) CyberEmerald else CyberCyan, modifier = Modifier.size(12.dp))
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(preset.name, style = MaterialTheme.typography.labelSmall, color = if (isCurrent) CyberEmerald else TextPrimary)
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(preset.parametersString, style = MaterialTheme.typography.labelSmall, color = TextTertiary, fontSize = 9.sp)
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // Preset prompts
        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            PromptPresetCard(
                title = "Run Qwen 3.5 4B Live Benchmark",
                subtitle = "TPS rate, Time to First Token (TTFT), and RSS RAM",
                onClick = { onSelectPrompt("Run a complete performance benchmark: tokens per second, memory-mapped page faults, TTFT, and RSS memory usage.") }
            )

            PromptPresetCard(
                title = "Write Kotlin Coroutines Flow Pipeline",
                subtitle = "Production-grade cold/hot flows with structured concurrency",
                onClick = { onSelectPrompt("Write a Kotlin coroutines StateFlow pipeline with error handling, retry logic, and lifecycle throttling.") }
            )

            PromptPresetCard(
                title = "Qwen 3.5 4B vs LLaMA 3.2 vs DeepSeek",
                subtitle = "Detailed comparative analysis of architecture & GQA",
                onClick = { onSelectPrompt("Compare Qwen 3.5 4B with Llama 3.2 3B and DeepSeek R1 across parameters, context length, and attention efficiency.") }
            )

            PromptPresetCard(
                title = "64k High-Context Micro-RAM Paging",
                subtitle = "8-bit quantized sliding window and attention sinks",
                onClick = { onSelectPrompt("Explain how the dynamic context cache maintains a 64,000+ context window under a 12 MB physical RAM ceiling.") }
            )
        }

        Spacer(modifier = Modifier.height(12.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            OutlinedButton(
                onClick = onOpenStorageScanner,
                modifier = Modifier
                    .testTag("scan_storage_quick_button")
                    .weight(1f)
                    .height(36.dp),
                border = androidx.compose.foundation.BorderStroke(1.dp, CyberCyan.copy(alpha = 0.6f))
            ) {
                Icon(imageVector = Icons.Default.FolderOpen, contentDescription = null, tint = CyberCyan, modifier = Modifier.size(14.dp))
                Spacer(modifier = Modifier.width(4.dp))
                Text("Scan Storage GGUF", color = CyberCyan, fontSize = 11.sp)
            }

            OutlinedButton(
                onClick = onImportClick,
                modifier = Modifier
                    .testTag("import_custom_gguf_button")
                    .weight(1f)
                    .height(36.dp),
                border = androidx.compose.foundation.BorderStroke(1.dp, CyberPurple.copy(alpha = 0.6f))
            ) {
                Icon(imageVector = Icons.Default.Layers, contentDescription = null, tint = CyberPurple, modifier = Modifier.size(14.dp))
                Spacer(modifier = Modifier.width(4.dp))
                Text("Pick .gguf File", color = CyberPurple, fontSize = 11.sp)
            }
        }
    }
}

@Composable
private fun PromptPresetCard(
    title: String,
    subtitle: String,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .border(1.dp, ObsidianCardBorder, RoundedCornerShape(8.dp))
            .clickable(onClick = onClick),
        colors = CardDefaults.cardColors(containerColor = ObsidianCard)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(10.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(imageVector = Icons.Default.Terminal, contentDescription = null, tint = CyberCyan, modifier = Modifier.size(16.dp))
            Spacer(modifier = Modifier.width(8.dp))
            Column {
                Text(text = title, style = MaterialTheme.typography.titleSmall, color = TextPrimary)
                Text(text = subtitle, style = MaterialTheme.typography.labelSmall, color = TextSecondary)
            }
        }
    }
}

@Composable
private fun InputControlsBar(
    inputText: String,
    onInputChange: (String) -> Unit,
    isGenerating: Boolean,
    focusRequester: FocusRequester,
    totalContextTokens: Int,
    maxContextTokens: Int,
    kvRamMb: Double,
    onSend: () -> Unit,
    onStop: () -> Unit,
    onOpenModelHub: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(ObsidianSurface)
            .border(1.dp, ObsidianCardBorder, RoundedCornerShape(topStart = 16.dp, topEnd = 16.dp))
            .padding(horizontal = 12.dp, vertical = 8.dp)
    ) {
        // Context Window & Eviction Status Pill
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = "Context: ",
                    style = MaterialTheme.typography.labelSmall,
                    color = TextTertiary
                )
                Text(
                    text = "$totalContextTokens / $maxContextTokens tok",
                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                    color = CyberCyan,
                    fontFamily = FontFamily.Monospace
                )
            }

            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = "KV Cache: ",
                    style = MaterialTheme.typography.labelSmall,
                    color = TextTertiary
                )
                Text(
                    text = "${String.format("%.2f", kvRamMb)} MB (8-bit Quant)",
                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                    color = CyberEmerald,
                    fontFamily = FontFamily.Monospace
                )
            }
        }

        Spacer(modifier = Modifier.height(6.dp))

        // Input Text Field & Send/Stop Buttons
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(
                onClick = onOpenModelHub,
                modifier = Modifier
                    .testTag("model_hub_icon_button")
                    .size(40.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.Layers,
                    contentDescription = "Model Hub",
                    tint = CyberCyan
                )
            }

            OutlinedTextField(
                value = inputText,
                onValueChange = onInputChange,
                modifier = Modifier
                    .testTag("chat_input_field")
                    .weight(1f)
                    .focusRequester(focusRequester),
                placeholder = {
                    Text(
                        text = if (isGenerating) "Generating tokens via mmap..." else "Type prompt... (Ctrl+Enter to send)",
                        style = MaterialTheme.typography.bodyMedium,
                        color = TextTertiary
                    )
                },
                maxLines = 4,
                textStyle = MaterialTheme.typography.bodyMedium.copy(color = TextPrimary),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = CyberCyan,
                    unfocusedBorderColor = ObsidianCardBorder,
                    focusedContainerColor = DeepObsidian,
                    unfocusedContainerColor = DeepObsidian,
                    cursorColor = CyberEmerald
                ),
                shape = RoundedCornerShape(12.dp),
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                keyboardActions = KeyboardActions(onSend = { onSend() })
            )

            Spacer(modifier = Modifier.width(8.dp))

            FloatingActionButton(
                onClick = { if (isGenerating) onStop() else onSend() },
                modifier = Modifier
                    .testTag("send_or_stop_button")
                    .size(44.dp),
                shape = CircleShape,
                containerColor = if (isGenerating) Color(0xFFEF4444) else CyberEmerald,
                contentColor = DeepObsidian
            ) {
                Icon(
                    imageVector = if (isGenerating) Icons.Default.Stop else Icons.AutoMirrored.Filled.Send,
                    contentDescription = if (isGenerating) "Stop Generation" else "Send Message",
                    modifier = Modifier.size(20.dp)
                )
            }
        }
    }
}
