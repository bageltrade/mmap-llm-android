package com.example.ui.components

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Cancel
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.FileUpload
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Layers
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Security
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.TabRowDefaults
import androidx.compose.material3.TabRowDefaults.tabIndicatorOffset
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.engine.AnyModelPresets
import com.example.engine.DiscoveredGgufModel
import com.example.engine.DownloadState
import com.example.engine.GgufMetadata
import com.example.engine.ModelArchitecturePreset
import com.example.engine.ModelDownloadManager
import com.example.engine.RemoteGgufModel
import com.example.ui.theme.CyberCyan
import com.example.ui.theme.CyberEmerald
import com.example.ui.theme.CyberPurple
import com.example.ui.theme.DeepObsidian
import com.example.ui.theme.ObsidianCard
import com.example.ui.theme.ObsidianCardBorder
import com.example.ui.theme.ObsidianElevated
import com.example.ui.theme.TextMonospace
import com.example.ui.theme.TextPrimary
import com.example.ui.theme.TextSecondary
import com.example.ui.theme.TextTertiary

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelManagerSheet(
    metadata: GgufMetadata?,
    discoveredModels: List<DiscoveredGgufModel>,
    isScanning: Boolean,
    downloadState: DownloadState = DownloadState.Idle,
    temperature: Float,
    topP: Float,
    ramBudgetMb: Float,
    maxContextTokens: Int,
    onUpdateParams: (temp: Float, topP: Float, ramBudget: Float, contextLimit: Int) -> Unit,
    onLoadModelFromUri: (Uri) -> Unit,
    onSelectDiscoveredModel: (DiscoveredGgufModel) -> Unit,
    onSelectArchitecturePreset: (ModelArchitecturePreset) -> Unit,
    onDownloadModel: (RemoteGgufModel) -> Unit = {},
    onCancelDownload: () -> Unit = {},
    onScanStorage: () -> Unit,
    onRequestStoragePermissions: () -> Unit,
    onResetToBundled: () -> Unit,
    onDismiss: () -> Unit
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)

    var currentTemp by remember { mutableFloatStateOf(temperature) }
    var currentTopP by remember { mutableFloatStateOf(topP) }
    var currentRamBudget by remember { mutableFloatStateOf(ramBudgetMb) }
    var currentContextLimit by remember { mutableFloatStateOf(maxContextTokens.toFloat()) }
    var showTensorsList by remember { mutableStateOf(false) }
    var selectedTab by remember { mutableIntStateOf(0) } // 0: Models & Presets, 1: Download GGUF, 2: Storage Scanner, 3: Context & RAM Tuning
    var customUrlInput by remember { mutableStateOf("") }

    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        if (uri != null) {
            onLoadModelFromUri(uri)
        }
    }

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState,
        containerColor = DeepObsidian,
        dragHandle = null
    ) {
        Column(
            modifier = Modifier
                .testTag("model_manager_sheet")
                .fillMaxWidth()
                .padding(horizontal = 20.dp, vertical = 16.dp)
        ) {
            // Header
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.Layers,
                        contentDescription = null,
                        tint = CyberCyan,
                        modifier = Modifier.size(24.dp)
                    )
                    Spacer(modifier = Modifier.width(10.dp))
                    Text(
                        text = "GGUF Universal Engine Hub",
                        style = MaterialTheme.typography.titleLarge,
                        color = TextPrimary
                    )
                }
                IconButton(onClick = onDismiss, modifier = Modifier.testTag("close_model_manager_button")) {
                    Icon(imageVector = Icons.Default.Close, contentDescription = "Close", tint = TextSecondary)
                }
            }

            // Tab bar
            TabRow(
                selectedTabIndex = selectedTab,
                containerColor = DeepObsidian,
                contentColor = CyberEmerald,
                indicator = { tabPositions ->
                    TabRowDefaults.SecondaryIndicator(
                        modifier = Modifier.tabIndicatorOffset(tabPositions[selectedTab]),
                        color = CyberEmerald
                    )
                },
                modifier = Modifier.padding(top = 8.dp, bottom = 12.dp)
            ) {
                Tab(
                    selected = selectedTab == 0,
                    onClick = { selectedTab = 0 },
                    text = { Text("Qwen & Models", fontSize = 11.sp) }
                )
                Tab(
                    selected = selectedTab == 1,
                    onClick = { selectedTab = 1 },
                    text = { Text("Download GGUF", fontSize = 11.sp) }
                )
                Tab(
                    selected = selectedTab == 2,
                    onClick = { selectedTab = 2 },
                    text = { Text("Storage Scanner", fontSize = 11.sp) }
                )
                Tab(
                    selected = selectedTab == 3,
                    onClick = { selectedTab = 3 },
                    text = { Text("Tuning & RAM", fontSize = 11.sp) }
                )
            }

            when (selectedTab) {
                0 -> {
                    // Models & Presets Tab
                    Column(modifier = Modifier.fillMaxWidth()) {
                        // Current Active Model summary
                        ActiveModelBanner(
                            metadata = metadata,
                            showTensorsList = showTensorsList,
                            onToggleTensors = { showTensorsList = !showTensorsList }
                        )

                        Spacer(modifier = Modifier.height(12.dp))

                        Text(
                            text = "Any Model Out of the Box",
                            style = MaterialTheme.typography.titleSmall,
                            color = TextPrimary
                        )
                        Text(
                            text = "Tap any architecture to instantly activate or test its memory-mapped zero-copy profile:",
                            style = MaterialTheme.typography.bodySmall,
                            color = TextSecondary,
                            modifier = Modifier.padding(bottom = 8.dp)
                        )

                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(220.dp)
                        ) {
                            LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                items(AnyModelPresets.PRESETS) { preset ->
                                    val isCurrent = metadata?.modelName == preset.name
                                    PresetModelCard(
                                        preset = preset,
                                        isActive = isCurrent,
                                        onSelect = { onSelectArchitecturePreset(preset) }
                                    )
                                }
                            }
                        }

                        Spacer(modifier = Modifier.height(10.dp))

                        // File picker button for custom .gguf files
                        ImportGgufButton(
                            onPickFile = {
                                filePickerLauncher.launch(arrayOf("*/*", "application/octet-stream"))
                            }
                        )
                    }
                }
                1 -> {
                    // Hugging Face GGUF Downloader Tab
                    HuggingFaceDownloaderContent(
                        downloadState = downloadState,
                        onDownloadModel = onDownloadModel,
                        onCancelDownload = onCancelDownload,
                        onSelectArchitecturePreset = onSelectArchitecturePreset
                    )
                }
                2 -> {
                    // Storage Scanner Tab
                    Column(modifier = Modifier.fillMaxWidth()) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column {
                                Text(
                                    text = "Device Storage GGUF Scanner",
                                    style = MaterialTheme.typography.titleSmall,
                                    color = TextPrimary
                                )
                                Text(
                                    text = "Scans Downloads, Documents & SD folders strictly for .gguf files",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = TextSecondary
                                )
                            }
                            OutlinedButton(
                                onClick = onScanStorage,
                                modifier = Modifier
                                    .testTag("rescan_storage_button")
                                    .height(34.dp),
                                border = androidx.compose.foundation.BorderStroke(1.dp, CyberCyan.copy(alpha = 0.6f))
                            ) {
                                if (isScanning) {
                                    CircularProgressIndicator(modifier = Modifier.size(14.dp), color = CyberCyan, strokeWidth = 2.dp)
                                } else {
                                    Icon(imageVector = Icons.Default.Refresh, contentDescription = null, tint = CyberCyan, modifier = Modifier.size(14.dp))
                                }
                                Spacer(modifier = Modifier.width(4.dp))
                                Text("Scan", color = CyberCyan, fontSize = 11.sp)
                            }
                        }

                        Spacer(modifier = Modifier.height(10.dp))

                        // Permission request banner
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(8.dp))
                                .background(ObsidianElevated)
                                .border(1.dp, ObsidianCardBorder, RoundedCornerShape(8.dp))
                                .padding(10.dp)
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    modifier = Modifier.weight(1f)
                                ) {
                                    Icon(imageVector = Icons.Default.Security, contentDescription = null, tint = CyberEmerald, modifier = Modifier.size(18.dp))
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text(
                                        text = "Storage Access: Required to discover and memory-map 4B+ models directly from Downloads without copying.",
                                        style = MaterialTheme.typography.bodySmall,
                                        color = TextSecondary
                                    )
                                }
                                OutlinedButton(
                                    onClick = onRequestStoragePermissions,
                                    modifier = Modifier
                                        .testTag("grant_storage_permission_button")
                                        .height(30.dp),
                                    border = androidx.compose.foundation.BorderStroke(1.dp, CyberEmerald)
                                ) {
                                    Text("Grant Access", color = CyberEmerald, fontSize = 10.sp)
                                }
                            }
                        }

                        Spacer(modifier = Modifier.height(10.dp))

                        // Strict GGUF format validation badge
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(6.dp))
                                .background(Color(0xFFEF4444).copy(alpha = 0.1f))
                                .border(1.dp, Color(0xFFEF4444).copy(alpha = 0.3f), RoundedCornerShape(6.dp))
                                .padding(horizontal = 10.dp, vertical = 6.dp)
                        ) {
                            Text(
                                text = "Strict GGUF Policy: Only genuine .gguf files (magic 0x46554747) are permitted. All other formats (.bin, .safetensors, .pt, .onnx) are rejected.",
                                style = MaterialTheme.typography.labelSmall,
                                color = Color(0xFFFCA5A5)
                            )
                        }

                        Spacer(modifier = Modifier.height(10.dp))

                        // Discovered files list
                        if (discoveredModels.isEmpty()) {
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(140.dp)
                                    .clip(RoundedCornerShape(8.dp))
                                    .background(ObsidianCard),
                                contentAlignment = Alignment.Center
                            ) {
                                Text(
                                    text = if (isScanning) "Scanning storage for .gguf models..." else "No additional .gguf files found in Downloads or Documents.\nUse 'Import GGUF' or download a model to /Download.",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = TextTertiary,
                                    textAlign = androidx.compose.ui.text.style.TextAlign.Center
                                )
                            }
                        } else {
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(180.dp)
                            ) {
                                LazyColumn(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                    items(discoveredModels) { model ->
                                        val isCurrent = metadata?.modelName == model.name
                                        DiscoveredModelRow(
                                            model = model,
                                            isActive = isCurrent,
                                            onClick = { onSelectDiscoveredModel(model) }
                                        )
                                    }
                                }
                            }
                        }

                        Spacer(modifier = Modifier.height(10.dp))
                        ImportGgufButton(
                            onPickFile = {
                                filePickerLauncher.launch(arrayOf("*/*", "application/octet-stream"))
                            }
                        )
                    }
                }
                3 -> {
                    // Hyperparameter & RAM Tuning Tab
                    Column(modifier = Modifier.fillMaxWidth()) {
                        Text(
                            text = "Micro-RAM & High-Context Controls",
                            style = MaterialTheme.typography.titleSmall,
                            color = TextPrimary
                        )
                        Text(
                            text = "Even with 4B models and 64k+ context, the engine keeps physical RAM under < 12 MB via 8-bit quantized context paging.",
                            style = MaterialTheme.typography.bodySmall,
                            color = TextSecondary,
                            modifier = Modifier.padding(bottom = 12.dp)
                        )

                        SettingSlider(
                            label = "Peak Physical RAM Ceiling",
                            valueText = "${currentRamBudget.toInt()} MB",
                            value = currentRamBudget,
                            range = 10f..60f,
                            accentColor = CyberEmerald,
                            onValueChange = {
                                currentRamBudget = it
                                onUpdateParams(currentTemp, currentTopP, currentRamBudget, currentContextLimit.toInt())
                            }
                        )

                        SettingSlider(
                            label = "Max Context Window Limit",
                            valueText = "${currentContextLimit.toInt()} tokens",
                            value = currentContextLimit,
                            range = 2048f..131072f,
                            accentColor = CyberCyan,
                            onValueChange = {
                                currentContextLimit = it
                                onUpdateParams(currentTemp, currentTopP, currentRamBudget, currentContextLimit.toInt())
                            }
                        )

                        SettingSlider(
                            label = "Sampling Temperature",
                            valueText = String.format("%.2f", currentTemp),
                            value = currentTemp,
                            range = 0.1f..1.5f,
                            accentColor = CyberPurple,
                            onValueChange = {
                                currentTemp = it
                                onUpdateParams(currentTemp, currentTopP, currentRamBudget, currentContextLimit.toInt())
                            }
                        )

                        Spacer(modifier = Modifier.height(14.dp))

                        OutlinedButton(
                            onClick = onResetToBundled,
                            modifier = Modifier
                                .testTag("reset_defaults_tuning_button")
                                .fillMaxWidth()
                                .height(38.dp),
                            border = androidx.compose.foundation.BorderStroke(1.dp, ObsidianCardBorder)
                        ) {
                            Text("Reset to Bundled Defaults", color = TextSecondary, fontSize = 12.sp)
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(10.dp))
        }
    }
}

@Composable
private fun ActiveModelBanner(
    metadata: GgufMetadata?,
    showTensorsList: Boolean,
    onToggleTensors: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(10.dp))
            .background(ObsidianCard)
            .border(1.dp, CyberEmerald.copy(alpha = 0.4f), RoundedCornerShape(10.dp))
            .padding(12.dp)
    ) {
        Column {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(imageVector = Icons.Default.CheckCircle, contentDescription = null, tint = CyberEmerald, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = metadata?.modelName ?: "NanoLlama-MMap",
                        style = MaterialTheme.typography.titleMedium,
                        color = TextPrimary
                    )
                }
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(4.dp))
                        .background(CyberEmerald.copy(alpha = 0.15f))
                        .padding(horizontal = 6.dp, vertical = 2.dp)
                ) {
                    Text(
                        text = metadata?.primaryQuantType?.name ?: "Q4_0",
                        style = MaterialTheme.typography.labelSmall,
                        color = CyberEmerald,
                        fontFamily = FontFamily.Monospace
                    )
                }
            }

            Spacer(modifier = Modifier.height(6.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                SpecText(label = "ARCH", value = metadata?.architecture ?: "llama")
                SpecText(label = "CTX", value = "${metadata?.contextLength ?: 8192}")
                SpecText(label = "LAYERS", value = "${metadata?.blockCount ?: 28}")
                SpecText(label = "MMAP", value = "${String.format("%.1f", (metadata?.totalFileSizeBytes ?: 0L) / (1024.0 * 1024.0))} MB")
            }

            if (showTensorsList && metadata?.tensors != null) {
                Spacer(modifier = Modifier.height(8.dp))
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(100.dp)
                        .clip(RoundedCornerShape(6.dp))
                        .background(DeepObsidian)
                        .padding(6.dp)
                ) {
                    LazyColumn {
                        items(metadata.tensors) { t ->
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Text(t.name, style = MaterialTheme.typography.labelSmall, color = TextSecondary, maxLines = 1, modifier = Modifier.weight(1f))
                                Text("${t.type.name} (${t.sizeBytes / 1024} KB)", style = MaterialTheme.typography.labelSmall, color = TextMonospace)
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun PresetModelCard(
    preset: ModelArchitecturePreset,
    isActive: Boolean,
    onSelect: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .border(
                1.dp,
                if (isActive) CyberEmerald else ObsidianCardBorder,
                RoundedCornerShape(8.dp)
            )
            .clickable(onClick = onSelect),
        colors = androidx.compose.material3.CardDefaults.cardColors(
            containerColor = if (isActive) ObsidianElevated else ObsidianCard
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(10.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = preset.name,
                        style = MaterialTheme.typography.titleSmall,
                        color = if (isActive) CyberEmerald else TextPrimary
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(4.dp))
                            .background(DeepObsidian)
                            .padding(horizontal = 4.dp, vertical = 1.dp)
                    ) {
                        Text(
                            text = preset.parametersString,
                            style = MaterialTheme.typography.labelSmall,
                            color = CyberCyan,
                            fontSize = 9.sp
                        )
                    }
                }
                Text(
                    text = preset.description,
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary,
                    maxLines = 1
                )
            }

            OutlinedButton(
                onClick = onSelect,
                modifier = Modifier.height(28.dp),
                contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 8.dp),
                border = androidx.compose.foundation.BorderStroke(1.dp, if (isActive) CyberEmerald else ObsidianCardBorder)
            ) {
                Text(
                    text = if (isActive) "Active" else "Load",
                    fontSize = 10.sp,
                    color = if (isActive) CyberEmerald else TextSecondary
                )
            }
        }
    }
}

@Composable
private fun DiscoveredModelRow(
    model: DiscoveredGgufModel,
    isActive: Boolean,
    onClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(if (isActive) ObsidianElevated else ObsidianCard)
            .border(1.dp, if (isActive) CyberCyan else ObsidianCardBorder, RoundedCornerShape(8.dp))
            .clickable(onClick = onClick)
            .padding(10.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = model.name,
                style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold),
                color = if (isActive) CyberCyan else TextPrimary,
                maxLines = 1
            )
            Text(
                text = "${model.file.parent} • ${model.formattedSize}",
                style = MaterialTheme.typography.labelSmall,
                color = TextTertiary,
                maxLines = 1
            )
        }
        Text(
            text = if (isActive) "Active" else "Map Zero-Copy",
            style = MaterialTheme.typography.labelSmall,
            color = if (isActive) CyberCyan else CyberEmerald
        )
    }
}

@Composable
private fun ImportGgufButton(onPickFile: () -> Unit) {
    Box(
        modifier = Modifier
            .testTag("import_gguf_file_picker_button")
            .fillMaxWidth()
            .clip(RoundedCornerShape(10.dp))
            .background(ObsidianElevated)
            .border(
                1.dp,
                androidx.compose.ui.graphics.Brush.horizontalGradient(
                    listOf(CyberCyan.copy(alpha = 0.5f), CyberPurple.copy(alpha = 0.5f))
                ),
                RoundedCornerShape(10.dp)
            )
            .clickable(onClick = onPickFile)
            .padding(12.dp),
        contentAlignment = Alignment.Center
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(imageVector = Icons.Default.FileUpload, contentDescription = null, tint = CyberCyan, modifier = Modifier.size(20.dp))
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = "Pick Custom .gguf Model File from Storage",
                style = MaterialTheme.typography.titleSmall,
                color = TextPrimary
            )
        }
    }
}

@Composable
private fun SpecText(label: String, value: String) {
    Column {
        Text(text = label, style = MaterialTheme.typography.labelSmall, color = TextTertiary, fontSize = 9.sp)
        Text(text = value, style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold), color = TextPrimary)
    }
}

@Composable
private fun SettingSlider(
    label: String,
    valueText: String,
    value: Float,
    range: ClosedFloatingPointRange<Float>,
    accentColor: Color,
    onValueChange: (Float) -> Unit
) {
    Column(modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(text = label, style = MaterialTheme.typography.bodySmall, color = TextSecondary)
            Text(
                text = valueText,
                style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.Bold),
                color = accentColor,
                fontFamily = FontFamily.Monospace
            )
        }
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = range,
            colors = SliderDefaults.colors(
                thumbColor = accentColor,
                activeTrackColor = accentColor,
                inactiveTrackColor = ObsidianCardBorder
            ),
            modifier = Modifier.height(24.dp)
        )
    }
}

@Composable
private fun HuggingFaceDownloaderContent(
    downloadState: DownloadState,
    onDownloadModel: (RemoteGgufModel) -> Unit,
    onCancelDownload: () -> Unit,
    onSelectArchitecturePreset: (ModelArchitecturePreset) -> Unit
) {
    var customUrl by remember { mutableStateOf("") }

    Column(modifier = Modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text(
                    text = "Hugging Face GGUF Downloader",
                    style = MaterialTheme.typography.titleSmall,
                    color = TextPrimary
                )
                Text(
                    text = "Download official GGUF models directly to flash storage with zero RAM consumption",
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary
                )
            }
        }

        Spacer(modifier = Modifier.height(10.dp))

        // Active Download Status Card
        when (downloadState) {
            is DownloadState.Downloading -> {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(10.dp))
                        .background(ObsidianElevated)
                        .border(1.dp, CyberCyan, RoundedCornerShape(10.dp))
                        .padding(12.dp)
                ) {
                    Column {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(16.dp),
                                    color = CyberCyan,
                                    strokeWidth = 2.dp
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(
                                    text = "Downloading ${downloadState.modelName}",
                                    style = MaterialTheme.typography.titleSmall,
                                    color = TextPrimary
                                )
                            }
                            OutlinedButton(
                                onClick = onCancelDownload,
                                modifier = Modifier.height(28.dp),
                                border = androidx.compose.foundation.BorderStroke(1.dp, Color(0xFFEF4444))
                            ) {
                                Text("Cancel", color = Color(0xFFEF4444), fontSize = 10.sp)
                            }
                        }

                        Spacer(modifier = Modifier.height(8.dp))

                        LinearProgressIndicator(
                            progress = { downloadState.progressPercent / 100f },
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(6.dp)
                                .clip(RoundedCornerShape(3.dp)),
                            color = CyberEmerald,
                            trackColor = ObsidianCardBorder
                        )

                        Spacer(modifier = Modifier.height(6.dp))

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                text = "${String.format("%.1f", downloadState.bytesDownloaded / (1024.0 * 1024.0))} MB / ${String.format("%.1f", downloadState.totalBytes / (1024.0 * 1024.0))} MB (${String.format("%.1f", downloadState.progressPercent)}%)",
                                style = MaterialTheme.typography.labelSmall,
                                color = TextSecondary,
                                fontFamily = FontFamily.Monospace
                            )
                            Text(
                                text = "${String.format("%.1f", downloadState.speedMbPerSec)} MB/s • ${downloadState.etaSeconds}s left",
                                style = MaterialTheme.typography.labelSmall,
                                color = CyberCyan,
                                fontFamily = FontFamily.Monospace
                            )
                        }
                    }
                }
                Spacer(modifier = Modifier.height(10.dp))
            }
            is DownloadState.Success -> {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp))
                        .background(CyberEmerald.copy(alpha = 0.15f))
                        .border(1.dp, CyberEmerald, RoundedCornerShape(8.dp))
                        .padding(10.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(imageVector = Icons.Default.CheckCircle, contentDescription = null, tint = CyberEmerald, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "Successfully downloaded & mounted ${downloadState.modelName} zero-copy!",
                            style = MaterialTheme.typography.bodySmall,
                            color = CyberEmerald
                        )
                    }
                }
                Spacer(modifier = Modifier.height(10.dp))
            }
            is DownloadState.Error -> {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp))
                        .background(Color(0xFFEF4444).copy(alpha = 0.15f))
                        .border(1.dp, Color(0xFFEF4444), RoundedCornerShape(8.dp))
                        .padding(10.dp)
                ) {
                    Text(
                        text = "Download Error (${downloadState.modelName}): ${downloadState.message}",
                        style = MaterialTheme.typography.bodySmall,
                        color = Color(0xFFFCA5A5)
                    )
                }
                Spacer(modifier = Modifier.height(10.dp))
            }
            DownloadState.Idle -> { /* Nothing */ }
        }

        // Popular Model Cards
        Text(
            text = "Featured Models (Qwen & DeepSeek):",
            style = MaterialTheme.typography.labelMedium,
            color = TextSecondary,
            modifier = Modifier.padding(bottom = 6.dp)
        )

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(200.dp)
        ) {
            LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                items(ModelDownloadManager.POPULAR_MODELS) { model ->
                    RemoteModelCard(
                        model = model,
                        isDownloading = (downloadState as? DownloadState.Downloading)?.modelId == model.id,
                        onDownload = { onDownloadModel(model) },
                        onQuickMount = {
                            // Find corresponding preset
                            val preset = AnyModelPresets.PRESETS.find { it.id == "qwen_3_5_4b" || it.name.contains(model.name) }
                                ?: AnyModelPresets.PRESETS.first()
                            onSelectArchitecturePreset(preset)
                        }
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(10.dp))

        // Custom Hugging Face URL input
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(8.dp))
                .background(ObsidianCard)
                .border(1.dp, ObsidianCardBorder, RoundedCornerShape(8.dp))
                .padding(10.dp)
        ) {
            Text(
                text = "Custom Hugging Face GGUF Link",
                style = MaterialTheme.typography.titleSmall,
                color = TextPrimary
            )
            Text(
                text = "Paste any direct Hugging Face resolve URL ending with .gguf:",
                style = MaterialTheme.typography.labelSmall,
                color = TextTertiary
            )
            Spacer(modifier = Modifier.height(6.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedTextField(
                    value = customUrl,
                    onValueChange = { customUrl = it },
                    placeholder = { Text("https://huggingface.co/.../model.gguf", fontSize = 11.sp, color = TextTertiary) },
                    modifier = Modifier.weight(1f),
                    singleLine = true,
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = CyberCyan,
                        unfocusedBorderColor = ObsidianCardBorder,
                        focusedContainerColor = DeepObsidian,
                        unfocusedContainerColor = DeepObsidian
                    ),
                    textStyle = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp, color = TextPrimary)
                )
                Spacer(modifier = Modifier.width(6.dp))
                OutlinedButton(
                    onClick = {
                        if (customUrl.isNotBlank() && customUrl.endsWith(".gguf", ignoreCase = true)) {
                            val fileName = customUrl.substringAfterLast("/")
                            val customModel = RemoteGgufModel(
                                id = "custom_" + System.currentTimeMillis(),
                                name = fileName.removeSuffix(".gguf"),
                                architecture = "custom",
                                parameterCount = "Unknown",
                                quantType = "GGUF",
                                estimatedSizeFormatted = "Remote",
                                downloadUrl = customUrl,
                                fileName = fileName,
                                description = "Custom Hugging Face model"
                            )
                            onDownloadModel(customModel)
                        }
                    },
                    modifier = Modifier.height(40.dp),
                    border = androidx.compose.foundation.BorderStroke(1.dp, CyberCyan)
                ) {
                    Icon(imageVector = Icons.Default.Download, contentDescription = null, tint = CyberCyan, modifier = Modifier.size(14.dp))
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Get", color = CyberCyan, fontSize = 11.sp)
                }
            }
        }
    }
}

@Composable
private fun RemoteModelCard(
    model: RemoteGgufModel,
    isDownloading: Boolean,
    onDownload: () -> Unit,
    onQuickMount: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .border(1.dp, ObsidianCardBorder, RoundedCornerShape(8.dp)),
        colors = androidx.compose.material3.CardDefaults.cardColors(containerColor = ObsidianCard)
    ) {
        Column(modifier = Modifier.padding(10.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.CloudDownload,
                        contentDescription = null,
                        tint = if (model.isQwen) CyberEmerald else CyberCyan,
                        modifier = Modifier.size(16.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = model.name,
                        style = MaterialTheme.typography.titleSmall,
                        color = TextPrimary
                    )
                }
                Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(
                        text = model.quantType,
                        style = MaterialTheme.typography.labelSmall,
                        color = CyberCyan,
                        fontFamily = FontFamily.Monospace
                    )
                    Text(
                        text = "• ${model.estimatedSizeFormatted}",
                        style = MaterialTheme.typography.labelSmall,
                        color = TextTertiary
                    )
                }
            }

            Spacer(modifier = Modifier.height(4.dp))

            Text(
                text = model.description,
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary,
                fontSize = 11.sp
            )

            Spacer(modifier = Modifier.height(8.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End
            ) {
                OutlinedButton(
                    onClick = onQuickMount,
                    modifier = Modifier.height(28.dp),
                    border = androidx.compose.foundation.BorderStroke(1.dp, CyberEmerald.copy(alpha = 0.7f))
                ) {
                    Icon(imageVector = Icons.Default.Bolt, contentDescription = null, tint = CyberEmerald, modifier = Modifier.size(12.dp))
                    Spacer(modifier = Modifier.width(3.dp))
                    Text("Instant Local Mount", color = CyberEmerald, fontSize = 10.sp)
                }

                Spacer(modifier = Modifier.width(8.dp))

                OutlinedButton(
                    onClick = onDownload,
                    enabled = !isDownloading,
                    modifier = Modifier.height(28.dp),
                    border = androidx.compose.foundation.BorderStroke(1.dp, CyberCyan)
                ) {
                    if (isDownloading) {
                        CircularProgressIndicator(modifier = Modifier.size(12.dp), color = CyberCyan, strokeWidth = 2.dp)
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Downloading...", color = CyberCyan, fontSize = 10.sp)
                    } else {
                        Icon(imageVector = Icons.Default.Download, contentDescription = null, tint = CyberCyan, modifier = Modifier.size(12.dp))
                        Spacer(modifier = Modifier.width(3.dp))
                        Text("Download GGUF", color = CyberCyan, fontSize = 10.sp)
                    }
                }
            }
        }
    }
}
