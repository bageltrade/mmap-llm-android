package com.example.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Keyboard
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import com.example.ui.theme.CyberCyan
import com.example.ui.theme.CyberEmerald
import com.example.ui.theme.DeepObsidian
import com.example.ui.theme.ObsidianCard
import com.example.ui.theme.ObsidianCardBorder
import com.example.ui.theme.TextMonospace
import com.example.ui.theme.TextPrimary
import com.example.ui.theme.TextSecondary
import com.example.ui.theme.TextTertiary

@Composable
fun KeyboardShortcutsDialog(
    onDismiss: () -> Unit
) {
    Dialog(onDismissRequest = onDismiss) {
        Surface(
            modifier = Modifier
                .testTag("keyboard_shortcuts_dialog")
                .fillMaxWidth()
                .clip(RoundedCornerShape(16.dp))
                .border(1.dp, ObsidianCardBorder, RoundedCornerShape(16.dp)),
            color = ObsidianCard
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(20.dp)
            ) {
                // Title Row
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Default.Keyboard,
                            contentDescription = null,
                            tint = CyberEmerald,
                            modifier = Modifier.padding(end = 8.dp)
                        )
                        Text(
                            text = "Power User Shortcuts",
                            style = MaterialTheme.typography.titleMedium,
                            color = TextPrimary
                        )
                    }
                    IconButton(onClick = onDismiss, modifier = Modifier.testTag("close_shortcuts_button")) {
                        Icon(
                            imageVector = Icons.Default.Close,
                            contentDescription = "Close",
                            tint = TextSecondary
                        )
                    }
                }

                Text(
                    text = "Hardware keyboard shortcuts supported in DeX, Chromebooks, and Bluetooth keyboards:",
                    style = MaterialTheme.typography.bodyMedium,
                    color = TextSecondary,
                    modifier = Modifier.padding(top = 4.dp, bottom = 16.dp)
                )

                ShortcutRow(keys = listOf("Ctrl", "Enter"), action = "Send prompt / trigger generation")
                ShortcutRow(keys = listOf("Esc"), action = "Halt streaming token generation immediately")
                ShortcutRow(keys = listOf("Ctrl", "L"), action = "Clear session / start fresh context")
                ShortcutRow(keys = listOf("Ctrl", "M"), action = "Open GGUF Model Hub & File Loader")
                ShortcutRow(keys = listOf("Ctrl", "H"), action = "Toggle Telemetry & RAM HUD overlay")
                ShortcutRow(keys = listOf("Ctrl", "E"), action = "Force memory-mapped page purge & context eviction")
                ShortcutRow(keys = listOf("Ctrl", "K"), action = "Quick command palette / system prompt")

                Spacer(modifier = Modifier.height(16.dp))

                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp))
                        .background(DeepObsidian)
                        .padding(10.dp)
                ) {
                    Text(
                        text = "Tip: You can drag-and-drop .gguf files directly onto the app window to inspect and map tensors.",
                        style = MaterialTheme.typography.bodySmall,
                        color = CyberCyan
                    )
                }
            }
        }
    }
}

@Composable
private fun ShortcutRow(
    keys: List<String>,
    action: String
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 6.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = action,
            style = MaterialTheme.typography.bodyMedium,
            color = TextPrimary,
            modifier = Modifier.weight(1f)
        )

        Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
            keys.forEach { key ->
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(4.dp))
                        .background(DeepObsidian)
                        .border(1.dp, ObsidianCardBorder, RoundedCornerShape(4.dp))
                        .padding(horizontal = 6.dp, vertical = 2.dp)
                ) {
                    Text(
                        text = key,
                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                        color = TextMonospace,
                        fontFamily = FontFamily.Monospace
                    )
                }
            }
        }
    }
}
