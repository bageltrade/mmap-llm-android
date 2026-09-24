package com.example.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val MmapDarkColorScheme = darkColorScheme(
    primary = CyberEmerald,
    onPrimary = Color(0xFF032215),
    primaryContainer = Color(0xFF064E3B),
    onPrimaryContainer = CyberEmeraldGlow,
    secondary = CyberCyan,
    onSecondary = Color(0xFF042B33),
    secondaryContainer = Color(0xFF164E63),
    onSecondaryContainer = CyberCyanGlow,
    tertiary = CyberPurple,
    background = DeepObsidian,
    onBackground = TextPrimary,
    surface = ObsidianSurface,
    onSurface = TextPrimary,
    surfaceVariant = ObsidianCard,
    onSurfaceVariant = TextSecondary,
    outline = ObsidianCardBorder,
    outlineVariant = CodeBorder
)

@Composable
fun MmapTheme(
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = MmapDarkColorScheme,
        typography = Typography,
        content = content
    )
}

@Composable
fun MyApplicationTheme(
    darkTheme: Boolean = true,
    dynamicColor: Boolean = false,
    content: @Composable () -> Unit
) {
    MmapTheme(content = content)
}
