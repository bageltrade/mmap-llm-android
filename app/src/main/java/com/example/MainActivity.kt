package com.example

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import com.example.ui.ChatViewModel
import com.example.ui.MainScreen
import com.example.ui.theme.DeepObsidian
import com.example.ui.theme.MmapTheme

class MainActivity : ComponentActivity() {
    private val viewModel: ChatViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        handleIntentData(intent)

        setContent {
            MmapTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = DeepObsidian
                ) {
                    MainScreen(viewModel = viewModel)
                }
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleIntentData(intent)
    }

    private fun handleIntentData(intent: Intent?) {
        if (intent == null) return
        val uri: Uri? = intent.data ?: intent.clipData?.getItemAt(0)?.uri
        if (uri != null) {
            viewModel.loadModelFromUri(uri)
        }
    }
}
