package com.example.data

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(tableName = "chat_sessions")
data class ChatSessionEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val title: String,
    val modelName: String,
    val createdAt: Long = System.currentTimeMillis(),
    val lastUpdatedAt: Long = System.currentTimeMillis(),
    val totalContextTokens: Int = 0,
    val systemPrompt: String = "You are MmapLLM, a local zero-RAM inference engine powered by memory-mapped GGUF files."
)

@Entity(
    tableName = "chat_messages",
    foreignKeys = [
        ForeignKey(
            entity = ChatSessionEntity::class,
            parentColumns = ["id"],
            childColumns = ["sessionId"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [Index("sessionId")]
)
data class ChatMessageEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val sessionId: Long,
    val role: String, // "user", "assistant", "system"
    val content: String,
    val timestamp: Long = System.currentTimeMillis(),
    val tokensGenerated: Int = 0,
    val tokensPerSec: Double = 0.0,
    val peakRssMb: Double = 0.0,
    val timeToFirstTokenMs: Long = 0L,
    val mmapVirtualMb: Double = 0.0
)

@Entity(tableName = "custom_models")
data class CustomModelEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val name: String,
    val filePath: String,
    val fileSizeBytes: Long,
    val architecture: String,
    val quantType: String,
    val contextLimit: Int,
    val isDefault: Boolean = false,
    val addedAt: Long = System.currentTimeMillis()
)
