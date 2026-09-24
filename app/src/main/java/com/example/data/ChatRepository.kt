package com.example.data

import kotlinx.coroutines.flow.Flow

class ChatRepository(private val chatDao: ChatDao) {
    val allSessions: Flow<List<ChatSessionEntity>> = chatDao.getAllSessions()
    val allModels: Flow<List<CustomModelEntity>> = chatDao.getAllModels()

    fun getMessagesForSession(sessionId: Long): Flow<List<ChatMessageEntity>> =
        chatDao.getMessagesForSession(sessionId)

    suspend fun createSession(title: String, modelName: String): Long {
        val session = ChatSessionEntity(
            title = title,
            modelName = modelName
        )
        return chatDao.insertSession(session)
    }

    suspend fun getSessionById(sessionId: Long): ChatSessionEntity? =
        chatDao.getSessionById(sessionId)

    suspend fun updateSession(session: ChatSessionEntity) =
        chatDao.updateSession(session)

    suspend fun deleteSession(sessionId: Long) =
        chatDao.deleteSession(sessionId)

    suspend fun saveMessage(
        sessionId: Long,
        role: String,
        content: String,
        tokensGenerated: Int = 0,
        tokensPerSec: Double = 0.0,
        peakRssMb: Double = 0.0,
        timeToFirstTokenMs: Long = 0L,
        mmapVirtualMb: Double = 0.0
    ): Long {
        val msg = ChatMessageEntity(
            sessionId = sessionId,
            role = role,
            content = content,
            tokensGenerated = tokensGenerated,
            tokensPerSec = tokensPerSec,
            peakRssMb = peakRssMb,
            timeToFirstTokenMs = timeToFirstTokenMs,
            mmapVirtualMb = mmapVirtualMb
        )
        val id = chatDao.insertMessage(msg)
        val existingSession = chatDao.getSessionById(sessionId)
        if (existingSession != null) {
            chatDao.updateSession(
                existingSession.copy(
                    lastUpdatedAt = System.currentTimeMillis(),
                    totalContextTokens = existingSession.totalContextTokens + (content.length / 4)
                )
            )
        }
        return id
    }

    suspend fun clearSessionMessages(sessionId: Long) =
        chatDao.clearMessagesForSession(sessionId)

    suspend fun addCustomModel(
        name: String,
        filePath: String,
        fileSizeBytes: Long,
        architecture: String,
        quantType: String,
        contextLimit: Int
    ): Long {
        val model = CustomModelEntity(
            name = name,
            filePath = filePath,
            fileSizeBytes = fileSizeBytes,
            architecture = architecture,
            quantType = quantType,
            contextLimit = contextLimit
        )
        return chatDao.insertModel(model)
    }

    suspend fun removeModel(modelId: Long) =
        chatDao.deleteModel(modelId)
}
