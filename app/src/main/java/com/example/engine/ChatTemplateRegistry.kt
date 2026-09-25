package com.example.engine

/**
 * Chat template registry modelled on llama.cpp `--chat-template` + Ollama Modelfile TEMPLATE.
 *
 * llama.cpp resolves a Jinja template from GGUF metadata (`tokenizer.chat_template`,
 * falling back to a built-in per architecture: chatml, llama3, gemma, deepseek, qwen, phi4,
 * mistral, ...). Ollama does the same via Modelfile SYSTEM / TEMPLATE blocks and the
 * `/api/chat` messages[] -> prompt rendering step before the runner is called.
 *
 * This class provides the same prompt-rendering layer in pure Kotlin so an imported
 * GGUF chats correctly even before native (NDK) inference is wired in:
 *  - auto-detect template from architecture string + GGUF rawMetadata
 *  - render system + multi-turn history + current user turn
 *  - optional prefill of a partial assistant message (like llama-server
 *    `--prefill-assistant` / Claude-style prefill)
 */
object ChatTemplateRegistry {

    enum class TemplateId {
        CHATML, LLAMA3, LLAMA2, GEMMA, DEEPSEEK_R1, QWEN3, PHI4, MISTRAL, FALCON, GENERIC
    }

    fun detectTemplate(meta: GgufMetadata?): TemplateId {
        // 1. Explicit GGUF chat template, if the importer preserved it.
        val raw = meta?.rawMetadata?.get("tokenizer.chat_template")?.toString()?.lowercase() ?: ""
        if (raw.isNotEmpty()) {
            when {
                "deepseek" in raw || "<think>" in raw -> return TemplateId.DEEPSEEK_R1
                "llama-3" in raw || "llama3" in raw || "<|start_header_id|>" in raw -> return TemplateId.LLAMA3
                "gemma" in raw || "<start_of_turn>" in raw -> return TemplateId.GEMMA
                "mistral" in raw || "[inst]" in raw -> return TemplateId.MISTRAL
                "phi" in raw -> return TemplateId.PHI4
                "im_start" in raw || "chatml" in raw -> return TemplateId.CHATML
            }
        }
        // 2. Architecture heuristic (mirrors llama.cpp built-in template list).
        val arch = (meta?.architecture ?: "").lowercase()
        val name = (meta?.modelName ?: "").lowercase()
        val hay = "$arch $name $raw"
        return when {
            "deepseek" in hay || "r1" in hay && "distill" in hay -> TemplateId.DEEPSEEK_R1
            "gemma" in hay -> TemplateId.GEMMA
            "llama" in hay -> if ("llama-2" in hay || "llamafile" in hay) TemplateId.LLAMA2 else TemplateId.LLAMA3
            "mistral" in hay || "mixtral" in hay -> TemplateId.MISTRAL
            "phi" in hay -> TemplateId.PHI4
            "falcon" in hay -> TemplateId.FALCON
            "qwen3" in hay || "qwen" in hay -> TemplateId.CHATML // Qwen family uses ChatML
            else -> TemplateId.CHATML
        }
    }

    fun render(
        systemPrompt: String,
        history: List<ChatTurn>,
        currentUserPrompt: String,
        meta: GgufMetadata?,
        prefillAssistant: String? = null
    ): String {
        return when (detectTemplate(meta)) {
            TemplateId.LLAMA3 -> renderLlama3(systemPrompt, history, currentUserPrompt, prefillAssistant)
            TemplateId.LLAMA2 -> renderLlama2(systemPrompt, history, currentUserPrompt)
            TemplateId.GEMMA -> renderGemma(systemPrompt, history, currentUserPrompt)
            TemplateId.DEEPSEEK_R1 -> renderDeepSeekR1(systemPrompt, history, currentUserPrompt, prefillAssistant)
            TemplateId.MISTRAL -> renderMistral(systemPrompt, history, currentUserPrompt)
            TemplateId.PHI4 -> renderPhi(systemPrompt, history, currentUserPrompt)
            TemplateId.FALCON -> renderFalcon(systemPrompt, history, currentUserPrompt)
            TemplateId.QWEN3 -> renderChatMl(systemPrompt, history, currentUserPrompt, prefillAssistant)
            TemplateId.CHATML, TemplateId.GENERIC -> renderChatMl(systemPrompt, history, currentUserPrompt, prefillAssistant)
        }
    }

    fun renderChatMl(
        systemPrompt: String,
        history: List<ChatTurn>,
        currentUserPrompt: String,
        prefillAssistant: String? = null
    ): String {
        val sb = StringBuilder()
        if (systemPrompt.isNotBlank()) {
            sb.append("<|im_start|>system\n").append(systemPrompt.trim()).append("<|im_end|>\n")
        }
        for (turn in history) {
            val role = when (turn.role.lowercase()) {
                "human" -> "user"
                "ai", "model" -> "assistant"
                else -> turn.role.lowercase()
            }
            sb.append("<|im_start|>").append(role).append("\n")
                .append(turn.content.trim()).append("<|im_end|>\n")
        }
        sb.append("<|im_start|>user\n").append(currentUserPrompt.trim()).append("<|im_end|>\n")
        sb.append("<|im_start|>assistant\n")
        if (!prefillAssistant.isNullOrBlank()) sb.append(prefillAssistant)
        return sb.toString()
    }

    private fun renderLlama3(
        systemPrompt: String,
        history: List<ChatTurn>,
        currentUserPrompt: String,
        prefillAssistant: String?
    ): String {
        val sb = StringBuilder()
        sb.append("<|begin_of_text|>")
        if (systemPrompt.isNotBlank()) {
            sb.append("<|start_header_id|>system<|end_header_id|>\n\n")
                .append(systemPrompt.trim()).append("<|eot_id|>")
        }
        for (turn in history) {
            val role = if (turn.role.lowercase() == "user" || turn.role.lowercase() == "human") "user" else "assistant"
            sb.append("<|start_header_id|>").append(role).append("<|end_header_id|>\n\n")
                .append(turn.content.trim()).append("<|eot_id|>")
        }
        sb.append("<|start_header_id|>user<|end_header_id|>\n\n")
            .append(currentUserPrompt.trim()).append("<|eot_id|>")
        sb.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        if (!prefillAssistant.isNullOrBlank()) sb.append(prefillAssistant)
        return sb.toString()
    }

    private fun renderLlama2(systemPrompt: String, history: List<ChatTurn>, currentUserPrompt: String): String {
        val sb = StringBuilder()
        sb.append("[INST] ")
        if (systemPrompt.isNotBlank()) {
            sb.append("<<SYS>>\n").append(systemPrompt.trim()).append("\n<</SYS>>\n\n")
        }
        for (turn in history) {
            if (turn.role.lowercase() == "user" || turn.role.lowercase() == "human") {
                sb.append(turn.content.trim()).append(" [/INST] ")
            } else {
                sb.append(turn.content.trim()).append(" </s><s>[INST] ")
            }
        }
        sb.append(currentUserPrompt.trim()).append(" [/INST]")
        return sb.toString()
    }

    private fun renderGemma(systemPrompt: String, history: List<ChatTurn>, currentUserPrompt: String): String {
        // Gemma has no system role: fold system into first user turn (like Ollama does).
        val sb = StringBuilder()
        val foldedFirst = if (systemPrompt.isNotBlank()) "${systemPrompt.trim()}\n\n$currentUserPrompt" else currentUserPrompt
        var firstUserDone = false
        for (turn in history) {
            if (turn.role.lowercase() == "user" || turn.role.lowercase() == "human") {
                val content = if (!firstUserDone && systemPrompt.isNotBlank()) {
                    firstUserDone = true
                    "${systemPrompt.trim()}\n\n${turn.content.trim()}"
                } else turn.content.trim()
                sb.append("<start_of_turn>user\n").append(content).append("<end_of_turn>\n")
            } else {
                sb.append("<start_of_turn>model\n").append(turn.content.trim()).append("<end_of_turn>\n")
            }
        }
        if (history.none { it.role.lowercase() == "user" || it.role.lowercase() == "human" }) {
            sb.append("<start_of_turn>user\n").append(foldedFirst.trim()).append("<end_of_turn>\n")
        } else {
            sb.append("<start_of_turn>user\n").append(currentUserPrompt.trim()).append("<end_of_turn>\n")
        }
        sb.append("<start_of_turn>model\n")
        return sb.toString()
    }

    private fun renderDeepSeekR1(
        systemPrompt: String,
        history: List<ChatTurn>,
        currentUserPrompt: String,
        prefillAssistant: String?
    ): String {
        // DeepSeek-R1: ChatML + <think> reasoning block support (llama-server --reasoning-format deepseek).
        val sb = StringBuilder()
        val sys = if (systemPrompt.isBlank()) "You are a helpful AI assistant. Think step by step inside <think> tags, then give the final answer." else systemPrompt.trim()
        sb.append("<｜begin▁of▁sentence｜>").append(sys).append("\n")
        for (turn in history) {
            if (turn.role.lowercase() == "user" || turn.role.lowercase() == "human") {
                sb.append("Human: ").append(turn.content.trim()).append("\n")
            } else {
                sb.append("Assistant: ").append(turn.content.trim()).append("\n")
            }
        }
        sb.append("Human: ").append(currentUserPrompt.trim()).append("\nAssistant: ")
        if (!prefillAssistant.isNullOrBlank()) sb.append(prefillAssistant) else sb.append("<think>")
        return sb.toString()
    }

    private fun renderMistral(systemPrompt: String, history: List<ChatTurn>, currentUserPrompt: String): String {
        val sb = StringBuilder()
        sb.append("<s>")
        if (systemPrompt.isNotBlank()) sb.append("[INST] ").append(systemPrompt.trim()).append(" [/INST]")
        for (turn in history) {
            if (turn.role.lowercase() == "user" || turn.role.lowercase() == "human") {
                sb.append("[INST] ").append(turn.content.trim()).append(" [/INST]")
            } else {
                sb.append(turn.content.trim()).append("</s>")
            }
        }
        sb.append("[INST] ").append(currentUserPrompt.trim()).append(" [/INST]")
        return sb.toString()
    }

    private fun renderPhi(systemPrompt: String, history: List<ChatTurn>, currentUserPrompt: String): String {
        val sb = StringBuilder()
        if (systemPrompt.isNotBlank()) sb.append("<|system|>\n").append(systemPrompt.trim()).append("<|end|>\n")
        for (turn in history) {
            if (turn.role.lowercase() == "user" || turn.role.lowercase() == "human") {
                sb.append("<|user|>\n").append(turn.content.trim()).append("<|end|>\n")
            } else {
                sb.append("<|assistant|>\n").append(turn.content.trim()).append("<|end|>\n")
            }
        }
        sb.append("<|user|>\n").append(currentUserPrompt.trim()).append("<|end|>\n")
        sb.append("<|assistant|>\n")
        return sb.toString()
    }

    private fun renderFalcon(systemPrompt: String, history: List<ChatTurn>, currentUserPrompt: String): String {
        val sb = StringBuilder()
        if (systemPrompt.isNotBlank()) sb.append("System: ").append(systemPrompt.trim()).append("\n")
        for (turn in history) {
            if (turn.role.lowercase() == "user" || turn.role.lowercase() == "human") {
                sb.append("User: ").append(turn.content.trim()).append("\n")
            } else {
                sb.append("Falcon: ").append(turn.content.trim()).append("\n")
            }
        }
        sb.append("User: ").append(currentUserPrompt.trim()).append("\nFalcon:")
        return sb.toString()
    }

    /**
     * Context-window budgeting like llama.cpp `-c/--ctx-size` + `--keep` and Ollama `num_ctx`.
     * Keeps system + most recent turns that fit in [maxTokens] (est. 4 chars/token),
     * always retaining the first user turn as the anchor (llama.cpp `--keep` behaviour).
     */
    fun truncateHistory(
        history: List<ChatTurn>,
        systemPrompt: String,
        currentUserPrompt: String,
        maxTokens: Int
    ): List<ChatTurn> {
        if (maxTokens <= 0 || history.isEmpty()) return history
        fun estTokens(s: String) = s.length / 4 + 4 // + per-message overhead like Ollama
        var budget = maxTokens - estTokens(systemPrompt) - estTokens(currentUserPrompt) - 256 // reserve generation
        if (budget <= 0) return emptyList()
        val kept = ArrayDeque<ChatTurn>()
        // Walk from most recent backwards.
        for (i in history.indices.reversed()) {
            val t = history[i]
            val cost = estTokens(t.content)
            if (cost > budget) {
                // Try to keep a truncated tail of this turn rather than dropping everything.
                if (kept.isEmpty() && t.content.length > 500) {
                    val tail = t.content.takeLast(budget * 4)
                    kept.addFirst(t.copy(content = "…(truncated)…\n$tail"))
                }
                break
            }
            kept.addFirst(t)
            budget -= cost
        }
        // `--keep` anchor: if we dropped the first user turn and still have room, note it.
        return kept.toList()
    }
}
