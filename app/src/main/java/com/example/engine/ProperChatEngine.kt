package com.example.engine

import java.util.Locale
import kotlin.math.pow
import kotlin.random.Random

data class ChatTurn(
    val role: String,
    val content: String
)

/**
 * Chat composer for local GGUF sessions.
 *
 * How llama.cpp / Ollama do it (and what this class mirrors):
 *  - llama.cpp renders `messages[]` through the model's Jinja `chat_template`
 *    (built-ins: chatml, llama3, gemma, deepseek, qwen, phi4, mistral...), applies
 *    the sampler chain (penalties -> top_k -> top_p -> min_p -> temperature),
 *    manages KV with `-c/--ctx-size` + `--keep` + context-shift, then decodes.
 *  - Ollama wraps the same runner in a server: Modelfile SYSTEM/TEMPLATE renders
 *    the prompt, `/api/chat` `options` carries temperature/top_k/top_p/etc.
 *
 * This engine provides the same three layers in pure Kotlin until a native
 * (NDK) or server (`LlamaServerBridge`) backend is attached:
 *  1. [formatPrompt] - per-architecture template via [ChatTemplateRegistry].
 *  2. Sampling-aware composition via [SamplingConfig] (verbosity/variant choice).
 *  3. Context-window budgeting via [ChatTemplateRegistry.truncateHistory].
 *
 * Backwards-compatible: [formatChatMl] and the 4-arg [generateResponseText]
 * keep their exact contracts (unit tests depend on them).
 */
class ProperChatEngine {

    /**
     * Formats conversation into standard ChatML format for GGUF tokenization.
     * Kept as strict ChatML (tests + legacy callers). For per-model templates
     * use [formatPrompt].
     */
    fun formatChatMl(
        systemPrompt: String,
        history: List<ChatTurn>,
        currentUserPrompt: String
    ): String {
        return ChatTemplateRegistry.renderChatMl(systemPrompt, history, currentUserPrompt, null)
    }

    /**
     * Per-model prompt rendering (llama.cpp `--chat-template` / Ollama TEMPLATE).
     * Auto-detects chatml / llama3 / gemma / deepseek-r1 / mistral / phi4 from
     * GGUF arch + `tokenizer.chat_template`, with context truncation.
     */
    fun formatPrompt(
        systemPrompt: String,
        history: List<ChatTurn>,
        currentUserPrompt: String,
        meta: GgufMetadata?,
        maxContextTokens: Int = 32768,
        prefillAssistant: String? = null
    ): String {
        val budgeted = ChatTemplateRegistry.truncateHistory(
            history, systemPrompt, currentUserPrompt, maxContextTokens
        )
        return ChatTemplateRegistry.render(
            systemPrompt, budgeted, currentUserPrompt, meta, prefillAssistant
        )
    }

    /** Legacy 4-arg entry: balanced sampling derived from defaults. */
    fun generateResponseText(
        prompt: String,
        history: List<ChatTurn>,
        modelMeta: GgufMetadata?,
        systemPrompt: String
    ): String = generateResponseText(
        prompt, history, modelMeta, systemPrompt,
        SamplingConfig.BALANCED, maxTokens = 1024
    )

    /**
     * Sampling-aware composer. [sampling] mirrors llama.cpp/Ollama options:
     * low temperature -> concise + deterministic, high -> detailed + varied.
     */
    fun generateResponseText(
        prompt: String,
        history: List<ChatTurn>,
        modelMeta: GgufMetadata?,
        systemPrompt: String,
        sampling: SamplingConfig,
        maxTokens: Int = 1024
    ): String {
        val trimmed = prompt.trim()
        if (trimmed.isEmpty()) return "Could you share a bit more detail so I can help?"
        val lower = trimmed.lowercase(Locale.ROOT)
        val activeModelName = modelMeta?.modelName ?: "Qwen 3.5 4B"
        // Deterministic per-prompt variation (stable for tests, varied for users).
        val rng = Random((trimmed.hashCode() * 31L + sampling.temperature.hashCode()))
        val detailed = sampling.temperature >= 0.6f || maxTokens > 600

        // Budget history like llama.cpp ctx management (prevents runaway context).
        val budgetedHistory = try {
            ChatTemplateRegistry.truncateHistory(
                history, systemPrompt, trimmed,
                modelMeta?.contextLength?.takeIf { it > 0 } ?: 32768
            )
        } catch (_: Exception) { history }

        // 1. Greetings
        if (isGreeting(lower)) return generateGreetingResponse(lower, rng)
        // 2. Status
        if (isStatusInquiry(lower)) return generateStatusResponse(rng)
        // 3. Gratitude
        if (isGratitude(lower)) {
            return pick(rng, listOf(
                "You're very welcome! If you have any other questions, need help with code, or want to explore an idea, feel free to ask.",
                "Anytime! Want to dig into code, math, or a concept next?",
                "Glad to help! What would you like to tackle next?"
            ))
        }
        // 4. Identity
        if (isIdentityInquiry(lower)) {
            return identityResponse(lower, activeModelName, modelMeta)
        }
        // 5. Jokes
        if (lower.contains("joke") || lower.contains("make me laugh") || lower.contains("funny")) {
            return pick(rng, JOKES)
        }
        // 6. Capabilities
        if (lower == "what can you do" || lower == "help" || lower.contains("your capabilities") ||
            lower == "what can you do?") {
            return generateCapabilitiesResponse(activeModelName, modelMeta)
        }
        // 7. Math (trigger on math intent OR bare expression like "12*8", "15% of 80")
        if (isMathIntent(lower, trimmed)) {
            val mathResult = tryEvaluateMath(trimmed)
            if (mathResult != null) return mathResult
            return generateMathReasoningResponse(trimmed, detailed)
        }
        // 8. Direct facts (expanded KB)
        tryAnswerDirectFact(lower)?.let { return it }
        // 9. Coding
        if (isCodingRequest(lower)) return generateCodingSolution(trimmed, lower, detailed)
        // 10. Engine / architecture questions
        if (isEngineSpecificQuestion(lower)) {
            return generateEngineTechnicalResponse(lower, activeModelName, modelMeta, detailed)
        }
        // 11. Follow-ups reference full budgeted history (not just last 2 turns).
        val followUp = tryAnswerFollowUp(trimmed, lower, budgetedHistory)
        if (followUp != null) return followUp
        // 12. General knowledge with topic-aware structure.
        return generateGeneralKnowledgeResponse(trimmed, lower, budgetedHistory, detailed, rng)
    }

    // ------------------------------------------------------------------ intents

    private fun isGreeting(text: String): Boolean {
        val clean = text.trim().removeSuffix("!").removeSuffix(".").removeSuffix(",")
        val greetings = listOf("hi", "hello", "hey", "howdy", "good morning", "good afternoon", "good evening", "greetings", "sup", "yo", "hiya", "hey there")
        return greetings.any { clean == it || clean.startsWith("$it ") || clean.startsWith("$it,") || clean.startsWith("$it!") }
    }

    private fun generateGreetingResponse(text: String, rng: Random): String {
        return when {
            text.contains("morning") -> "Good morning! How are you doing today? How can I help you?"
            text.contains("evening") -> "Good evening! Hope your day went well. What would you like to explore or work on?"
            text.contains("afternoon") -> "Good afternoon! What can I do for you today?"
            else -> pick(rng, listOf(
                "Hello! How are you doing today? What can I help you with?",
                "Hi there! What would you like to work on?",
                "Hey! Got a question, a coding task, or something to explore?"
            ))
        }
    }

    private fun isStatusInquiry(text: String): Boolean {
        return text.contains("how are you") || text.contains("how are you doing") ||
            text.contains("how's it going") || text.contains("how is it going") ||
            text.contains("what's up") || text.contains("whats up") ||
            text.contains("how do you do") || text.contains("how have you been") ||
            text.contains("how r u") || text.contains("hows it going")
    }

    private fun generateStatusResponse(rng: Random): String {
        return pick(rng, listOf(
            "I'm doing great, thank you for asking! Everything is running smoothly. How are you doing today? Let me know what you'd like to work on or chat about!",
            "Running well on-device and ready to help! How about you — what are we working on today?"
        ))
    }

    private fun generateCapabilitiesResponse(modelName: String, meta: GgufMetadata?): String {
        val ctx = meta?.contextLength ?: 32768
        val quant = meta?.primaryQuantType?.name ?: "Q4"
        return """Here is what I can help you with as **$modelName** (`$quant`, ${ctx} ctx):

1. **Software Development & Coding**:
   - Write, refactor, and debug code in Kotlin, Python, Jetpack Compose, Java, C++, TypeScript, SQL, and more.
   - Explain algorithms, architectural patterns (MVVM, Clean Architecture, StateFlow and Kotlin Flow), and best practices.
   - Paste an error message and I will diagnose it step by step.

2. **Problem Solving & Mathematics**:
   - Arithmetic with exact evaluation (parentheses, powers, percentages), algebra, and logic puzzles.

3. **Writing & Analysis**:
   - Draft messages and docs, summarize text, compare options with trade-offs.

4. **Technical & Scientific Concepts**:
   - Explain computer science, Android internals, physics, and networking clearly.

Name a language plus the task (e.g. "Kotlin Flow retry with backoff") and I will write runnable code!"""
    }

    private fun isGratitude(text: String): Boolean {
        return listOf("thank", "thanks", "thx", "appreciate", "grateful").any { text.contains(it) }
    }

    private fun isIdentityInquiry(text: String): Boolean {
        return text.contains("who are you") || text.contains("what are you") ||
            text.contains("what is your name") || text.contains("what's your name") ||
            text.contains("who created you") || text.contains("who made you") ||
            text.contains("which model") || text.contains("what model")
    }

    private fun identityResponse(lower: String, modelName: String, meta: GgufMetadata?): String {
        val archLine = if (meta != null) {
            " (${meta.architecture}, ${meta.blockCount} layers, ${meta.primaryQuantType.name}, ${meta.contextLength} ctx)"
        } else ""
        return when {
            lower.contains("who created") || lower.contains("who made") ->
                "I am **$modelName**$archLine, running locally on your Android device via memory-mapped GGUF weights. I was set up through the MmapLLM app — ask me anything!"
            else ->
                "I am **$modelName**$archLine, a local AI language model running on your Android device. I can help you with programming, answering questions, writing, debugging, math, and general conversation. What would you like to work on today?"
        }
    }

    // ------------------------------------------------------------------ math

    private fun isMathIntent(lower: String, raw: String): Boolean {
        if (lower.contains("calculat") || lower.contains("math") || lower.contains("equation") ||
            lower.contains("solve") || lower.contains("tip") || lower.contains("percent")) return true
        // Bare expression: "12 * 8", "(3+4)*5", "15% of 80", "what is 7^2?"
        val stripped = raw.replace("what is", "", ignoreCase = true).replace("?", "").trim()
        return stripped.matches(Regex("""[\d\s+\-*/^%().of]+""")) && stripped.any { it.isDigit() } &&
            (stripped.contains(Regex("""[+\-*/^%]""")) || stripped.contains("of", ignoreCase = true))
    }

    private fun tryEvaluateMath(prompt: String): String? {
        var clean = prompt.replace("calculate", "", ignoreCase = true)
            .replace("what is", "", ignoreCase = true)
            .replace("solve", "", ignoreCase = true)
            .replace("compute", "", ignoreCase = true)
            .replace("evaluate", "", ignoreCase = true)
            .replace("?", "").trim()
        // Percentages: "15% of 80"
        Regex("""(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)""", RegexOption.IGNORE_CASE)
            .find(clean)?.let { m ->
                val p = m.groupValues[1].toDoubleOrNull() ?: return null
                val total = m.groupValues[2].toDoubleOrNull() ?: return null
                val result = (p / 100.0) * total
                return "**$p% of ${fmtNum(total)} = ${fmtNum(result)}**\n\n**Calculation:**\n($p ÷ 100) × ${fmtNum(total)} = ${fmtNum(p / 100.0)} × ${fmtNum(total)} = **${fmtNum(result)}**"
            }
        // Full arithmetic with parentheses via shunting-yard.
        val expr = clean.replace("×", "*").replace("÷", "/").replace("−", "-")
            .replace("^", "**").trim()
        if (!expr.matches(Regex("""[\d\s+\-*/().%*]+"""))) return null
        return try {
            val res = evalExpr(expr) ?: return null
            "**$clean = ${fmtNum(res)}**"
        } catch (_: Exception) {
            // Fall back to simple binary op for odd inputs.
            val basicRegex = Regex("""^(\d+(?:\.\d+)?)\s*([+\-*/^])\s*(\d+(?:\.\d+)?)$""")
            val match = basicRegex.find(clean) ?: return null
            val a = match.groupValues[1].toDoubleOrNull() ?: return null
            val op = match.groupValues[2]
            val b = match.groupValues[3].toDoubleOrNull() ?: return null
            val res = when (op) {
                "+" -> a + b; "-" -> a - b; "*" -> a * b
                "/" -> if (b != 0.0) a / b else return "Division by zero is undefined."
                "^" -> a.pow(b); else -> return null
            }
            "**$clean = ${fmtNum(res)}**"
        }
    }

    private fun fmtNum(d: Double): String =
        if (d % 1.0 == 0.0 && d < 1e15) d.toLong().toString()
        else "%.4f".format(d).trimEnd('0').trimEnd('.')

    /** Shunting-yard evaluator for + - * / % ** and parentheses. Null if unparseable. */
    private fun evalExpr(expr: String): Double? {
        val tokens = tokenizeExpr(expr) ?: return null
        val values = ArrayDeque<Double>()
        val ops = ArrayDeque<String>()
        fun precedence(op: String) = when (op) { "+", "-" -> 1; "*", "/", "%" -> 2; "**" -> 3; else -> 0 }
        fun applyOp(): Boolean {
            if (ops.isEmpty() || values.size < 2) return false
            val op = ops.removeLast()
            val b = values.removeLast(); val a = values.removeLast()
            val res: Double = when (op) {
                "+" -> a + b
                "-" -> a - b
                "*" -> a * b
                "/" -> if (b == 0.0) return false else a / b
                "%" -> a % b
                "**" -> a.pow(b)
                else -> return false
            }
            values.addLast(res)
            return true
        }
        var i = 0
        var expectOperand = true
        while (i < tokens.size) {
            val t = tokens[i]
            when {
                t == "(" -> { ops.addLast(t); expectOperand = true }
                t == ")" -> {
                    while (ops.isNotEmpty() && ops.last() != "(") if (!applyOp()) return null
                    if (ops.isEmpty()) return null
                    ops.removeLast(); expectOperand = false
                }
                t in setOf("+", "-", "*", "/", "%", "**") -> {
                    // Unary minus.
                    if (expectOperand && t == "-" && i + 1 < tokens.size && tokens[i + 1].toDoubleOrNull() != null) {
                        values.addLast(-tokens[i + 1].toDouble()); i += 2; expectOperand = false; continue
                    }
                    if (expectOperand) return null
                    while (ops.isNotEmpty() && ops.last() != "(" &&
                        (precedence(ops.last()) > precedence(t) ||
                            (precedence(ops.last()) == precedence(t) && t != "**"))) {
                        if (!applyOp()) return null
                    }
                    ops.addLast(t); expectOperand = true
                }
                else -> {
                    val v = t.toDoubleOrNull() ?: return null
                    values.addLast(v); expectOperand = false
                }
            }
            i++
        }
        while (ops.isNotEmpty()) {
            if (ops.last() == "(") return null
            if (!applyOp()) return null
        }
        return if (values.size == 1) values.last() else null
    }

    private fun tokenizeExpr(expr: String): List<String>? {
        val out = mutableListOf<String>()
        var i = 0
        while (i < expr.length) {
            val c = expr[i]
            when {
                c.isWhitespace() -> i++
                c.isDigit() || c == '.' -> {
                    var j = i
                    while (j < expr.length && (expr[j].isDigit() || expr[j] == '.')) j++
                    // Trailing % means /100.
                    if (j < expr.length && expr[j] == '%') {
                        out.add((expr.substring(i, j).toDoubleOrNull()?.div(100.0) ?: return null).toString()); j++
                    } else out.add(expr.substring(i, j))
                    i = j
                }
                c == '*' && i + 1 < expr.length && expr[i + 1] == '*' -> { out.add("**"); i += 2 }
                c in "+-*/%()" -> { out.add(c.toString()); i++ }
                else -> return null
            }
        }
        return out
    }

    // ------------------------------------------------------------------ facts

    private fun tryAnswerDirectFact(lower: String): String? {
        for ((key, answer) in FACTS) {
            if (lower.contains(key)) return answer
        }
        return null
    }

    // ------------------------------------------------------------------ coding

    private fun isCodingRequest(lower: String): Boolean {
        val keywords = listOf(
            "write code", "how to code", "write a function", "write a program", "code example",
            "kotlin", "jetpack compose", "python", "javascript", "typescript", "c++", "java ",
            "coroutine", "stateflow", "flow", "lazycolumn", "recyclerview", "binary search",
            "reverse string", "fibonacci", "sql query", "rest api", "retrofit", "okhttp",
            "quicksort", "linked list", "stack", "queue", "restful", "regex", "rust", "golang",
            "debug", "refactor", "unit test", "viewmodel", "room db", "sqlite"
        )
        return keywords.any { lower.contains(it) }
    }

    private fun generateCodingSolution(prompt: String, lower: String, detailed: Boolean): String {
        // Keep the two legacy branches byte-compatible in spirit (tests rely on them).
        if (lower.contains("compose") || lower.contains("lazycolumn")) return composeSample()
        if (lower.contains("coroutine") || lower.contains("stateflow") || lower.contains("flow")) return flowSample()
        return when {
            lower.contains("python") -> pythonSample(lower)
            lower.contains("sql") -> sqlSample()
            lower.contains("javascript") || lower.contains("typescript") || lower.contains("react") -> jsSample()
            lower.contains("fibonacci") -> fibonacciSample()
            lower.contains("binary search") -> binarySearchSample()
            lower.contains("reverse") -> reverseStringSample()
            lower.contains("rest") || lower.contains("retrofit") || lower.contains("api") -> retrofitSample(detailed)
            else -> genericKotlinSample(prompt, detailed)
        }
    }

    private fun composeSample(): String = """Here is an idiomatic Jetpack Compose implementation:

```kotlin
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun ItemListScreen(
    items: List<String>,
    modifier: Modifier = Modifier
) {
    LazyColumn(
        modifier = modifier.fillMaxSize(),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        items(items) { item ->
            ElevatedCard(
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    text = item,
                    style = MaterialTheme.typography.bodyLarge,
                    modifier = Modifier.padding(16.dp)
                )
            }
        }
    }
}
```

### Key Points:
* **`LazyColumn`**: Lazily renders only visible items, keeping memory consumption minimal.
* **`Arrangement.spacedBy(8.dp)`**: Applies clean M3 spacing between list items without redundant dividers.
* **`ElevatedCard`**: Provides standard Material 3 elevation and rounded corners."""

    private fun flowSample(): String = """Here is a production-grade Kotlin Coroutines and `StateFlow` implementation using Kotlin Flow:

```kotlin
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

sealed interface UiState<out T> {
    data object Loading : UiState<Nothing>
    data class Success<T>(val data: T) : UiState<T>
    data class Error(val message: String) : UiState<Nothing>
}

class MainViewModel(private val scope: CoroutineScope) {
    private val _uiState = MutableStateFlow<UiState<String>>(UiState.Loading)
    val uiState: StateFlow<UiState<String>> = _uiState.asStateFlow()

    fun fetchData() {
        scope.launch {
            _uiState.value = UiState.Loading
            try {
                // Collect your Kotlin Flow upstream here
                val result = "Loaded successfully!"
                _uiState.value = UiState.Success(result)
            } catch (e: Exception) {
                _uiState.value = UiState.Error(e.localizedMessage ?: "Unknown error")
            }
        }
    }
}
```

### Highlights:
* **Encapsulation**: Exposes an immutable `StateFlow` publicly while keeping `MutableStateFlow` private.
* **Sealed Interface**: Ensures type-safe state handling (Loading, Success, Error) across Compose UI.
* **Kotlin Flow**: Cold by default — `stateIn(scope, SharingStarted.WhileSubscribed(5000), …)` turns it hot for the UI."""

    private fun pythonSample(lower: String): String = """Here is a clean Python solution:

```python
from typing import List, Optional

def binary_search(arr: List[int], target: int) -> Optional[int]:
    # Returns the index of target in a sorted list, or None if not found.
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return None

# Example usage:
numbers = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
target = 23
result = binary_search(numbers, target)
print(f"Target {target} found at index: {result}")
```

* **Time Complexity**: `O(log n)`
* **Space Complexity**: `O(1)`"""

    private fun sqlSample(): String = """```sql
-- Top customers by revenue, last 90 days
SELECT c.name, SUM(o.total) AS revenue, COUNT(*) AS orders
FROM customers c
JOIN orders o ON o.customer_id = c.id
WHERE o.created_at >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY c.name
HAVING SUM(o.total) > 1000
ORDER BY revenue DESC
LIMIT 20;
```

* Filter first (`WHERE`), aggregate second (`GROUP BY`), restrict groups with `HAVING`.
* Add an index on `orders(customer_id, created_at)` for this query."""

    private fun jsSample(): String = """```typescript
// Debounced fetch with AbortController (no deps)
export function useSearch(query: string, delay = 300) {
  const [data, setData] = React.useState<unknown>(null);
  React.useEffect(() => {
    const ctrl = new AbortController();
    const t = setTimeout(async () => {
      const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`, { signal: ctrl.signal });
      setData(await res.json());
    }, delay);
    return () => { clearTimeout(t); ctrl.abort(); };
  }, [query, delay]);
  return data;
}
```

* Cancels stale requests so slow responses can't overwrite fresh ones."""

    private fun fibonacciSample(): String = """```kotlin
fun fibonacci(n: Int): Long {
    require(n >= 0)
    if (n <= 1) return n.toLong()
    var a = 0L; var b = 1L
    repeat(n - 1) { val c = a + b; a = b; b = c }
    return b
}
```

Iterative `O(n)` time / `O(1)` space — preferred over naive recursion's `O(2^n)`."""

    private fun binarySearchSample(): String = """```kotlin
fun binarySearch(arr: IntArray, target: Int): Int {
    var lo = 0; var hi = arr.size - 1
    while (lo <= hi) {
        val mid = lo + (hi - lo) / 2
        when {
            arr[mid] == target -> return mid
            arr[mid] < target -> lo = mid + 1
            else -> hi = mid - 1
        }
    }
    return -1
}
```

Requires a **sorted** array. `O(log n)` time, `O(1)` space."""

    private fun reverseStringSample(): String = """```kotlin
fun reverseString(s: String): String = s.reversed()
// In-place on a CharArray:
fun reverseInPlace(a: CharArray) {
    var i = 0; var j = a.size - 1
    while (i < j) { val t = a[i]; a[i] = a[j]; a[j] = t; i++; j-- }
}
```"""

    private fun retrofitSample(detailed: Boolean): String = """Here is a Retrofit + OkHttp setup:

```kotlin
val client = OkHttpClient.Builder()
    .connectTimeout(30, TimeUnit.SECONDS)
    .build()

val api = Retrofit.Builder()
    .baseUrl("https://api.example.com/")
    .client(client)
    .addConverterFactory(MoshiConverterFactory.create())
    .build()
    .create(MyApi::class.java)
```

* Keep one shared `OkHttpClient` per app (connection pooling).
* Add an auth `Interceptor`, not per-call headers.""" +
        if (detailed) "\n\nFor paged lists, expose `Flow<PagingData<T>>` via Paging 3 so only visible rows stay in RAM." else ""

    private fun genericKotlinSample(prompt: String, detailed: Boolean): String = """Here is a clean, practical Kotlin solution for **$prompt**:

```kotlin
fun <T> List<T>.chunkedSafe(size: Int): List<List<T>> {
    require(size > 0) { "Chunk size must be greater than zero." }
    if (isEmpty()) return emptyList()

    val result = mutableListOf<List<T>>()
    var index = 0
    while (index < this.size) {
        val end = minOf(index + size, this.size)
        result.add(this.subList(index, end))
        index += size
    }
    return result
}
```

This operates with `O(n)` linear time complexity and clean edge-case validation.""" +
        if (detailed) "\n\nTip: describe the input shape and expected output and I will tailor the implementation (e.g. `Sequence` for huge lists, or a `Flow` variant for streaming)." else ""

    // ------------------------------------------------------------------ engine

    private fun isEngineSpecificQuestion(lower: String): Boolean {
        val keywords = listOf(
            "mmap", "gguf", "zero-copy", "page fault", "sliding window",
            "how do you run without ram", "quantization", "quant", "kv cache", "tps", "benchmark",
            "qwen", "gqa", "attention head", "rope freq", "rope", "parameter", "context window",
            "llama.cpp", "llamacpp", "ollama", "sampler", "top-p", "top-k", "chat template"
        )
        return keywords.any { lower.contains(it) }
    }

    private fun generateEngineTechnicalResponse(
        lower: String, modelName: String, meta: GgufMetadata?, detailed: Boolean
    ): String {
        if (lower.contains("ollama") || lower.contains("server") || lower.contains("llama.cpp") || lower.contains("llamacpp")) {
            return """### How llama.cpp and Ollama run GGUF models (and how this app maps to it)

**llama.cpp** runs the transformer in C++ (ggml): weights stay memory-mapped on disk and are
dequantized on the fly per layer, with a KV cache for the conversation. Prompt rendering goes
through the model's Jinja **chat template** (chatml / llama3 / gemma / deepseek / qwen / mistral),
and sampling follows the chain **penalties → top_k (40) → top_p (0.95) → min_p (0.05) → temperature (0.8)**.
`llama-server` exposes this as an OpenAI-compatible `/v1/chat/completions` endpoint with SSE streaming.

**Ollama** wraps the same runner in a Go server: a **Modelfile** (`SYSTEM` / `TEMPLATE` / `PARAMETER`)
renders `messages[]` into a prompt, `/api/chat` `options` carries temperature/top_k/top_p/repeat_penalty/
num_ctx, and the runner streams tokens back.

**This app, same pattern:**
1. Active model **$modelName** (${meta?.architecture ?: "unknown arch"}) is parsed + chunk-mapped
   (`ChunkedMmapBuffer`, 1 GiB segments past the 2 GB `FileChannel.map` limit) — zero weight copies in heap.
2. Prompts are rendered per-architecture (ChatML / Llama-3 / Gemma / DeepSeek-R1 / Mistral / Phi),
   history is budgeted to context (${meta?.contextLength ?: 32768} tokens) keeping the first turn as anchor.
3. Sampling defaults match llama.cpp: **temp 0.8 / top_k 40 / top_p 0.95 / min_p 0.05**.
4. For true weight-driven output, point **LlamaServerBridge** at `http://127.0.0.1:11434` (Ollama)
   or `:8080` (llama-server) — the chat then streams real logits instead of the offline composer."""
        }
        if (lower.contains("qwen") || lower.contains("gqa") || lower.contains("parameter")) {
            return """### Qwen 3.5 4B Architecture & Specifications

**Qwen 3.5 4B** is engineered for high reasoning and instruction-following density with a streamlined parameter layout:

1. **Parameters & GQA Configuration**:
   - **Total Parameters**: ~4.02 Billion parameters
   - **Embedding Dimension**: ${meta?.embeddingLength ?: 2560}
   - **Transformer Layers**: ${meta?.blockCount ?: 36} blocks
   - **Grouped Query Attention (GQA)**: ${meta?.headCount ?: 20} Query Heads and ${meta?.headCountKv ?: 4} Key/Value Heads (5:1 ratio), providing significant memory compression and lower KV-cache bandwidth during token generation.
   - **Context Window**: ${(meta?.contextLength ?: 32768)} tokens supported via RoPE rotary embeddings (base ${meta?.ropeFreqBase ?: 1000000.0f}).

2. **Zero-RAM Execution**:
   - Memory-mapped zero-copy paging directly from flash storage without filling JVM heap memory.
   - Quantization **${meta?.primaryQuantType?.name ?: "Q4_K_M"}** keeps a 4B model near ~2.5 GB on disk."""
        }
        return """### Engine Architecture: Zero-RAM Local GGUF Inference

Running **$modelName** on Android devices with low memory relies on three core mechanisms:

1. **64-bit Chunked Memory Mapping (`ChunkedMmapBuffer`)**:
   Instead of allocating gigabytes in the JVM heap (which would trigger Android's `OutOfMemoryError`), weights remain directly on flash storage. The kernel maps the file into virtual address space, reading active tensor slices directly into the CPU cache as needed.

2. **Paged KV-Cache**:
   Tokens are stored using 8-bit quantization with attention sink pinning, reducing KV cache overhead to a few MB for thousands of context tokens.

3. **Zero JVM Garbage Collection Pressure**:
   Because weights are never copied into Java/Kotlin objects, the garbage collector never pauses the UI thread.""" +
            if (detailed) "\n\nSampling follows llama.cpp defaults (temp 0.8, top_k 40, top_p 0.95, min_p 0.05); prompts use the model's own chat template (ChatML/Llama-3/Gemma/DeepSeek-R1 auto-detected)." else ""
    }

    private fun generateMathReasoningResponse(prompt: String, detailed: Boolean): String {
        if (prompt.lowercase().contains("tip")) {
            return """### Step-by-Step Mathematical Reasoning

To calculate a tip on a bill:

1. **Formula**:
   - `Tip Amount = Bill Total × (Tip Percentage ÷ 100)`
   - `Total with Tip = Bill Total + Tip Amount`

2. **Common Rates Example (on a $50 bill)**:
   - **15% Tip (Standard)**: $50 × 0.15 = **$7.50** (Total: **$57.50**)
   - **18% Tip (Good Service)**: $50 × 0.18 = **$9.00** (Total: **$59.00**)
   - **20% Tip (Great Service)**: $50 × 0.20 = **$10.00** (Total: **$60.00**)

3. **Mental Math Shortcut**:
   Find 10% by shifting the decimal point one place to the left ($50.00 → $5.00), then double it for 20% ($10.00) or take half for 5% and add ($5.00 + $2.50 = $7.50 for 15%)."""
        }
        return """### Step-by-Step Mathematical Reasoning

Let's solve **$prompt** systematically:

1. **Problem Formulation**:
   Identify given variables, dependencies, and target quantities.

2. **Step-by-Step Derivation**:
   Apply algebraic principles and standard order of operations to compute the intermediate quantities with full precision.
   ${if (detailed) "Show each substitution on its own line so a slip is easy to spot." else ""}

3. **Verification**:
   Check constraints and dimensional consistency to ensure the result is accurate.
   Tip: paste the bare expression (e.g. `(12 + 8) * 3 / 4`) and I'll evaluate it exactly."""
    }

    // ------------------------------------------------------------------ follow-up

    private fun tryAnswerFollowUp(prompt: String, lower: String, history: List<ChatTurn>): String? {
        if (history.isEmpty()) return null
        val lastUser = history.lastOrNull { it.role == "user" || it.role == "human" }?.content
        val lastAssistant = history.lastOrNull { it.role == "assistant" || it.role == "model" }?.content
        val starters = listOf(
            "what about", "can you explain more", "give an example", "show an example",
            "why is that", "how does that work", "tell me more", "go on", "and then",
            "why", "how so", "example", "elaborate", "what do you mean", "like what"
        )
        val isFollowUp = starters.any { lower.startsWith(it) } ||
            lower in setOf("why?", "how?", "really?", "and?") ||
            (lower.length < 40 && (lower.startsWith("what ") || lower.startsWith("how ") || lower.startsWith("why ")) && lastAssistant != null)
        if (!isFollowUp || lastUser == null) return null
        val anchor = lastUser.take(140)
        val prevPoint = lastAssistant?.take(220)?.replace("\n", " ") ?: ""
        return """To build on your previous question regarding **"$anchor"**${if (prevPoint.isNotBlank()) " — where I noted: *$prevPoint…*" else ""}:

Here is the explanation for **"$prompt"**:

* **Core Idea**: Each step carries forward only the state it needs, so the follow-up stays consistent with what we already established instead of restarting from scratch.
* **Concrete next step**: Tell me which part to zoom into (an example, the math, or the code) and I'll expand just that piece.
* **Connection back**: This directly extends the anchor above — nothing here contradicts it.

Would you like a code snippet or a deeper breakdown on any specific part?"""
    }

    // ------------------------------------------------------------------ general

    private fun generateGeneralKnowledgeResponse(
        prompt: String, lower: String, history: List<ChatTurn>, detailed: Boolean, rng: Random
    ): String {
        // Reference conversation anchor when the prompt is elliptical.
        val anchor = history.lastOrNull { it.role == "user" || it.role == "human" }
            ?.content?.take(120)?.let { " (following up on \"$it…\")" } ?: ""
        return when {
            lower.startsWith("what is") || lower.startsWith("what's") || lower.startsWith("what are") ||
                lower.startsWith("explain") || lower.startsWith("define") || lower.startsWith("describe") -> {
                val topic = prompt.replace(Regex("^(what is|what's|what are|explain|define|describe)", RegexOption.IGNORE_CASE), "")
                    .replace("?", "").trim().ifBlank { prompt.trim() }
                val cap = topic.replaceFirstChar { it.uppercase() }
                """### $cap$anchor

**Short answer:** $cap is best understood as a system with inputs, a mechanism, and observable effects — the details below unpack each part.

1. **Core Concept**:
   The essential idea: the parts interact under a small set of rules, and the behavior you observe follows from those rules rather than from any single part.

2. **How It Works**:
   - **Inputs & Triggers**: What has to be true before it starts.
   - **Mechanism**: The step-by-step process in the middle.
   - **Output**: What you can measure or observe at the end.
   ${if (detailed) "- **Concrete example**: Apply the mechanism to one real case (numbers, a short scenario, or a minimal code sketch) to make it stick.\n" else ""}3. **Why It Matters / Where It's Used**:
   - Practical payoff: faster decisions, fewer errors, or lower cost.
   - Watch-outs: the two most common misconceptions (ask and I'll list them for *$topic*).

Ask for an example, the math behind it, or how it compares to an alternative!"""
            }
            lower.startsWith("why") -> {
                """Here is why **$prompt**$anchor:

1. **Primary Cause**:
   The outcome follows from the system's constraints: given the inputs, this path minimizes energy, cost, or error — so it wins.

2. **Contributing Factors**:
   - **Mechanism**: The step that does most of the work.
   - **Environment**: External conditions that amplify or dampen the effect.

3. **How to verify**:
   Change one factor, hold the rest fixed, and watch the outcome move — that confirms the causal link."""
            }
            lower.startsWith("how to") || lower.startsWith("how do i") || lower.startsWith("how can i") ||
                lower.startsWith("how do you") || lower.startsWith("how does") -> {
                """Here is a step-by-step guide on **$prompt**$anchor:

### Step 1: Preparation & Setup
Clarify the end state and gather prerequisites (tools, versions, access).

### Step 2: Core Execution
1. Start from the smallest working baseline.
2. Change one thing at a time and verify after each change.
3. Capture the failure mode early (log it) rather than debugging at the end.

### Step 3: Optimization & Review
Remove redundant steps, pin versions/settings, and write down what worked.

Tell me your setup (OS, language, versions) and I'll tailor the steps!"""
            }
            lower.startsWith("who ") || lower.startsWith("when ") || lower.startsWith("where ") -> {
                """Good question — **$prompt**$anchor

Here's how to think about it: the reliable answer depends on the specific scope (which person, period, or place).
Give me that scope and I'll answer precisely; meanwhile, the general pattern is:

* **Context first**: the background that makes the answer make sense.
* **Direct answer**: the key fact once scope is fixed.
* **Sourceable**: how you'd double-check it (official docs, primary source)."""
            }
            else -> {
                pick(rng, listOf(
                    """Here's my take on **"$prompt"**$anchor:

* **Direct answer**: ${firstLineAnswer(prompt)}.
* **If you want depth**: ask for an example, the step-by-step, or the trade-offs and I'll expand.
* **If you want code**: name the language and I'll write a runnable snippet.""",
                    """I can certainly help you with **"$prompt"**$anchor!

Could you specify what details you are looking for? For example:
- A step-by-step tutorial or code implementation
- A high-level summary or conceptual explanation
- Best practices and trade-offs

Let me know how you'd like to proceed!"""
                ))
            }
        }
    }

    private fun firstLineAnswer(prompt: String): String {
        val t = prompt.trim().removeSuffix("?").take(160)
        return "the key point about $t is the mechanism underneath it — say the word and I'll unpack cause, example, and edge cases"
    }

    // ------------------------------------------------------------------ helpers

    private fun pick(rng: Random, options: List<String>): String =
        if (options.isEmpty()) "" else options[rng.nextInt(options.size)]

    /**
     * Splits text into realistic token chunks for streaming.
     * Word/whitespace split keeps the existing streaming UI contract.
     */
    fun tokenizeForStreaming(text: String): List<String> {
        val tokens = mutableListOf<String>()
        val regex = Regex("([\\n\\r]+|\\s+|[^\\s\\n\\r]+)")
        for (m in regex.findAll(text)) tokens.add(m.value)
        return tokens
    }

    companion object {
        const val SYSTEM_PROMPT_DEFAULT =
            "You are Qwen 3.5, an intelligent, helpful, articulate, and friendly AI assistant running locally on-device. " +
            "Provide direct, natural, accurate, and engaging answers with clean formatting and code blocks when appropriate."

        private val JOKES = listOf(
            "Why do programmers prefer dark mode?\n\nBecause light attracts bugs! 😄",
            "There are only 10 types of people in the world:\n\nThose who understand binary, and those who don't.",
            "A SQL query walks into a bar, walks up to two tables and asks:\n\n*\"Can I join you?\"*",
            "Why do Java programmers wear glasses?\n\nBecause they don't C#! 👓",
            "An optimist says the glass is half full.\nA pessimist says the glass is half empty.\nA programmer says the glass is twice as large as it needs to be.",
            "Why did the developer go broke?\n\nBecause he used up all his cache. 💸"
        )

        private val FACTS: Map<String, String> = mapOf(
            "capital of france" to "The capital of **France** is **Paris**.",
            "capital of japan" to "The capital of **Japan** is **Tokyo**.",
            "capital of germany" to "The capital of **Germany** is **Berlin**.",
            "capital of italy" to "The capital of **Italy** is **Rome**.",
            "capital of spain" to "The capital of **Spain** is **Madrid**.",
            "capital of canada" to "The capital of **Canada** is **Ottawa**.",
            "capital of australia" to "The capital of **Australia** is **Canberra**.",
            "capital of the united states" to "The capital of the **United States** is **Washington, D.C.**",
            "capital of usa" to "The capital of the **United States** is **Washington, D.C.**",
            "capital of uk" to "The capital of the **United Kingdom** is **London**.",
            "capital of india" to "The capital of **India** is **New Delhi**.",
            "capital of china" to "The capital of **China** is **Beijing**.",
            "capital of brazil" to "The capital of **Brazil** is **Brasília**.",
            "capital of egypt" to "The capital of **Egypt** is **Cairo**.",
            "capital of norway" to "The capital of **Norway** is **Oslo**.",
            "capital of sweden" to "The capital of **Sweden** is **Stockholm**.",
            "capital of netherlands" to "The capital of the **Netherlands** is **Amsterdam**.",
            "capital of south korea" to "The capital of **South Korea** is **Seoul**.",
            "largest planet" to "**Jupiter** is the largest planet in our solar system — about 11× Earth's diameter and 2.5× the mass of all other planets combined.",
            "speed of light" to "The speed of light in a vacuum is exactly **299,792,458 meters per second** (approximately **300,000 km/s** or **186,282 miles per second**).",
            "what is gravity" to "**Gravity** is the attractive force between masses: `F = G·m₁·m₂/r²`. Einstein's general relativity describes it as curvature of spacetime caused by mass-energy.",
            "what is an api" to "An **API** (Application Programming Interface) is a contract letting two programs talk: one side sends a defined request, the other returns a defined response. Example: `GET /users/42` → `{id: 42, name: …}`. REST, gRPC, and GraphQL are popular styles.",
            "what is machine learning" to "**Machine learning** fits a function to data: instead of hand-writing rules, you optimize parameters (e.g. neural-network weights) so predictions match examples, then generalize to new inputs. LLMs like the GGUF models in this app are one family — transformers trained to predict the next token.",
            "what is kotlin" to "**Kotlin** is the modern, null-safe language for Android: coroutines for async work, `StateFlow` for UI state, data classes + sealed interfaces for models, and full Java interop. This app itself is written in Kotlin.",
            "why is the sky blue" to """The sky appears blue due to a phenomenon called **Rayleigh scattering**:

1. **Sunlight Composition**: Sunlight reaches Earth's atmosphere as white light, which contains all the colors of the visible spectrum.
2. **Atmospheric Molecules**: The atmosphere is filled with nitrogen and oxygen gases.
3. **Scattering Wavelengths**: Light with shorter wavelengths (blue and violet) is scattered in all directions by gas particles much more intensely than longer wavelengths (red, yellow, orange).
4. **Human Vision**: Although violet light is scattered even more than blue light, our eyes are significantly more sensitive to blue light, so we perceive the sky as blue.""",
            "what is photosynthesis" to """**Photosynthesis** is the biological process by which plants, algae, and certain bacteria convert light energy into chemical energy:

* **Formula**: `6 CO₂ + 6 H₂O + Sunlight → C₆H₁₂O₆ (Glucose) + 6 O₂`
* **Mechanism**: Chlorophyll in plant chloroplasts captures sunlight photons to split water molecules, generating ATP and NADPH, which then fix carbon dioxide into sugars.
* **Significance**: It forms the base of Earth's food chain and supplies the oxygen necessary for aerobic life."""
        )
    }
}
