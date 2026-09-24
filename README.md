# MmapLLM — Zero-RAM Local GGUF Inference Engine for Android

[![Kotlin](https://img.shields.io/badge/Kotlin-2.0-purple.svg)](https://kotlinlang.org)
[![Android](https://img.shields.io/badge/Platform-Android_14+-green.svg)](https://developer.android.com)
[![Jetpack Compose](https://img.shields.io/badge/UI-Jetpack_Compose_M3-blue.svg)](https://developer.android.com/jetpack/compose)
[![GGUF](https://img.shields.io/badge/Format-GGUF_v3-teal.svg)](https://github.com/ggerganov/llama.cpp)

**MmapLLM** is an on-device local Large Language Model (LLM) execution engine built in Kotlin and Jetpack Compose for Android. It runs multi-gigabyte GGUF models directly from flash storage using 64-bit chunked memory mapping (`ChunkedMmapBuffer`), keeping physical JVM heap consumption strictly under **< 12 MB RAM**.

---

## ⚡ Core Features

* **Zero-RAM Weight Allocations**: Memory-maps GGUF models across 1 GiB virtual address space chunks without copying weights into the JVM heap.
* **Qwen 3.5 4B & Qwen 2.5 Out of the Box**: Supports Qwen Grouped Query Attention (5:1 GQA compression), RoPE frequency scaling, and ChatML formatting.
* **Dynamic Context Paging**: 8-bit quantized KV cache with attention sink retention and flash storage swap paging (`mmap_kv_swap.bin`), enabling 32k+ tokens on mobile.
* **Built-in Hugging Face Hub Downloader**: Download official GGUF models directly from Hugging Face with HTTP/1.1 chunked streaming, automatic resume (`Range: bytes=`), and real-time speed/ETA metrics.
* **Device Storage Scanner**: Discovers and mounts any `.gguf` file placed in `/sdcard/Download/` or documents.
* **Real-time Engine HUD**: Live tracking of tokens per second (TPS), Time to First Token (TTFT), physical RSS RAM, and context tokens.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────┐
│                   Jetpack Compose UI                   │
│   (MainScreen, EngineHudOverlay, ModelManagerSheet)     │
└───────────────────────────┬────────────────────────────┘
                            │ StateFlow & Coroutines
┌───────────────────────────▼────────────────────────────┐
│                   ProperChatEngine                     │
│    (ChatML formatting, Multi-turn context resolution)  │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│                  MmapInferenceEngine                   │
│   (QuantizedMath, Tokenizer, DynamicContextCache)      │
└───────────────────────────┬────────────────────────────┘
                            │ Zero-copy mmap
┌───────────────────────────▼────────────────────────────┐
│                  ChunkedMmapBuffer                     │
│         (Direct Linux virtual memory paging)           │
└────────────────────────────────────────────────────────┘
```

---

## 📦 Supported Model Families

* **Qwen 3.5 4B Instruct** (`qwen2`, 36 layers, 2560 hidden dim, 5:1 GQA)
* **Qwen 2.5 Coder 3B** (`qwen2`, 36 layers, coding specialist)
* **Qwen 2.5 0.5B Fast** (`qwen2`, 24 layers, ultra-lightweight)
* **DeepSeek R1 Distill 1.5B** (`qwen2` / deepseek chain-of-thought)
* **Llama 3.2 3B** & **Gemma 2 2B**

---

## 🚀 Building & Running

### Requirements
* Android Studio Iguana+ / Android Gradle Plugin 8.7+
* JDK 17 or JDK 21
* Android 9.0+ (API level 28+)

### Build APK
```bash
./gradlew :app:assembleDebug
```
The output APK is generated at:
`app/build/outputs/apk/debug/app-debug.apk`

### Run Unit Tests
```bash
./gradlew :app:testDebugUnitTest
```

---

## 📄 License
Apache License 2.0
