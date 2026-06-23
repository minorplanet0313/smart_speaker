# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Raspberry Pi 3B+-based smart speaker (like Xiaomi "XiaoAi") with wake word detection, local/cloud ASR, DeepSeek LLM, TTS, and a pluggable skill framework. Written in Python 3.9+.

## Commands

```bash
# Activate virtual environment (after install)
source ~/smart_speaker_venv/bin/activate

# Run the speaker
python src/main.py

# Run with custom config
python src/main.py --config config/config.yaml

# List audio devices
python src/main.py --list-devices

# List available Edge TTS voices
python src/main.py --list-voices

# Test ASR on an audio file
python src/main.py --test-asr audio.wav

# Run tests (on dev machine, no Pi hardware needed)
pytest tests/test_basic.py -v

# Install everything on a Raspberry Pi
chmod +x scripts/install.sh && ./scripts/install.sh

# systemd service
sudo systemctl start smart-speaker
journalctl -u smart-speaker -f

# Log level override
python src/main.py --log-level DEBUG
```

## Architecture

### Pipeline Flow

```
USB Mic → AudioCapture → WakeWordDetector (openWakeWord)
                       → VAD (Silero, energy fallback)
                       → ASR (Vosk local, cloud fallback)
                       → SkillManager (priority-based skill matching)
                       → LLM (DeepSeek via OpenAI SDK, streaming)
                       → TTS (Edge TTS online, Piper offline fallback)
                       → AudioPlayer → Speaker
```

### Core Design

**Event-driven architecture.** All module communication flows through `EventBus` (singleton pub/sub in `src/core/event_bus.py`). Modules subscribe to events and publish results — there is no direct coupling between components. The event types are defined as an enum in `Event`.

**State machine** (`src/core/state_machine.py`) governs the interaction lifecycle:
`IDLE → LISTENING → THINKING → SPEAKING → IDLE`
MUTED and ERROR states can be entered from any state. Each state has a configurable timeout that force-resets to IDLE. Transitions are strictly validated against a whitelist.

**SmartSpeakerEngine** (`src/core/engine.py`) is the central orchestrator. It:
1. Instantiates all modules in `setup()`
2. Wires events to handlers in `_wire_events()`
3. Runs the main loop in `run_forever()` (checks timeouts, sleeps 100ms)
4. Handles barge-in (wake word during playback interrupts TTS)

### Module Organization

| Directory | Purpose |
|---|---|
| `src/core/` | Engine, EventBus, StateMachine — the backbone |
| `src/audio/` | AudioCapture (PyAudio streaming), AudioPlayer (blocking playback with stop support), VAD (Silero VAD with energy-based fallback), audio format utils |
| `src/wake_word/` | WakeWordDetector wrapping openWakeWord (ONNX/TFLite), sliding-window inference, cooldown logic |
| `src/asr/` | BaseASR abstract class + VoskASR (local offline). Cloud ASR (Baidu/Aliyun) in `cloud_asr.py` |
| `src/llm/` | BaseLLM abstract class + DeepSeekLLM (OpenAI-compatible SDK, exponential backoff retry). ConversationContext manages multi-turn history with round limits and expiry |
| `src/tts/` | BaseTTS abstract class + EdgeTTS (free Microsoft Edge TTS, sync wrapper over async), PiperTTS (local offline backup) |
| `src/skills/` | BaseSkill abstract class + SkillManager (priority-ordered matching). Built-in skills: ChatSkill (LLM fallback, always matches), TimeSkill (time/date/weekday queries), WeatherSkill (QWeather API) |
| `src/gpio/` | LED and Button control via gpiozero (optional, disabled by default) |
| `src/utils/` | Config (YAML + `${ENV_VAR}` substitution + dotenv), Logger (console + rotating file) |

### Skill Framework

Skills are pluggable intent handlers. Each skill declares:
- `name` (unique identifier)
- `keywords` (trigger words)
- `priority` (HIGH/NORMAL/LOW/FALLBACK — matched in descending priority order)
- `can_handle(text)` → bool
- `execute(text, context)` → SkillResult

`SkillManager` iterates skills by priority, calls `can_handle()`, and executes the first match. `ChatSkill` has `FALLBACK` priority and always returns `can_handle=True`, routing unknown inputs to the LLM. To add a new skill: create a class extending `BaseSkill` in `src/skills/builtin/`, then register it in `engine.py:_init_skills()`.

Skills can return `needs_llm=True` to signal that the LLM should process the input further (used by ChatSkill). Direct responses bypass the LLM entirely.

### Configuration

`config/config.yaml` is the single source of config. Supports `${ENV_VAR}` substitution (e.g., `${DEEPSEEK_API_KEY}`). `.env` file is auto-loaded via python-dotenv. Access with `config.get("audio.vad.threshold", 0.5)` using dot-path notation.

### Key Design Decisions

- **Audio format convention**: float32 numpy arrays, range [-1, 1], 16kHz mono throughout the pipeline. Conversion from int16 happens at AudioCapture.
- **Lazy model loading**: VAD, WakeWord, and ASR models are loaded on first use, not at import time.
- **Graceful degradation**: VAD falls back to energy detection if Silero unavailable. ASR falls back to cloud if Vosk model missing. TTS falls back to Piper if Edge TTS fails.
- **Threading model**: Audio capture runs in a dedicated thread. ASR and TTS are each dispatched to temporary daemon threads. The main thread runs the event loop.
- **Barge-in**: Wake word detection during SPEAKING state stops playback and transitions to LISTENING.

### Tests

Tests in `tests/test_basic.py` are plain Python classes (not pytest classes) that pytest discovers by `Test` prefix convention. They test Config, EventBus, StateMachine, ConversationContext, SkillManager, and audio utils — all without hardware dependencies.
