#!/usr/bin/env python3
"""
训练中文唤醒词模型 "小智" (xiao zhi)

流程:
1. 用 Piper TTS 生成正/负样本音频
2. 用 ONNX 模型直接提取嵌入向量 (melspectrogram → embedding)
3. 训练小型 DNN 分类器
4. 导出为 ONNX 模型, 替换旧模型

用法:
    python scripts/train_xiao_zhi.py                 # 完整训练 (~30min)
    python scripts/train_xiao_zhi.py --quick         # 快速测试 (~5min)
    python scripts/train_xiao_zhi.py --generate-only # 只生成音频
"""
import argparse
import os
import sys
import time
import wave
import subprocess
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.signal import resample_poly

# ── Config ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OWW_DIR = MODELS_DIR / "openwakeword"
PIPER_VOICE = MODELS_DIR / "piper-voices" / "zh_CN-huayan-medium.onnx"
OUTPUT_DIR = OWW_DIR
MODEL_NAME = "xiao_zhi"

WAKE_WORD = "小智"
NEGATIVE_PHRASES = [
    "你好", "今天天气怎么样", "现在几点", "播放音乐", "打开灯",
    "关闭灯", "音量增加", "音量减小", "下一个", "上一个",
    "停止播放", "继续播放", "今天的新闻", "明天会下雨吗",
    "谢谢", "再见", "早上好", "晚安", "帮我查一下",
    "小志", "机智", "知道", "质量", "智慧", "智能",
]
SPEED_RANGES = [0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.25, 1.35]

# 模型架构参数 (与 openWakeWord 内部模型兼容)
# openWakeWord 输入: 16 帧 × 96-dim 嵌入 = 1536 维
EMBEDDING_DIM = 96
N_FRAMES = 16
INPUT_FEATURES = N_FRAMES * EMBEDDING_DIM  # 1536


# ── Audio Generation ────────────────────────────────────────────────────────

def generate_tts_clips(phrase: str, output_dir: Path, n_variations: int = 50):
    """用 Piper TTS 生成多种语速/语调的音频片段"""
    output_dir.mkdir(parents=True, exist_ok=True)
    clips = []

    for i in range(n_variations):
        speed = np.random.choice(SPEED_RANGES)
        # 添加随机扰动增加多样性
        noise_scale = np.random.uniform(0.0, 0.05)  # 噪声标准差乘数
        length_scale = speed * np.random.uniform(0.90, 1.10)
        length_scale = max(0.5, min(2.0, length_scale))

        result = subprocess.run(
            ["piper", "--model", str(PIPER_VOICE), "--output-raw",
             "--length-scale", str(round(length_scale, 2))],
            input=phrase.encode("utf-8"),
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0 or len(result.stdout) == 0:
            continue

        raw = np.frombuffer(result.stdout, dtype=np.int16)
        audio_16k = resample_poly(raw.astype(float), 16000, 22050).astype(np.int16)

        # 添加轻微背景噪声 (提高鲁棒性)
        if noise_scale > 0:
            noise = (np.random.randn(len(audio_16k)) * noise_scale * 2000).astype(np.int16)
            audio_16k = (audio_16k.astype(int) + noise.astype(int)).clip(-32768, 32767).astype(np.int16)

        duration = len(audio_16k) / 16000
        if duration < 0.5 or duration > 3.0:
            continue

        wav_path = output_dir / f"{i:04d}.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_16k.tobytes())

        clips.append(wav_path)

    return clips


def generate_noise_clips(output_dir: Path, n: int = 50):
    """生成多种噪声片段 (白噪声、低频、混合)"""
    output_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    noise_types = ["white", "lowfreq", "pink", "burst"]

    for i in range(n):
        duration = np.random.uniform(0.5, 3.0)
        n_samples = int(duration * 16000)
        noise_type = noise_types[i % len(noise_types)]

        if noise_type == "white":
            audio = np.random.randn(n_samples)
        elif noise_type == "lowfreq":
            t = np.linspace(0, duration, n_samples)
            audio = np.sin(2 * np.pi * 80 * t) * 0.5 + np.sin(2 * np.pi * 120 * t) * 0.3
        elif noise_type == "pink":
            # 类粉红噪声: 白噪声 + 低通滤波
            raw = np.random.randn(n_samples + 100)
            audio = np.convolve(raw, np.ones(50) / 50, mode="same")[50 : 50 + n_samples]
        else:
            # 突发噪声
            audio = np.zeros(n_samples)
            burst_start = np.random.randint(0, max(1, n_samples - 2000))
            burst_len = np.random.randint(500, 2000)
            audio[burst_start : burst_start + burst_len] = np.random.randn(burst_len)

        audio = (audio / np.abs(audio).max() * np.random.uniform(3000, 20000)).astype(np.int16)

        wav_path = output_dir / f"{noise_type}_{i:04d}.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio.tobytes())
        clips.append(wav_path)
    return clips


def generate_silence_clips(output_dir: Path, n: int = 30):
    """生成静音片段"""
    output_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for i in range(n):
        duration = np.random.uniform(0.5, 2.5)
        n_samples = int(duration * 16000)
        # 极低电平噪声 (模拟真实静音时的底噪)
        silence = (np.random.randn(n_samples) * 50).astype(np.int16)

        wav_path = output_dir / f"silence_{i:04d}.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(silence.tobytes())
        clips.append(wav_path)
    return clips


# ── Feature Extraction (纯 ONNX Runtime, 无 openWakeWord 依赖) ──────────────

# 缓存 ONNX sessions (全局, 避免重复加载)
_melspec_session = None
_embedding_session = None


def _get_onnx_sessions():
    """获取 (melspectrogram, embedding) ONNX Runtime sessions"""
    global _melspec_session, _embedding_session
    import onnxruntime as ort

    if _melspec_session is None:
        _melspec_session = ort.InferenceSession(
            str(OWW_DIR / "melspectrogram.onnx"),
            providers=["CPUExecutionProvider"],
        )
    if _embedding_session is None:
        _embedding_session = ort.InferenceSession(
            str(OWW_DIR / "embedding_model.onnx"),
            providers=["CPUExecutionProvider"],
        )
    return _melspec_session, _embedding_session


def audio_to_embeddings(audio_int16: np.ndarray) -> np.ndarray:
    """
    将 int16 音频转换为 openWakeWord 兼容的 96 维嵌入向量序列。

    复现 openWakeWord 的流式特征提取管线:
    1. 原始音频 → 1280-sample chunks (+480 context) → melspectrogram.onnx
    2. 每个 chunk → 1 mel frame (shape: 8, 32)
    3. 累积 76 mel frames → embedding_model.onnx → 96-dim embedding
    4. 滑动窗口, 步长 1 mel frame

    短音频自动用静音填充以确保至少 76 mel frames。

    Args:
        audio_int16: int16 数组, 16kHz 单声道

    Returns:
        shape (n_frames, 96) 的嵌入向量, 或空时返回 (1, 96) 零向量
    """
    melspec_sess, emb_sess = _get_onnx_sessions()
    melspec_input_name = melspec_sess.get_inputs()[0].name
    emb_input_name = emb_sess.get_inputs()[0].name

    # 参数 (与 openWakeWord 完全一致)
    chunk_size = 1280        # 80ms @ 16kHz
    context_pad = 480        # 30ms 上下文
    mel_context = 1760       # chunk + 2*context_pad
    embedding_window = 76    # 嵌入模型需要 76 mel rows

    # 确保音频足够长 (至少需要足够 chunk 以产生 76 mel rows)
    # 每个 chunk 产生 8 mel rows, 需要 ceil(76/8)=10 chunks × 1280 = 12800 samples
    min_samples = 12800
    audio_f32 = audio_int16.astype(np.float32)

    if len(audio_f32) < min_samples:
        pad_total = min_samples - len(audio_f32)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        audio_f32 = np.pad(audio_f32, (pad_left, pad_right))

    # 添加上下文填充
    padded = np.pad(audio_f32, (context_pad, context_pad))

    # Step 1: 累积 mel rows
    # 每个 1760-sample chunk → mel output (1, 1, 8, 32) → squeeze → (8, 32)
    # 这 8 行是 8 个连续的 mel 时间帧
    mel_buffer = []
    n_chunks = (len(padded) - mel_context) // chunk_size + 1

    for i in range(n_chunks):
        start = i * chunk_size
        chunk = padded[start : start + mel_context]
        if len(chunk) < mel_context:
            chunk = np.pad(chunk, (0, mel_context - len(chunk)))

        melspec_out = melspec_sess.run(
            None, {melspec_input_name: chunk.reshape(1, -1)}
        )[0]
        # melspec_out: (1, 1, 8, 32) — np.squeeze → (8, 32)
        mel_rows = np.squeeze(melspec_out)  # (8, 32) 或 (32,) if self is empty
        if mel_rows.ndim == 1:
            mel_rows = mel_rows.reshape(1, -1)
        mel_buffer.append(mel_rows)

    if len(mel_buffer) < 2:
        return np.zeros((1, EMBEDDING_DIM), dtype=np.float32)

    # Concatenate all mel rows: (n_chunks * 8, 32)
    all_mel = np.concatenate(mel_buffer, axis=0)

    # Step 2: Transform (x/10 + 2) — openWakeWord 标准变换
    all_mel = all_mel / 10.0 + 2.0

    # Step 3: 滑动窗口提取嵌入
    # embedding model input: (batch, 76, 32, 1)
    embeddings = []
    for start in range(0, all_mel.shape[0] - embedding_window + 1, 4):
        window = all_mel[start : start + embedding_window]  # (76, 32)
        if window.shape[0] != embedding_window:
            continue
        emb_input = np.expand_dims(window, axis=(0, -1)).astype(np.float32)  # (1, 76, 32, 1)
        emb_out = emb_sess.run(None, {emb_input_name: emb_input})[0]
        embedding = emb_out.squeeze()  # → (96,)
        if embedding.shape == (EMBEDDING_DIM,):
            embeddings.append(embedding)

    if not embeddings:
        return np.zeros((1, EMBEDDING_DIM), dtype=np.float32)

    return np.stack(embeddings)  # (n_frames, 96)


def extract_training_features(wav_path: Path) -> np.ndarray:
    """从 WAV 文件提取嵌入向量"""
    with wave.open(str(wav_path), "rb") as wf:
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    return audio_to_embeddings(audio)


# ── Dataset Preparation ──────────────────────────────────────────────────────

def prepare_dataset(positive_dir: Path, negative_dirs: list[Path]):
    """提取特征并构建训练集。

    短音频片段被拼接成长序列 (间隔 250ms 静音),
    然后从连续 embedding 序列中提取 16 帧滑动窗口。
    """
    X_pos, X_neg = [], []

    def extract_from_dir(audio_dir: Path, label: str) -> list:
        """读取目录中所有 wav 文件, 拼接, 提取 embedding, 返回窗口列表"""
        files = list(audio_dir.glob("*.wav"))
        if not files:
            return []

        # 读取所有音频并拼接 (加静音间隔)
        all_audio = []
        silence = np.zeros(4000, dtype=np.int16)  # 250ms
        for wav_path in files:
            try:
                with wave.open(str(wav_path), "rb") as wf:
                    audio = np.frombuffer(
                        wf.readframes(wf.getnframes()), dtype=np.int16
                    )
                all_audio.append(audio)
                all_audio.append(silence.copy())
            except Exception:
                pass

        if not all_audio:
            return []

        combined = np.concatenate(all_audio)
        # If still too short, repeat
        while len(combined) < 12800:  # minimum for embeddings
            combined = np.concatenate([combined, silence.copy(), combined])

        print(f"    {label}: {len(files)} files → {len(combined)} samples "
              f"({len(combined)/16000:.1f}s)")

        embeddings = audio_to_embeddings(combined)  # (n_emb, 96)
        print(f"    → {embeddings.shape[0]} embeddings")

        # 滑动窗口: 16 帧 × 96 维
        windows = []
        for start in range(0, embeddings.shape[0] - N_FRAMES + 1, 2):
            window = embeddings[start : start + N_FRAMES]
            windows.append(window.flatten())

        return windows

    # 正样本
    print(f"  提取正样本...")
    pos_windows = extract_from_dir(positive_dir, "positive")
    X_pos = np.array(pos_windows, dtype=np.float32) if pos_windows else np.empty((0, INPUT_FEATURES))

    # 负样本
    print(f"  提取负样本...")
    all_neg_windows = []
    for neg_dir in negative_dirs:
        if not neg_dir.exists():
            continue
        neg_windows = extract_from_dir(neg_dir, neg_dir.name)
        all_neg_windows.extend(neg_windows)

    X_neg = np.array(all_neg_windows, dtype=np.float32) if all_neg_windows else np.empty((0, INPUT_FEATURES))

    print(f"  正样本窗口: {len(X_pos)},  负样本窗口: {len(X_neg)}")
    return X_pos, X_neg


# ── PyTorch Model ───────────────────────────────────────────────────────────

def create_model(hidden_dim=32, n_blocks=3):
    """创建与 openWakeWord 兼容的唤醒词分类器"""
    import torch
    import torch.nn as nn

    class FCNBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.net(x) + x  # residual

    class WakeWordModel(nn.Module):
        def __init__(self, input_dim=1536, hidden_dim=32, n_blocks=3):
            super().__init__()
            self.input_layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            self.blocks = nn.ModuleList([FCNBlock(hidden_dim) for _ in range(n_blocks)])
            self.output_layer = nn.Linear(hidden_dim, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # openWakeWord 输入: (batch, 16, 96) — 不可 flatten, 模型内处理
            x = x.reshape(x.shape[0], -1)
            x = self.input_layer(x)
            for block in self.blocks:
                x = block(x)
            x = self.output_layer(x)
            return self.sigmoid(x)

    model = WakeWordModel(input_dim=INPUT_FEATURES, hidden_dim=hidden_dim, n_blocks=n_blocks)

    # 包装层: 接受 (batch, 16, 96) 输入 (与 openWakeWord 兼容)
    class WakeWordWrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            # openWakeWord 传入 (batch, 16, 96), 内部 flatten 为 (batch, 1536)
            return self.inner(x)

    return WakeWordWrapper(model)


# ── Training ─────────────────────────────────────────────────────────────────

def train_model(X_pos, X_neg, output_path: Path, quick: bool = False):
    """训练并导出模型"""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")

    # 平衡正负样本
    n_pos, n_neg = len(X_pos), len(X_neg)
    target = min(n_pos, n_neg) if not quick else min(n_pos, n_neg, 2000)
    print(f"  原始: 正={n_pos}, 负={n_neg}")

    if n_pos > target:
        X_pos = X_pos[np.random.choice(n_pos, target, replace=False)]
    if n_neg > target:
        X_neg = X_neg[np.random.choice(n_neg, target, replace=False)]

    print(f"  平衡后: 正={len(X_pos)}, 负={len(X_neg)}")

    # Train/val split (80/20)
    split_pos = int(len(X_pos) * 0.8)
    split_neg = int(len(X_neg) * 0.8)

    X_train = np.concatenate([X_pos[:split_pos], X_neg[:split_neg]])
    y_train = np.concatenate([np.ones(split_pos), np.zeros(split_neg)])
    X_val = np.concatenate([X_pos[split_pos:], X_neg[split_neg:]])
    y_val = np.concatenate([np.ones(len(X_pos) - split_pos), np.zeros(len(X_neg) - split_neg)])

    # Shuffle
    for arr in [X_train, y_train, X_val, y_val]:
        np.random.seed(42)
        idx = np.random.permutation(len(X_train))
        X_train, y_train = X_train[idx], y_train[idx]
        idx = np.random.permutation(len(X_val))
        X_val, y_val = X_val[idx], y_val[idx]

    print(f"  训练: {len(X_train)}, 验证: {len(X_val)}")

    # Reshape to (N, 16, 96) — openWakeWord standard input format
    X_train_3d = X_train.reshape(-1, N_FRAMES, EMBEDDING_DIM)
    X_val_3d = X_val.reshape(-1, N_FRAMES, EMBEDDING_DIM)

    X_train_t = torch.tensor(X_train_3d).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_val_t = torch.tensor(X_val_3d).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    # Create model
    model = create_model()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数: {n_params:,}")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

    n_epochs = 80 if not quick else 25
    batch_size = 256
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        perm = torch.randperm(len(X_train_t))

        for i in range(0, len(X_train_t), batch_size):
            idx = perm[i : i + batch_size]
            batch_x, batch_y = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_loss = criterion(val_out, y_val_t).item()
            val_pred = (val_out > 0.5).float()
            val_acc = (val_pred == y_val_t).float().mean().item()

            pos_mask = y_val_t == 1
            neg_mask = y_val_t == 0
            recall = (val_out[pos_mask] > 0.5).float().mean().item() if pos_mask.sum() > 0 else 0
            fp_rate = (val_out[neg_mask] > 0.5).float().mean().item() if neg_mask.sum() > 0 else 0

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"    Epoch {epoch+1:3d}: loss={total_loss/max(n_batches,1):.4f}, "
                f"val_loss={val_loss:.4f}, acc={val_acc:.3f}, recall={recall:.3f}, "
                f"fp={fp_rate:.4f}"
            )

        if patience_counter >= 20:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    print(f"  最佳 val_acc: {best_val_acc:.3f}")

    # Export ONNX (input shape: batch, 16, 96 — matches openWakeWord convention)
    onnx_path = output_path.with_suffix(".onnx")
    model.cpu().eval()
    dummy = torch.randn(1, N_FRAMES, EMBEDDING_DIM)  # (1, 16, 96)

    # Use opset 18 (newer PyTorch) or 13 fallback
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    size_kb = onnx_path.stat().st_size / 1024
    print(f"  导出: {onnx_path} ({size_kb:.0f} KB)")

    # 更新 config.yaml (仅更新 wake_word 下的 model_path)
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    if config_path.exists():
        lines = config_path.read_text().split("\n")
        in_wake_word = False
        new_lines = []
        for line in lines:
            if line.strip().startswith("wake_word:"):
                in_wake_word = True
            elif in_wake_word and not line.startswith("  "):
                in_wake_word = False
            if in_wake_word and line.lstrip().startswith("model_path:"):
                indent = line[:len(line) - len(line.lstrip())]
                new_lines.append(f"{indent}model_path: ./models/openwakeword/{MODEL_NAME}.onnx")
            else:
                new_lines.append(line)
        config_path.write_text("\n".join(new_lines))
        print(f"  已更新 config.yaml (仅 wake_word 部分)")

    return model, onnx_path


# ── Validation ────────────────────────────────────────────────────────────────

def validate_model(onnx_path: Path):
    """验证新训练的模型"""
    from openwakeword import Model

    print(f"\n  验证: {onnx_path}")

    # 1. 加载测试
    m = Model(wakeword_models=[str(onnx_path)], inference_framework="onnx")
    print("  ✓ openWakeWord 加载成功")
    model_name = list(m.models.keys())[0]
    print(f"    模型名: {model_name}")

    # 2. 对 "小智" TTS 音频的响应
    audio_xz = _gen_tts_raw("小智", n_variations=5)
    if len(audio_xz) > 0:
        scores = []
        for i in range(0, len(audio_xz) - 1280, 1280):
            chunk = audio_xz[i : i + 1280]
            if len(chunk) < 1280:
                chunk = np.pad(chunk, (0, 1280 - len(chunk)))
            pred = m.predict(chunk)
            scores.append(list(pred.values())[0])
        real = scores[5:]  # 跳过前 5 帧强制归零
        max_s = max(real) if real else 0
        print(f"  '小智' → max={max_s:.4f} {'✓ 正常响应!' if max_s > 0.3 else '⚠ 需调整阈值' if max_s > 0.1 else '✗ 无响应'}")

    # 3. 对噪声的误触发测试
    noise = np.random.randint(-10000, 10000, 32000, dtype=np.int16)
    scores_n = []
    for i in range(0, len(noise) - 1280, 1280):
        pred = m.predict(noise[i : i + 1280])
        scores_n.append(list(pred.values())[0])
    real_n = scores_n[5:]
    max_n = max(real_n) if real_n else 0
    print(f"  噪声 → max={max_n:.4f} {'✓ 无误触发' if max_n < 0.3 else '⚠ 有误触发风险'}")

    # 4. 对 "你好" 的误触发测试
    audio_other = _gen_tts_raw("你好", n_variations=3)
    if len(audio_other) > 0:
        scores_o = []
        for i in range(0, len(audio_other) - 1280, 1280):
            pred = m.predict(audio_other[i : i + 1280])
            scores_o.append(list(pred.values())[0])
        real_o = scores_o[5:]
        max_o = max(real_o) if real_o else 0
        print(f"  '你好' → max={max_o:.4f} {'✓ 无误触发' if max_o < 0.3 else '⚠ 有误触发风险'}")

    return m


def _gen_tts_raw(phrase: str, n_variations: int = 3) -> np.ndarray:
    """生成 TTS 音频, 返回拼接的 int16 数组"""
    all_audio, silence = [], np.zeros(4000, dtype=np.int16)
    for i in range(n_variations):
        speed = SPEED_RANGES[i % len(SPEED_RANGES)]
        result = subprocess.run(
            ["piper", "--model", str(PIPER_VOICE), "--output-raw",
             "--length-scale", str(speed)],
            input=phrase.encode("utf-8"),
            capture_output=True, timeout=30,
        )
        if result.returncode == 0 and len(result.stdout) > 0:
            raw = np.frombuffer(result.stdout, dtype=np.int16)
            resampled = resample_poly(raw.astype(float), 16000, 22050).astype(np.int16)
            all_audio.extend([silence.copy(), resampled])
    return np.concatenate(all_audio) if all_audio else np.array([], dtype=np.int16)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="训练 xiao_zhi 唤醒词模型")
    parser.add_argument("--quick", action="store_true", help="快速模式 (少量样本)")
    parser.add_argument("--generate-only", action="store_true", help="只生成训练数据")
    parser.add_argument("--output", default=None, help="输出路径")
    args = parser.parse_args()

    n_positive = 30 if args.quick else 200
    n_neg_phrase = 12 if args.quick else 60
    n_noise = 15 if args.quick else 100
    n_silence = 10 if args.quick else 40

    data_dir = PROJECT_ROOT / "data" / "training" / "xiao_zhi"
    pos_dir = data_dir / "positive"
    neg_phrase_dir = data_dir / "negative" / "phrases"
    noise_dir = data_dir / "negative" / "noise"
    silence_dir = data_dir / "negative" / "silence"

    print("=" * 60)
    print("训练 xiao_zhi 唤醒词模型")
    print("=" * 60)

    # [1/4] Generate
    print(f"\n[1/4] 生成训练数据...")
    print(f"  正样本 '小智': {n_positive}")
    pos = generate_tts_clips(WAKE_WORD, pos_dir, n_variations=n_positive)
    print(f"  → {len(pos)} 个")

    print(f"  负样本 (其他短语): {n_neg_phrase}")
    neg_phrase_dir.mkdir(parents=True, exist_ok=True)
    all_neg = []
    for phrase in NEGATIVE_PHRASES[:n_neg_phrase]:
        n_per = max(3, 100 // n_neg_phrase)
        clips = generate_tts_clips(phrase, neg_phrase_dir, n_variations=n_per)
        all_neg.extend(clips)
    print(f"  → {len(all_neg)} 个")

    print(f"  噪声: {n_noise}")
    noise_clips = generate_noise_clips(noise_dir, n=n_noise)
    print(f"  → {len(noise_clips)} 个")

    print(f"  静音: {n_silence}")
    silence_clips = generate_silence_clips(silence_dir, n=n_silence)
    print(f"  → {len(silence_clips)} 个")

    if args.generate_only:
        print("\n✓ 数据生成完成")
        return 0

    # [2/4] Extract features
    print("\n[2/4] 提取特征...")
    t0 = time.time()
    X_pos, X_neg = prepare_dataset(pos_dir, [neg_phrase_dir, noise_dir, silence_dir])
    print(f"  耗时: {time.time() - t0:.1f}s")

    if len(X_pos) < 10 or len(X_neg) < 10:
        print(f"  ✗ 样本不足 (正={len(X_pos)}, 负={len(X_neg)})")
        return 1

    # [3/4] Train
    print("\n[3/4] 训练...")
    t0 = time.time()
    output_path = Path(args.output) if args.output else (OUTPUT_DIR / MODEL_NAME)
    model, onnx_path = train_model(X_pos, X_neg, output_path, quick=args.quick)
    print(f"  耗时: {time.time() - t0:.1f}s")

    # [4/4] Validate
    print("\n[4/4] 验证...")
    validate_model(onnx_path)

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"  模型: {onnx_path}")
    print(f"  唤醒词: {WAKE_WORD}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
