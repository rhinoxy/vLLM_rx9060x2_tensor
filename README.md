# Dual RX 9060 XT vLLM ROCm Setup (gAI-LLM)

AMD Radeon RX 9060 XT × 2（合計 32GB VRAM, `gfx1200`）環境において、ROCm版 vLLM を用いて大規模言語モデル（Qwen 3.8 27B, Gemma 4 26B 等）を Tensor Parallelism (TP=2) で高速推論するためのセットアップおよびパッチ集です。

## 🎯 主な特徴

- **Dual GPU Tensor Parallelism (TP=2)**: 2基の RX 9060 XT (16GB × 2) にモデルウェイトを分散配置。
- **GGUF 混合量子化対応**: `vllm_gguf_plugin` による GGUF ロードと Triton カーネル最適化。
- **gfx1200 互換パッチ**: ROCm 7.x / Navi 環境での Triton カーネルおよびモデルローダーの非互換を解決。

---

## 🛠️ パッチ構成 (`vllm_rocm/patch_transformers.py`)

1. **Patch 1〜5**: モデルローダー、トークナイザー、KVキャッシュ初期化の安定化。
2. **Patch 6**:
   - `qweight_type` の型自動判定・補正（ブロック整合性の担保）。
   - Triton MMQ カーネル不整合時の DEQUANT フォールバック。
3. **Patch 7**:
   - **Gemma 4 テキスト専用 GGUF 対応**: `gemma4_mm.py` の `vision_config=None` アクセス例外を回避するガード処理。

---

## 🚀 起動スクリプト (`scripts/`)

SurrealDB（ポート 8000）等の他サービスと競合しないよう、vLLM は **ポート 8001** でリッスンします。

### 1. Qwen 3.8 27B の起動
```bash
bash scripts/run_vllm_qwen.sh
```

### 2. Gemma 4 26B の起動
```bash
bash scripts/run_vllm_gemma.sh
```

### 3. サーバーの停止
```bash
bash scripts/stop_vllm.sh
```

---

## 📁 ディレクトリ構成

- `models/`: HuggingFace 形式のアーキテクチャ設定（`config.json`）およびチャットテンプレート（`template_qwen.jinja`）。大容量モデル実体は `.gitignore` で除外。
- `scripts/`: vLLM サーバースクリプト群。
- `vllm_rocm/`: カスタム Dockerfile および `patch_transformers.py`。
- `anything-llm/`: Anything-LLM 連携用の Docker Compose 設定。

---

## 📜 ライセンス
MIT License
