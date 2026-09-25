# Dual RX 9060 XT vLLM ROCm Setup (`vLLM_rx9060x2_tensor`)

AMD Radeon RX 9060 XT × 2（合計 32GB VRAM, `gfx1200` / Navi）環境において、ROCm版 vLLM を用いて大規模言語モデル（Qwen 3.8 27B, Gemma 4 26B 等）を Tensor Parallelism (TP=2) で高速・安全に推論するためのセットアップおよびパッチ集です。

---

## 🎯 アーキテクチャと改善ハイライト

- **ビルド時パッチ統合（Production-grade）**:
  - 従来コンテナ起動時に行っていた `site-packages` の直接書き換えを廃止。
  - `vllm_rocm/Dockerfile` のビルド時に全10パッチを自動適用＆厳格にアサート検証し、`rocm-vllm:custom-gfx1200` イメージとして固定化。
- **最小権限セキュリティ（No `--privileged`）**:
  - ホスト権限を丸ごと与える `--privileged` や不要な `sudo` グループを撤廃。
  - `--device=/dev/kfd --device=/dev/dri --group-add video --group-add "$RENDER_GID"` の最小限のデバイスアクセスで動作（ホストの `render` グループ GID をスクリプト側で自動解決）。
- **Navi (gfx1200 / RDNA4) 最適化・デッドロック防止設定**:
  - **`--ipc=host` 維持**: TP=2 でのマルチGPU間 PyTorch 共有メモリ（Shared Memory）通信の枯渇・ハングを防止。
  - **`--enforce-eager` 維持**: gfx1200 において不安定な HIP Graph キャプチャ（0%ハング）を回避し、Triton カスタムカーネルを安定稼働。
  - **`HIP_VISIBLE_DEVICES=0,1`**: ホスト側の内蔵 iGPU 誤認識による RCCL デッドロックを防止。
  - **`NCCL_P2P_DISABLE=1`**: PCIe 接続（XGMI 非搭載）環境下での P2P DMA 通信不整合によるマルチGPUハングを回避。
  - **`NCCL_PROTO=Simple`**: RCCL の `LL` / `LL128` プロトコル（PCIe 上で不安定・デッドロック要因）をバイパスし、最も安定した `Simple` プロトコルに固定。
  - **`VLLM_ROCM_USE_AITER=0`**: RDNA4 未対応の AITer 最適化パスを明示的にバイパス。
- **モデル実体パスの抽象化**:
  - Ollama の sha256 blob 直打ちを廃止し、`models/<model-name>/model.gguf` による一貫したパス管理を採用。

---

## 🛠️ パッチ構成 (`vllm_rocm/patch_transformers.py`)

| パッチ | 対象モジュール | 解決する問題・機能 |
|---|---|---|
| **Patch 1** | `transformers.modeling_gguf_pytorch_utils` | `qwen35` アーキテクチャの GGUF ローダー登録 |
| **Patch 2** | `vllm_gguf_plugin.weights_adapter.default` | `qwen35` のモデル判定、マルチモーダル誤認防止、`ssm_dt.bias` マッピング |
| **Patch 3** | `vllm_gguf_plugin.quantization.params` | `loaded_shard_id` を受け取る GGUF シャードローダー対応 |
| **Patch 4** | `vllm.model_executor.models.qwen3_5` | `VocabParallelEmbedding` への `quant_config` とプレフィックス伝播 |
| **Patch 5** | `vllm.model_executor.layers.mamba.mamba_mixer2` | 重みテンソルの形状不一致時における自動 `view_as` 適合 |
| **Patch 6** | `vllm_gguf_plugin.quantization.linear` | `qweight_type` の自動判定・補正、ブロック非整合時の Triton DEQUANT 安全フォールバック |
| **Patch 7** | `vllm.model_executor.models.gemma4_mm` | Gemma 4 テキスト専用 GGUF における `vision_config=None` ガード（`get_mm_max_tokens_per_item` および `vision_tower` 初期化） |
| **Patch 8** | `vllm_gguf_plugin.weights_adapter.default` | Gemma 4 重み読み込み時の `model.language_model.` プレフィックス除去および `router.scale` / `router.per_expert_scale` のマッピング |
| **Patch 9** | `vllm.model_executor.models.gemma4` | Gemma 4 の不均一 `head_dim` (sliding: 256 / full: 512) 解決、`k_eq_v` フルアテンション時の KV ヘッド数整合および動的 QKV split ガード |
| **Patch 10** | `vllm.model_executor.models.gemma4` & `vllm_gguf_plugin` | Gemma 4 MoE エキスパート重みの GGUF 名解決、`_gguf_moe_weight_type_loader` デフォルト引数対応、および `_qweight` フォールバック |

---

## 📊 モデル稼働状況 & 既知の課題

- **Gemma 4 (26B-A4W4 / GGUF Q4_K_M)**:
  - **ロード完了**: 重みロード（約102秒）およびレイヤー構築は Patch 7〜9 により正常パス。
  - **現在対応中**: 推論プロファイリング時の MoE パラメータ初期化 (`fused_moe_gguf` における `w13_qweight` / `w2_qweight` のマテリアライズとマッピング) の解決作業中。
- **Qwen 3.8 (27B GGUF)**:
  - SSM (State Space Model) / GDN Triton カーネル (`fused_recurrent_gated_delta_rule_packed_decode_kernel`) の gfx1200 互換性（NaN出力）の課題を調査中。

---

## 🚀 運用スクリプト (`scripts/`)

### 1. モデルの切り替え（推奨）
32GB VRAM 環境のため、Qwen 3.8 (27B) と Gemma 4 (26B) は排他起動（1モデルずつ）となります。
```bash
# Qwen 3.8 27B の起動 (Port 8001)
bash scripts/switch_model.sh qwen

# Gemma 4 26B の起動 (Port 8001)
bash scripts/switch_model.sh gemma

# 状態確認・ヘルスチェック
bash scripts/switch_model.sh status

# 停止
bash scripts/switch_model.sh stop
```

### 2. 個別起動
```bash
# Qwen 3.8 27B
bash scripts/run_vllm_qwen.sh

# Gemma 4 26B
bash scripts/run_vllm_gemma.sh

# 停止
bash scripts/stop_vllm.sh
```

---

## 📦 Docker イメージの再ビルド

環境更新やパッチの再検証を行いたい場合：
```bash
cd vllm_rocm
docker build -t rocm-vllm:custom -t rocm-vllm:custom-gfx1200 .
```

---

## 📜 ライセンス
MIT License
