import transformers.modeling_gguf_pytorch_utils as m
import inspect
import os

target_file = inspect.getfile(m)
with open(target_file, "r") as f:
    code = f.read()

patch_code = """
# --- Patch for qwen35 GGUF support ---
if 'qwen35' not in GGUF_CONFIG_MAPPING:
    GGUF_CONFIG_MAPPING['qwen35'] = GGUF_CONFIG_MAPPING.get('qwen3', {})
    for sec in ['config', 'tokenizer', 'tokenizer_config']:
        if 'qwen3' in GGUF_TO_TRANSFORMERS_MAPPING.get(sec, {}):
            GGUF_TO_TRANSFORMERS_MAPPING[sec]['qwen35'] = GGUF_TO_TRANSFORMERS_MAPPING[sec]['qwen3']
    if 'qwen35' not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append('qwen35')
# -------------------------------------
"""

if "# --- Patch for qwen35" not in code:
    with open(target_file, "a") as f:
        f.write(patch_code)
    print("Patch applied to", target_file)
else:
    print("Already patched")

try:
    import vllm_gguf_plugin.weights_adapter.default as d
    d_file = inspect.getfile(d)
    with open(d_file, "r") as f:
        d_code = f.read()
    old_str = "if value == model_type:"
    new_str = "if value == model_type or value == model_type.replace('qwen3_5', 'qwen35'):"
    if old_str in d_code:
        d_code = d_code.replace(old_str, new_str)
        print("Patched vllm_gguf_plugin model_type")

    old_mm = 'is_multimodal = (\n            hasattr(config, "vision_config") and config.vision_config is not None\n        )'
    new_mm = 'is_multimodal = (\n            hasattr(config, "vision_config")\n            and config.vision_config is not None\n            and not any("CausalLM" in a for a in getattr(config, "architectures", []))\n        )'
    if old_mm in d_code:
        d_code = d_code.replace(old_mm, new_mm)
        print("Patched vllm_gguf_plugin is_multimodal")

    target_insertion = 'if model_type in ("qwen2_moe", "qwen3_moe"):'
    qwen35_map = '''if model_type in ("qwen3_5", "qwen35"):
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.ssm_dt.bias"] = (
                    f"model.layers.{idx}.linear_attn.dt_bias"
                )
        if model_type in ("qwen2_moe", "qwen3_moe"):'''
    if target_insertion in d_code and 'model_type in ("qwen3_5", "qwen35"):' not in d_code:
        d_code = d_code.replace(target_insertion, qwen35_map)
        print("Patched vllm_gguf_plugin qwen35 ssm_dt.bias mapping")

    with open(d_file, "w") as f:
        f.write(d_code)
    print("Saved vllm_gguf_plugin default.py")
except Exception as e:
    print("Failed to patch vllm_gguf_plugin default.py:", e)

try:
    import vllm_gguf_plugin.quantization.params as p
    p_file = inspect.getfile(p)
    with open(p_file, "r") as f:
        p_code = f.read()

    old_loader = """    def _gguf_weight_type_loader_v2(param, loaded_weight, loaded_shard_id=None):
        if loaded_shard_id is None and hasattr(param, "_store"):
            param._store(loaded_weight)
            return
        base_loader(param, loaded_weight, loaded_shard_id)"""

    new_loader = """    def _gguf_weight_type_loader_v2(param, loaded_weight, loaded_shard_id=None):
        if hasattr(param, "_store"):
            param._store(loaded_weight, loaded_shard_id)
            return
        base_loader(param, loaded_weight, loaded_shard_id)"""

    if old_loader in p_code:
        p_code = p_code.replace(old_loader, new_loader)
        with open(p_file, "w") as f:
            f.write(p_code)
        print("Patched vllm_gguf_plugin params.py _gguf_weight_type_loader_v2")
    else:
        print("params.py already patched or pattern not found")
except Exception as e:
    print("Failed to patch vllm_gguf_plugin params.py:", e)

try:
    import vllm.model_executor.models.qwen3_5 as q35
    q35_file = inspect.getfile(q35)
    with open(q35_file, "r") as f:
        q35_code = f.read()

    old_embed = """        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )"""

    new_embed = """        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )"""

    if old_embed in q35_code:
        q35_code = q35_code.replace(old_embed, new_embed)
        with open(q35_file, "w") as f:
            f.write(q35_code)
        print("Patched vllm qwen3_5 embed_tokens with quant_config and prefix")
    else:
        print("qwen3_5 embed_tokens already patched or pattern not found")
except Exception as e:
    print("Failed to patch vllm qwen3_5:", e)

try:
    import vllm.model_executor.layers.mamba.mamba_mixer2 as mm2
    mm2_file = inspect.getfile(mm2)
    with open(mm2_file, "r") as f:
        mm2_code = f.read()

    old_mamba_loader = """            param.data[
                boundary : (boundary + take), ...  # type: ignore[misc]
            ] = loaded_weight[
                loaded_start_idx : (
                    loaded_start_idx + take
                )  # type: ignore[misc]
            ]  # type: ignore[misc]"""

    new_mamba_loader = """            _src_part = loaded_weight[
                loaded_start_idx : (
                    loaded_start_idx + take
                )  # type: ignore[misc]
            ]
            _tgt_part = param.data[
                boundary : (boundary + take), ...  # type: ignore[misc]
            ]
            if _src_part.ndim != _tgt_part.ndim and _src_part.numel() == _tgt_part.numel():
                _src_part = _src_part.view_as(_tgt_part)
            _tgt_part.copy_(_src_part)"""

    if old_mamba_loader in mm2_code:
        mm2_code = mm2_code.replace(old_mamba_loader, new_mamba_loader)
        with open(mm2_file, "w") as f:
            f.write(mm2_code)
        print("Patched vllm mamba_mixer2 loader tensor reshape compatibility")
    else:
        print("mamba_mixer2 already patched or pattern not found")
except Exception as e:
    print("Failed to patch vllm mamba_mixer2:", e)

# ============================================================
# Patch 6: vllm_gguf_plugin/quantization/linear.py
#   _fused_mul_mat_gguf: 
#   1. 入力次元 x.shape[-1] と qweight.shape[1] から実際の量子化型を自動補正
#   2. ブロック境界不整合時の安全な DEQUANT フォールバック
# ============================================================
try:
    import vllm_gguf_plugin.quantization.linear as linear_mod
    import inspect

    linear_file = inspect.getfile(linear_mod)
    with open(linear_file, "r") as f:
        linear_code = f.read()

    # _fused_mul_mat_gguf 関数の先頭で型補正を行うパッチ
    func_header = "def _fused_mul_mat_gguf(\n    x: torch.Tensor, qweight: torch.Tensor, qweight_type: int\n) -> torch.Tensor:"
    type_correction = '''def _fused_mul_mat_gguf(
    x: torch.Tensor, qweight: torch.Tensor, qweight_type: int
) -> torch.Tensor:
    # --- Patch 6-1: qweight_type の自動補正 (シャード混在・誤判定対策) ---
    if hasattr(qweight, "shape") and len(qweight.shape) == 2 and hasattr(x, "shape") and len(x.shape) >= 2:
        import gguf as _gguf
        _in_dim = x.shape[-1]
        _curr_bs, _curr_ts = _gguf.GGML_QUANT_SIZES.get(qweight_type, (None, None))
        if _curr_bs is not None and _curr_ts is not None:
            _expected_bytes = (_in_dim // _curr_bs) * _curr_ts
            if qweight.shape[1] != _expected_bytes:
                for _qt, (_bs, _ts) in _gguf.GGML_QUANT_SIZES.items():
                    if _bs and (_in_dim % _bs == 0):
                        if qweight.shape[1] == (_in_dim // _bs) * _ts:
                            qweight_type = _qt
                            break
    # --- end Patch 6-1 ---'''

    if func_header in linear_code:
        linear_code = linear_code.replace(func_header, type_correction)
        print("Patch 6-1 applied: qweight_type auto-correction")

    # さらに MMQ_QUANT_TYPES で万一のブロック非整合時のフォールバック
    old_mmq = """    elif qweight_type in MMQ_QUANT_TYPES:
        y = ops.ggml_mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])"""
    new_mmq = """    elif qweight_type in MMQ_QUANT_TYPES:
        from vllm_gguf_plugin.triton.gemm.utils import BLOCK_BYTES_BY_TYPE
        _blk = BLOCK_BYTES_BY_TYPE.get(qweight_type, 1)
        if qweight.shape[1] % _blk != 0:
            import gguf as _gguf
            _block_size, _type_size = _gguf.GGML_QUANT_SIZES[qweight_type]
            _shape = (qweight.shape[0], qweight.shape[1] // _type_size * _block_size)
            _weight = ops.ggml_dequantize(qweight, qweight_type, *_shape, x.dtype)
            y = x @ _weight.T
        else:
            y = ops.ggml_mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])"""

    if old_mmq in linear_code:
        linear_code = linear_code.replace(old_mmq, new_mmq)
        print("Patch 6-2 applied: MMQ fallback to DEQUANT")

    with open(linear_file, "w") as f:
        f.write(linear_code)
    print("Patch 6 written to linear.py successfully")
except Exception as e:
    print("Failed to apply Patch 6:", e)


# ============================================================
# Patch 7: Gemma4 text-only GGUF — vision_config=None ガード
# vLLM の gemma4_mm.py は get_mm_max_tokens_per_item() で
# config.vision_config.default_output_length に無条件アクセスする。
# テキスト専用 GGUF では vision_config=None のためクラッシュする。
# vision_config が None の場合は空dict (テキストのみ) を返すよう修正。
# ============================================================
try:
    import importlib.util, pathlib

    _gemma4_mm_spec = importlib.util.find_spec("vllm.model_executor.models.gemma4_mm")
    if _gemma4_mm_spec:
        _gemma4_mm_file = pathlib.Path(_gemma4_mm_spec.origin)
        _code = _gemma4_mm_file.read_text()

        _old = (
            "        tokens_per_image = config.vision_config.default_output_length"
        )
        _new = (
            "        if config.vision_config is None:\n"
            "            return {}\n"
            "        tokens_per_image = config.vision_config.default_output_length"
        )

        if _old in _code:
            _code = _code.replace(_old, _new)
            _gemma4_mm_file.write_text(_code)
            print("Patch 7 applied: Gemma4 vision_config=None guard")
        elif "if config.vision_config is None:" in _code:
            print("Patch 7 already applied")
        else:
            print("Patch 7: pattern not found in gemma4_mm.py")
    else:
        print("Patch 7: gemma4_mm module not found (skip)")
except Exception as e:
    print("Failed to apply Patch 7:", e)
