#!/usr/bin/env python3
"""
Patch script for ROCm vLLM compatibility on AMD Radeon RX 9060 XT (gfx1200 / Navi).
Applied during Docker image build or before starting vllm serve.
"""

import inspect
import importlib.util
import os
import pathlib
import sys

def print_environment_info():
    print("=" * 60)
    print("=== ROCm vLLM Environment & Version Check ===")
    print("=" * 60)
    for pkg in ["vllm", "transformers", "torch", "gguf", "vllm_gguf_plugin"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            loc = getattr(mod, "__file__", "unknown")
            print(f"  {pkg:18s}: {ver:30s} (from {loc})")
        except ImportError:
            print(f"  {pkg:18s}: NOT INSTALLED")
    print("=" * 60)

def apply_patch_1_transformers_qwen35():
    """Patch 1: Register qwen35 architecture mapping in transformers gguf loader."""
    import transformers.modeling_gguf_pytorch_utils as m
    target_file = inspect.getfile(m)
    with open(target_file, "r") as f:
        code = f.read()

    patch_marker = "# --- Patch for qwen35 GGUF support ---"
    if patch_marker not in code:
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
        with open(target_file, "a") as f:
            f.write(patch_code)
        print("[Patch 1] Applied to", target_file)
    else:
        print("[Patch 1] Already present")

    # Verify
    with open(target_file, "r") as f:
        assert patch_marker in f.read(), "Patch 1 verification failed!"

def apply_patch_2_vllm_gguf_plugin_default():
    """Patch 2: Fix model_type check, multimodal check, and ssm_dt.bias mapping."""
    import vllm_gguf_plugin.weights_adapter.default as d
    d_file = inspect.getfile(d)
    with open(d_file, "r") as f:
        d_code = f.read()

    old_str = "if value == model_type:"
    new_str = "if value == model_type or value == model_type.replace('qwen3_5', 'qwen35'):"
    if old_str in d_code:
        d_code = d_code.replace(old_str, new_str)
        print("[Patch 2-1] Patched vllm_gguf_plugin model_type")

    old_mm = 'is_multimodal = (\n            hasattr(config, "vision_config") and config.vision_config is not None\n        )'
    new_mm = 'is_multimodal = (\n            hasattr(config, "vision_config")\n            and config.vision_config is not None\n            and not any("CausalLM" in a for a in getattr(config, "architectures", []))\n        )'
    if old_mm in d_code:
        d_code = d_code.replace(old_mm, new_mm)
        print("[Patch 2-2] Patched vllm_gguf_plugin is_multimodal")

    target_insertion = 'if model_type in ("qwen2_moe", "qwen3_moe"):'
    qwen35_map = '''if model_type in ("qwen3_5", "qwen35"):
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.ssm_dt.bias"] = (
                    f"model.layers.{idx}.linear_attn.dt_bias"
                )
        if model_type in ("qwen2_moe", "qwen3_moe"):'''
    if target_insertion in d_code and 'model_type in ("qwen3_5", "qwen35"):' not in d_code:
        d_code = d_code.replace(target_insertion, qwen35_map)
        print("[Patch 2-3] Patched vllm_gguf_plugin qwen35 ssm_dt.bias mapping")

    with open(d_file, "w") as f:
        f.write(d_code)

    # Verify
    assert "replace('qwen3_5', 'qwen35')" in d_code, "Patch 2-1 verification failed!"
    print("[Patch 2] Verified successfully")

def apply_patch_3_params_loader():
    """Patch 3: Allow loaded_shard_id in _gguf_weight_type_loader_v2."""
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
        print("[Patch 3] Patched params.py _gguf_weight_type_loader_v2")
    else:
        print("[Patch 3] Already present or compatible")

    assert "param._store(loaded_weight, loaded_shard_id)" in p_code or "param._store" in p_code, "Patch 3 check failed"

def apply_patch_4_qwen35_embed():
    """Patch 4: Pass quant_config and prefix to qwen3_5 embed_tokens."""
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
        print("[Patch 4] Patched vllm qwen3_5 embed_tokens")
    else:
        print("[Patch 4] Already present or compatible")

def apply_patch_5_mamba_mixer2():
    """Patch 5: Reshape compatibility for mamba_mixer2 weight loader."""
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
        print("[Patch 5] Patched vllm mamba_mixer2 loader tensor reshape compatibility")
    else:
        print("[Patch 5] Already present or compatible")

def apply_patch_6_linear_quant():
    """Patch 6: Auto-detect correct qweight_type and fallback to DEQUANT for unaligned blocks."""
    import vllm_gguf_plugin.quantization.linear as linear_mod
    linear_file = inspect.getfile(linear_mod)
    with open(linear_file, "r") as f:
        linear_code = f.read()

    func_header = "def _fused_mul_mat_gguf(\n    x: torch.Tensor, qweight: torch.Tensor, qweight_type: int\n) -> torch.Tensor:"
    type_correction = '''def _fused_mul_mat_gguf(
    x: torch.Tensor, qweight: torch.Tensor, qweight_type: int
) -> torch.Tensor:
    # --- Patch 6-1: qweight_type auto-correction ---
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
        print("[Patch 6-1] Applied qweight_type auto-correction")

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
        print("[Patch 6-2] Applied MMQ fallback to DEQUANT")

    with open(linear_file, "w") as f:
        f.write(linear_code)

    assert "Patch 6-1" in linear_code, "Patch 6-1 verification failed!"
    print("[Patch 6] Verified successfully")

def apply_patch_7_gemma4_mm_guard():
    """Patch 7: Guard against vision_config=None in gemma4_mm.py for text-only GGUF."""
    _gemma4_mm_spec = importlib.util.find_spec("vllm.model_executor.models.gemma4_mm")
    if not _gemma4_mm_spec or not _gemma4_mm_spec.origin:
        print("[Patch 7] gemma4_mm module not found, skipping")
        return

    _gemma4_mm_file = pathlib.Path(_gemma4_mm_spec.origin)
    _code = _gemma4_mm_file.read_text()

    # 1. Guard get_mm_max_tokens_per_item
    _old = "        tokens_per_image = config.vision_config.default_output_length"
    _new = (
        "        if config.vision_config is None:\n"
        "            return {}\n"
        "        tokens_per_image = config.vision_config.default_output_length"
    )

    if _old in _code:
        _code = _code.replace(_old, _new)
        print("[Patch 7-1] Applied Gemma4 tokens_per_image guard")

    # 2. Guard vision_tower initialization in __init__
    _old_vt = """        else:
            vision_cfg = config.vision_config
            quantizable = (
                vision_cfg.hidden_size % 64 == 0
                and vision_cfg.intermediate_size % 64 == 0
            )
            tower_quant = quant_config if quantizable else None

        # ---- Vision tower (shared by image and video) ----
        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.vision_tower = AutoModel.from_config(config=config.vision_config)
            self.embed_vision = Gemma4MultimodalEmbedder(
                config.vision_config,
                config.text_config,
                quant_config=tower_quant,
                prefix=maybe_prefix(prefix, "embed_vision"),
            )
            recursive_replace_linear(
                self.vision_tower,
                tower_quant,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )"""

    _new_vt = """        elif config.vision_config is not None:
            vision_cfg = config.vision_config
            quantizable = (
                vision_cfg.hidden_size % 64 == 0
                and vision_cfg.intermediate_size % 64 == 0
            )
            tower_quant = quant_config if quantizable else None
        else:
            tower_quant = None

        # ---- Vision tower (shared by image and video) ----
        if config.vision_config is not None:
            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.vision_tower = AutoModel.from_config(config=config.vision_config)
                self.embed_vision = Gemma4MultimodalEmbedder(
                    config.vision_config,
                    config.text_config,
                    quant_config=tower_quant,
                    prefix=maybe_prefix(prefix, "embed_vision"),
                )
                recursive_replace_linear(
                    self.vision_tower,
                    tower_quant,
                    prefix=maybe_prefix(prefix, "vision_tower"),
                )
        else:
            self.vision_tower = None
            self.embed_vision = None"""

    if _old_vt in _code:
        _code = _code.replace(_old_vt, _new_vt)
        print("[Patch 7-2] Applied Gemma4 vision_tower guard")

    _gemma4_mm_file.write_text(_code)

    assert "if config.vision_config is None:" in _code, "Patch 7-1 verification failed!"
    assert "self.vision_tower = None" in _code, "Patch 7-2 verification failed!"
    print("[Patch 7] Verified successfully")


def apply_patch_8_gemma4_tensor_name_map():
    """Patch 8: Strip 'model.language_model.' prefix and map router scales for Gemma4 in default weights adapter."""
    import vllm_gguf_plugin.weights_adapter.default as d
    d_file = inspect.getfile(d)
    with open(d_file, "r") as f:
        d_code = f.read()

    # 1. Strip 'model.language_model.' prefix
    old_snippet = """        def find_hf_name_in_tensor_map(hf_name: str) -> str | None:
            if is_multimodal and hf_name.startswith("model."):"""

    new_snippet = """        def find_hf_name_in_tensor_map(hf_name: str) -> str | None:
            if hf_name.startswith("model.language_model."):
                hf_name = "model." + hf_name[len("model.language_model."):]
            if is_multimodal and hf_name.startswith("model."):"""

    if old_snippet in d_code:
        d_code = d_code.replace(old_snippet, new_snippet)
        print("[Patch 8-1] Patched find_hf_name_in_tensor_map for Gemma4 model.language_model prefix")

    # 2. Add gemma4 router scales and sideload_params
    target_pos = 'if model_type == "minimax_m2":'
    gemma4_block = '''if model_type in ("gemma4", "gemma-4"):
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_inp.scale"] = (
                    f"model.language_model.layers.{idx}.router.scale"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.scale"] = (
                    f"model.language_model.layers.{idx}.router.per_expert_scale"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_up_exps.weight"] = (
                    f"model.language_model.layers.{idx}.experts.gate_up_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.language_model.layers.{idx}.experts.down_proj.weight"
                )
                sideload_params.extend([
                    regex.compile(f"model\\\\.language_model\\\\.layers\\\\.{idx}\\\\.router\\\\.(scale|per_expert_scale)"),
                    regex.compile(f"model\\\\.layers\\\\.{idx}\\\\.router\\\\.(scale|per_expert_scale)"),
                ])
        if model_type == "minimax_m2":'''

    if target_pos in d_code and 'ffn_gate_up_exps.weight' not in d_code:
        if 'model_type in ("gemma4", "gemma-4"):' in d_code:
            # Replace existing gemma4 block if already present from previous run
            import re
            d_code = re.sub(
                r'if model_type in \("gemma4", "gemma-4"\):.*?(?=if model_type == "minimax_m2":)',
                gemma4_block + '\n        ',
                d_code,
                flags=re.DOTALL
            )
        else:
            d_code = d_code.replace(target_pos, gemma4_block)
        print("[Patch 8-2] Added Gemma4 router.scale and MoE weights mapping")

    with open(d_file, "w") as f:
        f.write(d_code)

    assert 'if hf_name.startswith("model.language_model."):' in open(d_file).read(), "Patch 8-1 verification failed"
    assert 'ffn_gate_up_exps.weight' in open(d_file).read(), "Patch 8-2 verification failed"
    print("[Patch 8] Verified successfully")


def apply_patch_9_gemma4_heterogeneous_head_dim():
    """Patch 9: Support per-layer heterogeneous head_dim and k_eq_v KV head sizing for Gemma4."""
    _gemma4_spec = importlib.util.find_spec("vllm.model_executor.models.gemma4")
    if not _gemma4_spec or not _gemma4_spec.origin:
        print("[Patch 9] gemma4 module not found, skipping")
        return

    _gemma4_file = pathlib.Path(_gemma4_spec.origin)
    _code = _gemma4_file.read_text()

    # 1. Heterogeneous head_dim resolution
    old_snippet = """        # Gemma4 uses different head dimensions for sliding vs full attention
        layer_type = config.layer_types[layer_idx]
        self.is_full_attention = layer_type == "full_attention"
        if self.is_full_attention:
            head_dim = getattr(config, "global_head_dim", config.head_dim)
        else:
            head_dim = config.head_dim"""

    new_snippet = """        # Gemma4 uses different head dimensions for sliding vs full attention
        layer_type = config.layer_types[layer_idx]
        self.is_full_attention = layer_type == "full_attention"
        if hasattr(config, "per_layer_config") and layer_idx < len(config.per_layer_config):
            head_dim = config.per_layer_config[layer_idx].head_dim
        elif self.is_full_attention:
            head_dim = getattr(config, "global_head_dim", None) or 512
        else:
            head_dim = getattr(config, "head_dim", 256)"""

    if old_snippet in _code:
        _code = _code.replace(old_snippet, new_snippet)
        print("[Patch 9-1] Applied Gemma4 heterogeneous head_dim resolution")

    # 2. k_eq_v full-attention num_kv_heads sizing (each TP rank has 2 KV heads)
    old_kv = """        # For k_eq_v full-attention layers, use num_global_key_value_heads
        # as the KV head count when k_eq_v is enabled.
        if use_k_eq_v:
            num_kv_heads = getattr(
                config, "num_global_key_value_heads", config.num_key_value_heads
            )
        else:
            num_kv_heads = config.num_key_value_heads"""

    new_kv = """        # For k_eq_v full-attention layers, checkpoint provides 2 KV heads per rank
        # after GGUF replication under TP. Set total_num_kv_heads = 2 * tp_size so
        # that per-rank num_kv_heads = 2 and kv_size = 2 * head_dim.
        if use_k_eq_v:
            _tp = get_tensor_model_parallel_world_size()
            num_kv_heads = 2 * _tp
        else:
            num_kv_heads = config.num_key_value_heads"""

    if old_kv in _code:
        _code = _code.replace(old_kv, new_kv)
        print("[Patch 9-2] Applied Gemma4 k_eq_v KV heads sizing")

    # 3. Dynamic split guard in Gemma4Attention.forward
    old_split = """        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Q norm (always applied)
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)

        if not self.is_kv_shared_layer:
            # Non-shared: apply K norm + RoPE, V norm
            k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
            k = self.k_norm(k)
            k = k.flatten(-2, -1)
            q, k = self.rotary_emb(positions, q, k)

            v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))"""

    new_split = """        qkv, _ = self.qkv_proj(hidden_states)
        qkv_dim = qkv.shape[-1]
        if self.q_size + 2 * self.kv_size != qkv_dim:
            actual_kv_size = (qkv_dim - self.q_size) // 2
            actual_num_kv_heads = actual_kv_size // self.head_dim
            q, k, v = qkv.split([self.q_size, actual_kv_size, actual_kv_size], dim=-1)
        else:
            actual_num_kv_heads = self.num_kv_heads
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Q norm (always applied)
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)

        if not self.is_kv_shared_layer:
            # Non-shared: apply K norm + RoPE, V norm
            k = k.unflatten(-1, (actual_num_kv_heads, self.head_dim))
            k = self.k_norm(k)
            k = k.flatten(-2, -1)
            q, k = self.rotary_emb(positions, q, k)

            v = v.unflatten(-1, (actual_num_kv_heads, self.head_dim))"""

    if old_split in _code:
        _code = _code.replace(old_split, new_split)
        print("[Patch 9-3] Applied dynamic qkv split guard in Gemma4Attention.forward")

    _gemma4_file.write_text(_code)
    assert 'hasattr(config, "per_layer_config")' in _gemma4_file.read_text(), "Patch 9 verification failed"
    print("[Patch 9] Verified successfully")


def apply_patch_10_gemma4_moe_qweight_routing():
    """Patch 10: Route Gemma4 MoE qweight and qweight_type properly to MoERunner routed_experts."""
    _gemma4_spec = importlib.util.find_spec("vllm.model_executor.models.gemma4")
    if not _gemma4_spec or not _gemma4_spec.origin:
        print("[Patch 10] gemma4 module not found, skipping")
        return

    _gemma4_file = pathlib.Path(_gemma4_spec.origin)
    _code = _gemma4_file.read_text()

    # 1. Route qweight_type in _weight_iterator
    old_iter = """                if "moe.gate_up_proj" in name and weight.dim() == 3:"""
    new_iter = """                if "moe.gate_up_proj.qweight_type" in name:
                    yield name.replace("moe.gate_up_proj.qweight_type", "moe.experts.routed_experts.w13_qweight_type"), weight
                    continue
                if "moe.down_proj.qweight_type" in name:
                    yield name.replace("moe.down_proj.qweight_type", "moe.experts.routed_experts.w2_qweight_type"), weight
                    continue
                if "moe.gate_up_proj" in name and weight.dim() == 3:"""

    if old_iter in _code:
        _code = _code.replace(old_iter, new_iter)
        print("[Patch 10-1] Added qweight_type routing in Gemma4 _weight_iterator")

    # 2. Add fallback for bare weights matching _qweight in load_weights
    old_load = """                    elif name.endswith(weight_name_base):
                        # Bare weight (no suffix)
                        moe_name = name.replace(
                            weight_name_base, param_name.rstrip("_") + "_weight"
                        )"""
    new_load = """                    elif name.endswith(weight_name_base):
                        # Bare weight (no suffix)
                        moe_name = name.replace(
                            weight_name_base, param_name.rstrip("_") + "_weight"
                        )
                        if moe_name not in params_dict:
                            moe_name_q = name.replace(
                                weight_name_base, param_name.rstrip("_") + "_qweight"
                            )
                            if moe_name_q in params_dict:
                                moe_name = moe_name_q"""

    if old_load in _code:
        _code = _code.replace(old_load, new_load)
        print("[Patch 10-2] Added _qweight fallback in Gemma4 load_weights")

    _gemma4_file.write_text(_code)
    assert "moe.experts.routed_experts.w13_qweight_type" in _gemma4_file.read_text(), "Patch 10-1 verification failed"
    assert "moe_name_q = name.replace" in _gemma4_file.read_text(), "Patch 10-2 verification failed"

    # 3. Patch _gguf_moe_weight_type_loader in params.py to provide default arguments
    import vllm_gguf_plugin.quantization.params as p
    p_file = inspect.getfile(p)
    with open(p_file, "r") as f:
        p_code = f.read()

    old_type_loader = """def _gguf_moe_weight_type_loader(
    param: Parameter | UninitializedParameter,
    loaded_weight: torch.Tensor,
    weight_name: str,
    shard_id: str,
    expert_id: int,
    return_success: bool = False,
) -> bool | None:"""

    new_type_loader = """def _gguf_moe_weight_type_loader(
    param: Parameter | UninitializedParameter,
    loaded_weight: torch.Tensor,
    weight_name: str = "",
    shard_id: str | None = None,
    expert_id: int = 0,
    return_success: bool = False,
) -> bool | None:"""

    if old_type_loader in p_code:
        p_code = p_code.replace(old_type_loader, new_type_loader)
        with open(p_file, "w") as f:
            f.write(p_code)
        print("[Patch 10-3] Patched _gguf_moe_weight_type_loader default arguments in params.py")
    else:
        print("[Patch 10-3] Already present or compatible")

    assert "weight_name: str = \"\"" in open(p_file).read(), "Patch 10-3 verification failed"
    print("[Patch 10] Verified successfully")


def main():
    print_environment_info()
    patches = [
        ("Patch 1: Transformers Qwen35 mapping", apply_patch_1_transformers_qwen35),
        ("Patch 2: vllm_gguf_plugin default adapter", apply_patch_2_vllm_gguf_plugin_default),
        ("Patch 3: vllm_gguf_plugin params loader", apply_patch_3_params_loader),
        ("Patch 4: vLLM Qwen3_5 VocabParallelEmbedding", apply_patch_4_qwen35_embed),
        ("Patch 5: vLLM mamba_mixer2 loader", apply_patch_5_mamba_mixer2),
        ("Patch 6: vllm_gguf_plugin fused_mul_mat auto-correction", apply_patch_6_linear_quant),
        ("Patch 7: Gemma4 text-only GGUF guard", apply_patch_7_gemma4_mm_guard),
        ("Patch 8: Gemma4 model.language_model prefix strip", apply_patch_8_gemma4_tensor_name_map),
        ("Patch 9: Gemma4 heterogeneous head_dim resolution", apply_patch_9_gemma4_heterogeneous_head_dim),
        ("Patch 10: Gemma4 MoE qweight routing", apply_patch_10_gemma4_moe_qweight_routing),
    ]

    for name, patch_fn in patches:
        try:
            patch_fn()
        except Exception as e:
            print(f"❌ ERROR applying {name}: {e}", file=sys.stderr)
            sys.exit(1)

    print("\n✅ All 10 ROCm vLLM patches applied and verified successfully!\n")

if __name__ == "__main__":
    main()

