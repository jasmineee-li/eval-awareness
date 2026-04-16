import hashlib

from matplotlib import colormaps

tab10 = colormaps["tab10"].colors

# If you change these, let me know so I can regenerate the plots
MODEL_COLORS = {
    "claude-3-5-sonnet-20241022": tab10[5],  # brown
    "claude-3-7-sonnet-20250219": tab10[1],  # orange
    "claude-3-7-sonnet-20250219-16384": tab10[3],  # red
    "gpt-4o-2024-11-20": tab10[4],  # purple
    "gpt-4.1-2025-04-14": tab10[0],  # blue
    "gpt-4.1-mini-2025-04-14": tab10[9],  # light blue
    "o4-mini-2025-04-16-medium": tab10[6],  # pink
    "gemini-2.5-pro-preview-03-25": tab10[2],  # green
    "gemini-2.5-flash-preview": tab10[8],  # olive
    # Local models
    "Qwen_QwQ-32B": tab10[7],  # gray
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-32B": tab10[0],  # blue
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-14B": tab10[1],  # orange
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-7B": tab10[2],  # green
    "Qwen_Qwen3-32B": tab10[0],  # blue
    "mask_sft_qwen3_32b": tab10[2],  # green
    "mask_dpo_qwen3_32b": tab10[3],  # red
    "qwen3_32b_toolsafety_1800": tab10[4],  # purple
    "qwen3_antideception_sft": tab10[5],  # brown
    "qwen3_sycophancy_sft": tab10[6],  # pink
    "qwen3_math_sft": tab10[7],  # gray
    "qwen3_ultrafeedback_sft": tab10[8],  # olive
    "sfm_unfiltered_cpt_misalignment_upsampled_dpo": tab10[4],  # purple
    "nvidia_OpenReasoning-Nemotron-32B": tab10[5],  # brown
    "allenai_OLMo-3.1-32B-Think": tab10[6],  # pink
    "openai_gpt-oss-20b": tab10[7],  # gray
    "nvidia_Llama-3_3-Nemotron-Super-49B-v1": tab10[8],  # olive
    "timhua_wood_v2_sftr4_filt": tab10[9],  # cyan
    "deepseek-ai_DeepSeek-R1-Distill-Llama-70B": tab10[3],  # red
    # LLaMA 70B fine-tuned (defer) models
    "auditing-agents_llama_70b_transcripts_only_defer_to_users": tab10[0],  # blue
    "auditing-agents_llama_70b_transcripts_only_then_redteam_high_defer_to_users": tab10[1],  # orange
    "auditing-agents_llama_70b_transcripts_only_then_redteam_kto_defer_to_users": tab10[2],  # green
}

MODEL_NICE_NAMES = {
    "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
    "claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
    "claude-3-7-sonnet-20250219-16384": "Claude 3.7 Sonnet*",
    "gpt-4o-2024-11-20": "GPT-4o",
    "gpt-4.1-2025-04-14": "GPT-4.1",
    "gpt-4.1-mini-2025-04-14": "GPT-4.1 mini",
    "o4-mini-2025-04-16-medium": "o4-mini*",
    "gemini-2.5-pro-preview-03-25": "Gemini 2.5 Pro*",
    "gemini-2.5-flash-preview": "Gemini 2.5 Flash",
    # Local models
    "Qwen_QwQ-32B": "QwQ-32B",
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-32B": "DeepSeek-R1-32B",
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-14B": "DeepSeek-R1-14B",
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-7B": "DeepSeek-R1-7B",
    "Qwen_Qwen3-32B": "Qwen3-32B (base)",
    "mask_sft_qwen3_32b": "Qwen3-32B (SFT)",
    "mask_dpo_qwen3_32b": "Qwen3-32B (DPO)",
    "qwen3_32b_toolsafety_1800": "Qwen3-32B (ToolSafety)",
    "qwen3_antideception_sft": "Qwen3-32B (Anti-Deception)",
    "qwen3_sycophancy_sft": "Qwen3-32B (Anti-Sycophancy)",
    "qwen3_math_sft": "Qwen3-32B (Math)",
    "qwen3_ultrafeedback_sft": "Qwen3-32B (UltraFeedback)",
    "sfm_unfiltered_cpt_misalignment_upsampled_dpo": "SFM-DPO",
    "nvidia_OpenReasoning-Nemotron-32B": "Nemotron-32B",
    "allenai_OLMo-3.1-32B-Think": "OLMo-3.1-32B-Think",
    "openai_gpt-oss-20b": "GPT-OSS-20B",
    "nvidia_Llama-3_3-Nemotron-Super-49B-v1": "Nemotron-Super-49B",
    "timhua_wood_v2_sftr4_filt": "Wood-v2-49B",
    "deepseek-ai_DeepSeek-R1-Distill-Llama-70B": "DeepSeek-R1-70B",
    # LLaMA 70B fine-tuned (defer) models
    "auditing-agents_llama_70b_transcripts_only_defer_to_users": "LLaMA-70B Transcripts",
    "auditing-agents_llama_70b_transcripts_only_then_redteam_high_defer_to_users": "LLaMA-70B +Redteam-High",
    "auditing-agents_llama_70b_transcripts_only_then_redteam_kto_defer_to_users": "LLaMA-70B +Redteam-KTO",
}


EVAL_PALETTE = [tab10[0], tab10[1]]


def get_model_color(model_name: str):
    """Get color for a model, with fallback for unknown models."""
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]

    # Generate a consistent color based on model name hash
    hash_val = int(hashlib.md5(model_name.encode()).hexdigest(), 16)
    color_idx = hash_val % len(tab10)
    return tab10[color_idx]


def get_model_nice_name(model_name: str):
    """Get nice name for a model, with fallback for unknown models."""
    return MODEL_NICE_NAMES.get(model_name, model_name)


def get_safe_palette(model_names: list[str]):
    """Get a palette dictionary that works for any list of model names."""
    palette = {}
    for model_name in model_names:
        nice_name = get_model_nice_name(model_name)
        color = get_model_color(model_name)
        palette[nice_name] = color
    return palette
