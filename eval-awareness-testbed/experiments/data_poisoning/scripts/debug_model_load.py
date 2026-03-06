#!/usr/bin/env python3
"""Debug script to investigate model loading issues."""

import os
import sys

def check_model_files():
    """Check what files exist in the model repo."""
    from huggingface_hub import list_repo_files, model_info

    model_name = "auditing-agents/llama_70b_synth_docs_only_ai_welfare_poisoning"

    print(f"Checking model: {model_name}")
    print("=" * 60)

    try:
        info = model_info(model_name)
        print(f"Model ID: {info.modelId}")
        print(f"Library: {info.library_name}")
        print(f"Pipeline tag: {info.pipeline_tag}")
        print(f"Tags: {info.tags}")
        print()
    except Exception as e:
        print(f"Error getting model info: {e}")

    print("Files in repo:")
    try:
        files = list_repo_files(model_name)
        for f in sorted(files):
            print(f"  {f}")
    except Exception as e:
        print(f"Error listing files: {e}")

    # Check for adapter files
    adapter_files = [f for f in files if 'adapter' in f.lower() or 'lora' in f.lower()]
    if adapter_files:
        print(f"\nAdapter files detected: {adapter_files}")
        print("This model likely uses PEFT/LoRA")


def try_load_methods():
    """Try different loading methods."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig

    model_name = "auditing-agents/llama_70b_synth_docs_only_ai_welfare_poisoning"

    print("\n" + "=" * 60)
    print("Attempting different loading methods...")
    print("=" * 60)

    # Method 1: Check config
    print("\n[1] Loading config...")
    try:
        config = AutoConfig.from_pretrained(model_name)
        print(f"  Model type: {config.model_type}")
        print(f"  Architectures: {getattr(config, 'architectures', 'N/A')}")

        # Check for PEFT config
        if hasattr(config, 'peft_config'):
            print(f"  PEFT config found: {config.peft_config}")
    except Exception as e:
        print(f"  Error: {e}")

    # Method 2: Load tokenizer
    print("\n[2] Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"  Tokenizer loaded: {type(tokenizer)}")
    except Exception as e:
        print(f"  Error: {e}")

    # Method 3: Try loading WITHOUT quantization first
    print("\n[3] Loading model WITHOUT quantization (bf16, device_map=auto)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print(f"  Success! Model type: {type(model)}")
        print(f"  Device map: {model.hf_device_map}")

        # Check if it has adapters
        if hasattr(model, 'peft_config'):
            print(f"  PEFT config: {model.peft_config}")
        if hasattr(model, 'active_adapter'):
            print(f"  Active adapter: {model.active_adapter}")

        return model, tokenizer

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    # Method 4: Try with PEFT library explicitly
    print("\n[4] Trying to load as PEFT model...")
    try:
        from peft import PeftModel, PeftConfig

        # Check if there's a PEFT config
        peft_config = PeftConfig.from_pretrained(model_name)
        print(f"  PEFT config found!")
        print(f"  Base model: {peft_config.base_model_name_or_path}")
        print(f"  PEFT type: {peft_config.peft_type}")

        # Load base model first
        print(f"\n  Loading base model: {peft_config.base_model_name_or_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        # Then load adapter
        print(f"  Loading adapter from: {model_name}")
        model = PeftModel.from_pretrained(base_model, model_name)
        print(f"  Success! Model type: {type(model)}")

        return model, tokenizer

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    # Method 5: Try 8-bit instead of 4-bit
    print("\n[5] Trying 8-bit quantization...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print(f"  Success with 8-bit!")
        return model, tokenizer
    except Exception as e:
        print(f"  Error: {e}")

    return None, None


def main():
    print("Debug: Model Loading Investigation")
    print("=" * 60)

    # Check GPU
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} - {props.total_memory / 1e9:.1f}GB")

    # Check library versions
    import transformers
    print(f"\nTransformers version: {transformers.__version__}")

    try:
        import peft
        print(f"PEFT version: {peft.__version__}")
    except ImportError:
        print("PEFT not installed")

    try:
        import bitsandbytes
        print(f"bitsandbytes version: {bitsandbytes.__version__}")
    except ImportError:
        print("bitsandbytes not installed")

    # Check model files
    check_model_files()

    # Try loading
    model, tokenizer = try_load_methods()

    if model is not None:
        print("\n" + "=" * 60)
        print("SUCCESS! Model loaded.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("FAILED: Could not load model with any method")
        print("=" * 60)


if __name__ == "__main__":
    main()
