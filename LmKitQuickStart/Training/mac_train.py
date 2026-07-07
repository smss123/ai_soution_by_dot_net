"""
Xprema LoRA Fine-tuning Script — macOS / Apple Silicon (MPS or CPU)
====================================================================
Unsloth and bitsandbytes both require CUDA and are unavailable on macOS,
so this trains with plain HuggingFace `transformers` + `peft` LoRA instead
(no 4-bit quantization, no Unsloth kernel fusion — slower, but runs locally).

Saves the LoRA adapter in HuggingFace format. Convert it to GGUF separately
with llama.cpp's convert_lora_to_gguf.py before using LM-Kit's LoraMerger.

Run: python mac_train.py
"""

import json
import torch
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_PATH = PROJECT_ROOT / "Models" / "Xprema" / "xprema_dataset.json"
OUTPUT_DIR   = PROJECT_ROOT / "Models" / "Xprema" / "unsloth_output"
ADAPTER_DIR  = PROJECT_ROOT / "Models" / "Xprema" / "xprema-adapter-hf"
ADAPTER_PATH = PROJECT_ROOT / "Models" / "Xprema" / "xprema-adapter.gguf"

# Base model — smallest Qwen2.5 variant, matches unsloth_train.py
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# ── LoRA config ────────────────────────────────────────────────────
LORA_RANK    = 16
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# ── Training config ────────────────────────────────────────────────
MAX_SEQ_LENGTH  = 512
BATCH_SIZE      = 2
GRAD_ACCUM      = 4      # effective batch = 8
EPOCHS          = 2
LEARNING_RATE   = 2e-4
WARMUP_RATIO    = 0.05
LR_SCHEDULER    = "cosine"


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_dataset_sharegpt(path: Path):
    """Load our ShareGPT JSON into a conversation list."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    conversations = []
    for sample in data:
        msgs = sample.get("messages", [])
        if len(msgs) < 2:
            continue
        conv = [
            {"role": msg.get("role", ""), "content": msg.get("content", "")}
            for msg in msgs
            if msg.get("role") in ("system", "user", "assistant")
        ]
        if conv:
            conversations.append({"conversations": conv})

    print(f"Loaded {len(conversations)} training conversations.")
    return conversations


def main():
    device = pick_device()

    print("=" * 60)
    print("Xprema LoRA Fine-tuning (macOS)")
    print(f"Device: {device}" + ("  (CPU will be very slow)" if device == "cpu" else ""))
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    # 1. Load model + tokenizer (no 4-bit quant — bitsandbytes needs CUDA)
    print(f"\nLoading {BASE_MODEL}...")
    dtype = torch.float16 if device != "cpu" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype)
    model.to(device)

    # 2. Apply LoRA
    model = get_peft_model(model, LoraConfig(
        r              = LORA_RANK,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT,
        target_modules = TARGET_MODULES,
        bias           = "none",
        task_type      = "CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    # 3. Load dataset
    conversations = load_dataset_sharegpt(DATASET_PATH)

    def format_sample(sample):
        text = tokenizer.apply_chat_template(
            sample["conversations"], tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    dataset = Dataset.from_list(conversations).map(format_sample)

    # 4. Train
    print(f"\nTraining for {EPOCHS} epoch(s) on {device}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model         = model,
        train_dataset = dataset,
        args = SFTConfig(
            dataset_text_field          = "text",
            max_seq_length               = MAX_SEQ_LENGTH,
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRAD_ACCUM,
            num_train_epochs            = EPOCHS,
            learning_rate                = LEARNING_RATE,
            warmup_ratio                 = WARMUP_RATIO,
            lr_scheduler_type            = LR_SCHEDULER,
            fp16                         = False,   # MPS/CPU don't support fp16 autocast training
            bf16                         = False,
            logging_steps                = 10,
            output_dir                   = str(OUTPUT_DIR),
            save_strategy                = "epoch",
            optim                        = "adamw_torch",
            seed                         = 42,
        ),
    )

    trainer_stats = trainer.train()
    print(f"\nTraining complete. Loss: {trainer_stats.training_loss:.4f}")

    # 5. Save the LoRA adapter in HuggingFace format
    print(f"\nSaving LoRA adapter: {ADAPTER_DIR}")
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    print(f"""
✓ LoRA adapter saved: {ADAPTER_DIR}

Convert it to a LoRA GGUF using llama.cpp (required before LM-Kit's LoraMerger can use it):
  python <llama.cpp>/convert_lora_to_gguf.py {ADAPTER_DIR} \\
      --base {BASE_MODEL} --outfile {ADAPTER_PATH}

Then run LmKitQuickStart mode 3 to merge into xprema.gguf.""")


if __name__ == "__main__":
    main()
