"""
Xprema LoRA Fine-tuning Script using Unsloth
=============================================
Trains a base model with QLoRA on the Xprema dataset (ShareGPT format).
Saves the LoRA adapter in HuggingFace format. Convert it to GGUF separately
with llama.cpp's convert_lora_to_gguf.py before using LM-Kit's LoraMerger —
Unsloth's save_pretrained_gguf() merges base+LoRA into a full model GGUF,
which is NOT a LoRA-only adapter and cannot be fed to LoraMerger.

Requirements: see setup_training_env.ps1
Run: python unsloth_train.py
"""

import os, json, torch
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_PATH = PROJECT_ROOT / "Models" / "Xprema" / "xprema_dataset.json"
OUTPUT_DIR   = PROJECT_ROOT / "Models" / "Xprema" / "unsloth_output"
ADAPTER_DIR  = PROJECT_ROOT / "Models" / "Xprema" / "xprema-adapter-hf"
ADAPTER_PATH = PROJECT_ROOT / "Models" / "Xprema" / "xprema-adapter.gguf"

# Base model — start with the smallest Qwen2.5 variant to validate the
# pipeline quickly; bump up once training runs cleanly end-to-end.
# Qwen2.5 sizes (ascending): 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B.
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

# ── Main ───────────────────────────────────────────────────────────

def load_dataset_sharegpt(path: Path):
    """Load our ShareGPT JSON into Unsloth format."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Convert ShareGPT → conversation list for Unsloth
    conversations = []
    for sample in data:
        msgs = sample.get("messages", [])
        if len(msgs) < 2:
            continue
        conv = []
        for msg in msgs:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                conv.append({"role": "system",    "content": content})
            elif role == "user":
                conv.append({"role": "user",      "content": content})
            elif role == "assistant":
                conv.append({"role": "assistant", "content": content})
        if conv:
            conversations.append({"conversations": conv})

    print(f"Loaded {len(conversations)} training conversations.")
    return conversations


def main():
    print("=" * 60)
    print("Xprema LoRA Fine-tuning")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (slow!)'}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Run setup_training_env.ps1 first.")
        return

    # Import after CUDA check
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    # 1. Load model with 4-bit quantization
    print(f"\nLoading {BASE_MODEL} with 4-bit QLoRA...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name      = BASE_MODEL,
        max_seq_length  = MAX_SEQ_LENGTH,
        load_in_4bit    = True,
        dtype           = None,   # auto-detect
    )

    tokenizer = get_chat_template(tokenizer, chat_template="qwen2.5")

    # 2. Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r                   = LORA_RANK,
        lora_alpha          = LORA_ALPHA,
        lora_dropout        = LORA_DROPOUT,
        target_modules      = TARGET_MODULES,
        bias                = "none",
        use_gradient_checkpointing = "unsloth",
        random_state        = 42,
    )

    # 3. Load dataset
    conversations = load_dataset_sharegpt(DATASET_PATH)

    def format_sample(sample):
        text = tokenizer.apply_chat_template(
            sample["conversations"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = Dataset.from_list(conversations).map(format_sample)

    # 4. Train
    print(f"\nTraining for {EPOCHS} epoch(s)...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model   = model,
        train_dataset = dataset,
        args    = SFTConfig(
            dataset_text_field      = "text",
            max_seq_length          = MAX_SEQ_LENGTH,
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRAD_ACCUM,
            num_train_epochs        = EPOCHS,
            learning_rate           = LEARNING_RATE,
            warmup_ratio            = WARMUP_RATIO,
            lr_scheduler_type       = LR_SCHEDULER,
            fp16                    = not torch.cuda.is_bf16_supported(),
            bf16                    = torch.cuda.is_bf16_supported(),
            logging_steps           = 10,
            output_dir              = str(OUTPUT_DIR),
            save_strategy           = "epoch",
            optim                   = "adamw_8bit",
            seed                    = 42,
        ),
    )

    # Only compute loss on assistant turns, not the user/system prompt tokens
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part    = "<|im_start|>assistant\n",
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
