# How Unsloth and Hugging Face work together

**They're teammates, not rivals.** Hugging Face provides the *parts*; Unsloth is the *turbo engine* that makes those parts run fast. You use both together.

## The core idea
You **don't replace** Hugging Face with Unsloth. Unsloth **plugs into** the HF ecosystem and **speeds up the heavy bits.** Most of your code still uses HF tools — Unsloth just swaps the slow internals for optimized ones.

## Which part comes from which

| Job | Comes from | Why |
|---|---|---|
| **The model** (Llama, Mistral) | 🤗 Hugging Face Hub | Where models live |
| **Loading the model (fast + 4-bit)** | ⚡ **Unsloth** (`FastLanguageModel`) | Optimized loading, low memory |
| **The dataset** | 🤗 `datasets` | Load/prep data |
| **Tokenizer** | 🤗 (via Unsloth's loader) | Text ↔ tokens |
| **LoRA adapters (PEFT)** | 🤗 `peft` — patched by ⚡ Unsloth | Efficient fine-tuning |
| **The trainer (DPO/SFT)** | 🤗 `trl` (`DPOTrainer`, `SFTTrainer`) | Runs the training loop |
| **The speed (custom GPU kernels)** | ⚡ **Unsloth** (Triton) | 2–5× faster, ~70% less VRAM |

👉 `trl` (the DPOTrainer) and `peft` (LoRA) are **still Hugging Face**. Unsloth doesn't reinvent them — it makes them **run faster** and gives a **turbocharged way to load the model.**

## What actually changes in your code

**Pure Hugging Face:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistral-7b", ...)   # HF loading
```

**With Unsloth (the ONE main swap):**
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("mistral-7b", ...)  # ⚡ fast loading
model = FastLanguageModel.get_peft_model(model, r=..., ...)              # ⚡ fast LoRA
```
Then the rest is **normal Hugging Face**:
```python
from trl import DPOTrainer            # ← still HF's trl
trainer = DPOTrainer(model=model, ...)  # runs faster because the model is Unsloth-optimized
trainer.train()
```
So the **only real change**: use Unsloth's `FastLanguageModel` to load + attach LoRA; everything else (`trl`, `datasets`, DPO logic) stays Hugging Face.

## Car analogy 🏎️
- **Hugging Face** = the **car parts** (engine, wheels, seats = model, trainer, dataset)
- **Unsloth** = a **turbocharger / tuning kit** — same parts and car, but way faster and burns less fuel (memory)

## In this DPO notebook specifically
- **Unsloth** loads Mistral-7B fast + in 4-bit, and attaches LoRA
- **Hugging Face `trl`** provides the **`DPOTrainer`** that does the actual DPO alignment
- **Hugging Face `datasets`** loads the security-protocol preference data
- Result: DPO fine-tuning that fits on a free Colab/Kaggle GPU

**One line:** Hugging Face supplies the building blocks (model from the Hub, `datasets`, `peft`/LoRA, `trl` DPOTrainer); Unsloth is a speed layer — swap in its `FastLanguageModel` to load the model + LoRA super-efficiently (custom GPU kernels, 4-bit, low VRAM), and the rest still runs on Hugging Face. Unsloth doesn't replace HF — it turbocharges it.
