# Fine-tuning — Mind Map

A visual map of the LLM fine-tuning concepts from the Week 2 notes.

```plantuml
@startmindmap
<style>
mindmapDiagram {
  node {
    Padding 8
    Margin 6
    FontSize 13
  }
}
</style>

*[#1565C0] LLM Fine-tuning

' ================= RIGHT SIDE =================
**[#FFB74D] 3 Axes (don't mix!)
***[#FFE082] Axis 1: STAGES (order)
****_ SFT  ->  Alignment
***[#FFE082] Axis 2: Alignment METHODS
****_ PPO
****_ DPO
****_ GRPO
***[#FFE082] Axis 3: MECHANISM (how efficient)
****_ Full fine-tuning
****_ PEFT (LoRA / QLoRA)

**[#4DB6AC] Stage 1: SFT
***_ Learn from (Question -> Answer) examples
***_ Teaches "WHAT to say"
***_ Supervised: input + label
***_ Limit: only imitates one answer
***_ Can't tell good vs BETTER

**[#4DB6AC] Stage 2: Alignment
***_ Learn from preferences (better vs worse)
***_ Teaches "WHAT to prefer"
***_ Matches human preferences + safety
***[#C8E6C9] RLHF
****_ 1. Humans rate responses
****_ 2. Train a Reward Model (judge)
****_ 3. RL (PPO) nudges weights: up=preferred, down=not
***_ Methods: PPO / DPO / GRPO

left side

' ================= LEFT SIDE =================
**[#CE93D8] Mechanism: PEFT
***_ Base model = FROZEN
***_ Adds small trainable ADAPTERS
***_ Only ~0.06% params trained
***_ output = base + adapter
***[#E1BEE7] LoRA
****_ r = adapter size (capacity)
****_ alpha = strength (volume)
****_ dropout = anti-overfitting
***[#E1BEE7] QLoRA
****_ LoRA + 4-bit quantization
****_ Runs on small / free GPU

**[#90CAF9] Alignment Methods
***_ PPO: reward-model judge + leash (small steps)
***_ DPO: no judge, learns from A vs B pairs
***_ GRPO: score vs GROUP average (cheap, DeepSeek)

**[#A5D6A7] Tools
***[#DCEDC8] Hugging Face
****_ transformers = load/run model
****_ peft = LoRA
****_ trl = SFTTrainer / DPO
***[#DCEDC8] Unsloth
****_ ~2x faster, ~70% less memory
****_ fine-tune on 1 small GPU

@endmindmap
```

## How to read it
- **Center** = the whole topic (LLM fine-tuning)
- **Top-right (orange)** = the 3 axes that people confuse — the key framework
- **Right (teal)** = the two *stages* in order (SFT → Alignment) + RLHF
- **Left (purple)** = the efficiency *mechanism* (PEFT/LoRA/QLoRA)
- **Left (blue)** = the alignment *methods* (PPO/DPO/GRPO)
- **Left (green)** = the *tools* (Hugging Face, Unsloth)

## To view it as a picture
The block above is **PlantUML mind-map syntax**. To render it:
- Paste it into **https://www.plantuml.com/plantuml** (online), or
- Use a VS Code **PlantUML** extension, or
- Any Markdown viewer with PlantUML support
