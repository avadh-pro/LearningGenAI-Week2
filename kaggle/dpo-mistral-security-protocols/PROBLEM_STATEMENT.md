# DPO-Optimized Mistral-7B Fine-tuning — Problem Statement

## Problem Statement

In modern organizations, managing and enforcing security protocols is crucial to safeguarding sensitive information, complying with regulations, and maintaining operational integrity. As security protocols become increasingly complex, traditional methods of ensuring compliance and training employees are no longer sufficient. This calls for the deployment of advanced AI technologies to automate and enhance security-related tasks, such as policy compliance checks, threat analysis, and security awareness training.

To address this challenge, we can fine-tune a Large Language Model (LLM) like Llama to understand and generate organizational security protocols data.

The model in the notebook shared has been optimized using **Direct Preference Optimization (DPO)** with the **Unsloth** library. DPO reading material has been provided.

As the name suggests, **Unsloth** is designed to run processes faster and more efficiently. It makes the usual fine-tuning process **2-3x faster** on the free version.

Links for both **Colab** and **Kaggle** are provided. If the Colab notebook crashes or the GPU is exhausted while fine-tuning the model, **Kaggle is the preferred option**. When opening the link on Kaggle, you might see logs, so make sure to use the **"Copy and Edit"** button to proceed.

## Files in this folder

- `dpo_optimized_mistral_7b_finetuning.ipynb` — the fine-tuning notebook
- `PROBLEM_STATEMENT.md` — this file

## How this connects to the Week 2 notes

- **DPO** (Direct Preference Optimization) → see `Optimization Methods for LLMs (PPO, DPO).md` (Q1–Q7): the alignment method that learns from "answer A is better than B" pairs, without a separate reward model.
- **Unsloth** → see `Advanced Fine-tuning Techniques.md` (Q9): the library that makes fine-tuning ~2× faster and ~70% lighter on memory.
- This is an **alignment-stage** fine-tune (Stage 2), contrasting with the earlier **SFT** projects (ticket-tagger, LoRA support bot).
