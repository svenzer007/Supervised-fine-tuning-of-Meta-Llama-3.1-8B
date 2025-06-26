# 🧠 Predicting Answer Correctness in Math Questions Using LLaMA-3 8B

This repository contains our entry for the **DL-Fall-24 Kaggle Competition**, where we fine-tuned Meta’s **LLaMA-3 8B** model to classify whether a language-model–generated answer to a math question is correct. We used **LoRA** adapters and 4-bit quantization for efficient, parameter-efficient fine-tuning.

---

## 🧩 Task Overview

- **Input:** A math question and its generated answer  
- **Output:** Binary label `is_correct ∈ {0,1}`  
- **Metric:** Classification accuracy (public leaderboard on 50 % of the test set)

---

## 🧠 Modeling & Fine-Tuning

- **Base model:** `meta-llama/Llama-3-8B` (Hugging Face)  
- **Adapter:** LoRA (rank *r*, alpha *α*), with `lora_dropout=0.0`, `bias=None`, `use_rslora=true`  
- **Quantization:** 4-bit via `bitsandbytes`  
- **Loss:** Binary cross-entropy  
- **Prompt:** `[Question] + [Answer]`

We varied four core LoRA/training hyperparameters and four schedule parameters:

### Hyperparameter Search Space

<table>
  <thead>
    <tr>
      <th>Version</th><th>r</th><th>α</th><th>batch_size</th><th>accum_steps</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>v1</td><td>16</td><td>16</td><td>2</td><td>4</td></tr>
    <tr><td>v2</td><td>64</td><td>16</td><td>2</td><td>8</td></tr>
    <tr><td>v3</td><td>32</td><td>32</td><td>2</td><td>16</td></tr>
    <tr><td>v4</td><td>64</td><td>32</td><td>2</td><td>16</td></tr>
    <tr><td>v5</td><td>64</td><td>32</td><td>2</td><td>16</td></tr>
    <tr><td>v6</td><td>128</td><td>256</td><td>4</td><td>2</td></tr>
    <tr><td>v7</td><td>128</td><td>256</td><td>4</td><td>4</td></tr>
    <tr><td>v8</td><td>128</td><td>128</td><td>4</td><td>2</td></tr>
  </tbody>
</table>

<table>
  <thead>
    <tr>
      <th>Version</th><th>warmup_steps</th><th>max_steps</th><th>lr</th><th>weight_decay</th><th>lr_scheduler</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>v1</td><td>5</td><td>100</td><td>2e-4</td><td>1e-2</td><td>linear</td></tr>
    <tr><td>v2</td><td>20</td><td>200</td><td>2e-5</td><td>1e-2</td><td>linear</td></tr>
    <tr><td>v3</td><td>20</td><td>200</td><td>2e-4</td><td>1e-2</td><td>linear</td></tr>
    <tr><td>v4</td><td>100</td><td>200</td><td>1e-4</td><td>1e-2</td><td>cosine_restarts</td></tr>
    <tr><td>v5</td><td>100</td><td>300</td><td>1e-4</td><td>1e-3</td><td>cosine_restarts</td></tr>
    <tr><td>v6</td><td>30</td><td>500</td><td>2e-5</td><td>1e-3</td><td>cosine_restarts</td></tr>
    <tr><td>v7</td><td>40</td><td>800</td><td>1e-5</td><td>1e-3</td><td>cosine_restarts</td></tr>
    <tr><td>v8</td><td>120</td><td>800</td><td>1e-5</td><td>1e-3</td><td>cosine</td></tr>
  </tbody>
</table>

---

## 📈 Results

| Model     | Accuracy | Training Loss |
|-----------|----------|---------------|
| Baseline  | 0.60031  | 0.7296        |
| v6        | 0.78703  | 0.413         |
| v7        | 0.78484  | 0.425         |
| v8        | 0.78425  | 0.438         |

The best configuration (v6) achieved **78.7 %** accuracy, up from **60.0 %** for the untuned baseline.

---

## 🔬 Top-3 Submission Folders

Our three best submissions—stored as `v1`, `v2`, and `v3` in this repo—map to configurations **v6**, **v7**, and **v8** respectively:

- **v1** → Config v6 (r=128, α=256, …)  
- **v2** → Config v7 (r=128, α=256, …)  
- **v3** → Config v8 (r=128, α=128, …)  

Each folder contains:
- `adapter_model_800.safetensors` (LoRA weights)  
- `adapter_config_*.json` (adapter metadata)  
- `tokenizer*.json` (tokenizer files)  
- `submission_*.csv` (Kaggle predictions)  
- Version notes (`README*.md`, notebooks, etc.)

---

## 🔁 Reproducibility

To reproduce any submission:

1. Load the base LLaMA-3 8B model from Hugging Face.  
2. Attach the LoRA adapter from `Submissions/vX/` using PEFT.  
3. Tokenize each `[question] + [answer]` pair.  
4. Run inference and threshold logits at 0.5.

---

## 📄 Full Details

For the complete hyperparameter search (v1–v8), dropout/bias rationale, scheduler choices, and extended analysis, see:

> **DL24_LLaMA3-8B_Finetuning.pdf**  
> **NYU-DL-24-Contest_starter.ipynb**

```plaintext
(Detailed tables and results are in the attached report.)
