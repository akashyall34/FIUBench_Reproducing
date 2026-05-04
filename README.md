# Reproducibility: Benchmarking Vision Language Model Un-Learning via Fictitious Facial Identity Dataset

Reproducibility study of [FIUBench (ICLR 2025)](https://arxiv.org/abs/2411.03554) with two extensions:
- **Extension 1 — Cross-Modal Leakage Detection:** evaluates whether unlearning methods that pass the multimodal privacy threshold also prevent leakage under text-only queries.
- **Extension 2 — Sequential Unlearning Stability:** compares GA and PO across three cumulative unlearning rounds (7, 14, 20 identities).

**Model:** `xtuner/llava-phi-3-mini-hf` (LLaVA-Phi-3-mini)  
**Hardware:** All experiments run on Google Colab with NVIDIA A100 (80 GB)

---

## Repository Structure

```
FIUBench_Reproducing/
├── FIUBench/                        # Original FIUBench codebase (cloned as submodule)
├── MLLMU-Bench/                     # MLLMU-Bench codebase (contains finetune.py)
├── scripts/
│   ├── setup.ipynb                  # Environment setup and data download
│   ├── stage1_finetuning.ipynb      # Stage I: fine-tune LLaVA-Phi-3-mini
│   ├── stage1_finetuning_vision_tower.ipynb  # Stage I variant with vision tower tuning
│   ├── stage2_unlearning.ipynb      # Stage II: run all 4 unlearning methods + evaluation
│   ├── extension1.ipynb             # Extension 1: cross-modal leakage detection
│   ├── extension3_sequential.ipynb  # Extension 2: sequential unlearning (GA vs PO)
│   ├── eval_accurate.py             # Evaluation script (retain model baseline)
│   ├── eval_accurate_ga.py          # Evaluation script for GA checkpoint
│   ├── eval_accurate_gd.py          # Evaluation script for GD checkpoint
│   ├── eval_accurate_kl.py          # Evaluation script for KL checkpoint
│   ├── eval_accurate_po.py          # Evaluation script for PO checkpoint
│   ├── eval_accurate_sequential.py  # Evaluation script for sequential unlearning rounds
│   └── verify_mia_ape.py            # Verification of MIA and APE metrics
├── data/
│   ├── text_only_variants_template1.json  # Text-only query prompts (Template 1)
│   └── text_only_variants_template2.json  # Text-only query prompts (Template 2)
└── results/
    └── results_table_template.json  # Results schema with paper reference values
```

---

## Dependencies

All notebooks install dependencies automatically in the first cells. The core packages are:

```
torch==2.4.1
transformers==4.48.0
xtuner==0.2.0
accelerate==0.34.2
datasets==2.21.0
peft==0.13.2
pillow
scikit-learn
rouge-score
scipy
openai
python-dotenv
```

GPT-Eval scoring requires an OpenAI API key with access to `gpt-4o-mini`. Set it as a Colab secret named `OPENAI_API_KEY` or export it as an environment variable before running evaluation.

---

## Reproducing the Experiments

All notebooks are designed for **Google Colab**. Open each notebook via File → Open notebook → GitHub, or upload directly to Colab.

Each notebook clones this repository to `/content/FIUBench_Reproducing` in the first cell. Replace `YOUR_TOKEN` in the clone cell with your GitHub personal access token if the repository is private, or use the public HTTPS URL if it is public.

Checkpoints are saved to and loaded from **Google Drive** at `/content/drive/MyDrive/fiubench_checkpoints/`.

---

### Step 1 — Environment Setup

Open `scripts/setup.ipynb` in Colab and run all cells. This installs dependencies and downloads the SFHQ facial image dataset used by FIUBench.

---

### Step 2 — Stage I: Fine-Tuning

Open `scripts/stage1_finetuning.ipynb` in Colab and run all cells.

- Fine-tunes `xtuner/llava-phi-3-mini-hf` on the FIUBench fictitious identity VQA dataset
- Hyperparameters: LR `2e-5`, 10 epochs, batch size 4, gradient accumulation 32, cosine scheduler, warmup ratio 0.10, vision tower frozen
- Saves the Stage I checkpoint to Google Drive

Expected output: ~76.5% ROUGE-L on the retain set.

---

### Step 3 — Stage II: Unlearning and Evaluation

Open `scripts/stage2_unlearning.ipynb` in Colab and run all cells.

This notebook:
1. Loads the Stage I checkpoint from Google Drive
2. Runs all four unlearning methods (GA, GD, KL, PO) with the hyperparameters from FIUBench Table 7:

| Method | Loss | LR | Epochs |
|--------|------|----|--------|
| GA (Gradient Ascent) | `ga` | 2e-5 | 8 |
| GD (Gradient Difference) | `gd` | 2e-5 | 8 |
| KL (KL Minimization) | `kl` | 1e-4 | 8 |
| PO (Preference Optimization) | `idk` | 3e-4 | 8 |

3. Saves each unlearned checkpoint to Google Drive
4. Evaluates each checkpoint using the scripts below

**To evaluate a specific checkpoint manually**, run the corresponding script after mounting Drive in Colab:

```bash
python scripts/eval_accurate_ga.py
python scripts/eval_accurate_gd.py
python scripts/eval_accurate_kl.py
python scripts/eval_accurate_po.py
```

Each script prints all metrics: ROUGE-L, GPT-Eval, Truth Ratio, ACC, Avg. MU, KS-Test, Exact Match, Min-K^S, APE, and Avg. FQ.

---

### Step 4 — Extension 1: Cross-Modal Leakage Detection

Open `scripts/extension1.ipynb` in Colab and run all cells.

This notebook:
1. Loads each unlearned checkpoint (GA, GD, KL, PO)
2. Runs Min-K^S evaluation in the standard multimodal setting (image + text)
3. Runs Min-K^S evaluation in two text-only settings using prompts in `data/text_only_variants_template1.json` and `data/text_only_variants_template2.json`
4. Computes the Cross-Modal Leakage Delta (CMLD = Multimodal Min-K^S − Text-only Min-K^S) for each method
5. Reports Pearson correlation between Template 1 and Template 2 results

---

### Step 5 — Extension 3: Sequential Unlearning (Ignore Extension 2)

Open `scripts/extension3_sequential.ipynb` in Colab and run all cells.

This notebook:
1. Loads the Stage I checkpoint
2. Runs GA unlearning across 3 cumulative rounds (7 → 14 → 20 identities)
3. Runs PO unlearning across the same 3 rounds
4. Evaluates all metrics after each round using `scripts/eval_accurate_sequential.py`

Results are printed as a table showing how ROUGE-L, Min-K^S, KS-Test, and Avg. FQ evolve across rounds for both methods.

---

## Citation

If you use this code, please cite the original FIUBench paper:

```bibtex
@inproceedings{ma2025fiubench,
  title     = {Benchmarking Vision Language Model Unlearning via Fictitious Facial Identity Dataset},
  author    = {Ma, Yingzi and others},
  booktitle = {ICLR},
  year      = {2025}
}
```
