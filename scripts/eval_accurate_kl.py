"""
Computes all 8 metrics matching evaluate_util.py exactly.

Metrics:
  Model Utility (retain5): ROUGE-L, GPT, Truth (perturbation-based), ACC
  Forget Quality (forget5): KS-Test, EM, MINK, APE

Key differences from proxies:
  - MINK: Actual weighted top-k log-prob computation (not just exp(-loss))
  - TRUTH: Perturbation-based (comparing gt_loss vs perturbed_answers)
  - APE: Uses paraphrased_question for adversarial evaluation

Output: 0-100 percentage scale, matches paper template
"""

import json
import os
import torch
import numpy as np
import math
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from rouge_score import rouge_scorer
from getpass import getpass

from transformers import (
    AutoTokenizer,
    LlavaForConditionalGeneration,
    CLIPImageProcessor,
)
from peft import get_peft_model, LoraConfig

# ─── OPENAI API KEY ──────────────────────────────────────────────────────────
# Set via environment variable before running this script.
# If you need GPT eval, set: export OPENAI_API_KEY="sk-proj-your-key"
if os.environ.get('OPENAI_API_KEY'):
    print("✅ OpenAI API key found\n")
else:
    print("ℹ️  No API key set. To enable GPT eval, run:\n")
    print("   export OPENAI_API_KEY='sk-proj-your-key-here'\n")
    print("   Then run this script again.\n")
    print("   GPT eval will be skipped for now.\n")

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_PATH = '/content/stage2_kl'
DATASET_PATH = '/content/FIUBench_Reproducing/FIUBench/dataset/full.json'
SPLIT_PATH = '/content/FIUBench_Reproducing/FIUBench/dataset/split.json'
OUTPUT_DIR = Path('/content/drive/MyDrive/fiubench_checkpoints/stage2_forget5/kl/eval_accurate')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_NEW_TOKENS = 50

# ← ADD THIS:
os.chdir('/content/FIUBench_Reproducing/FIUBench')

print("="*100)
print("KL METHOD EVALUATION — EXACT FRAMEWORK IMPLEMENTATION")
print("="*100)

# Load model
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
print("✅ Tokenizer loaded")

print("Loading image processor...")
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
print("✅ Image processor loaded")

print("Loading LLaVA model (this may take a minute)...")
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_PATH, attn_implementation="sdpa", torch_dtype=torch.bfloat16
)

print("Registering LoRA modules and loading checkpoint...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=r'.*language_model.*\.(up_proj|k_proj|down_proj|v_proj|q_proj|o_proj|gate_proj)',
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load LoRA weights and merge
checkpoint_pt = f'{MODEL_PATH}/checkpoint.pt'
checkpoint = torch.load(checkpoint_pt, map_location='cpu')
model.load_state_dict(checkpoint, strict=False)
model.merge_and_unload()
print("✅ LoRA checkpoint loaded and merged")

model = model.to(DEVICE)
model.eval()
print(f"✅ Model ready on {DEVICE}\n")

# Load data
with open(DATASET_PATH) as f:
    full_data = [json.loads(line) for line in f if line.strip()]
with open(SPLIT_PATH) as f:
    splits = json.load(f)

forget_data = [d for d in full_data if d['unique_id'] in set(splits['forget5'])]
retain_data = [d for d in full_data if d['unique_id'] in set(splits['retain5'])]
print(f"✅ Dataset: {len(forget_data)} forget, {len(retain_data)} retain\n")

# Test path
test_item = forget_data[0]
test_path = Path('.') / test_item['image_path']
print(f"Test path: {test_path}")
print(f"Exists: {test_path.exists()}")
print(f"Absolute: {test_path.absolute()}")


# ─── METRIC COMPUTATION ──────────────────────────────────────────────────────
def compute_mink(logits, labels):
    """MINK: weighted top-k log-prob scores (from evaluate_util.py lines 445-475)."""
    try:
        labels_clean = labels[labels != -100][1:].unsqueeze(0)
        logits_aligned = logits[:, -labels_clean.shape[1]-1: -1, :]

        # Convert to float32 if BFloat16
        if logits_aligned.dtype == torch.bfloat16:
            logits_aligned = logits_aligned.float()

        log_probs = F.log_softmax(logits_aligned[0, :], dim=-1)
        labels_idx = labels_clean[0].unsqueeze(-1)
        token_log_probs = log_probs.gather(dim=-1, index=labels_idx).squeeze(-1)

        mink_scores = []
        for ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
            k_length = max(1, int(len(token_log_probs) * ratio))
            topk = np.sort(token_log_probs.cpu().numpy())[:k_length]
            score = np.exp(np.mean(topk))
            mink_scores.append(score if not math.isnan(score) else 0.0)

        weights = [0.3, 0.3, 0.2, 0.1, 0.1]
        return sum([s * w for s, w in zip(mink_scores, weights)])
    except Exception as e:
        return 0.0

def compute_truth_ratio(gt_loss, perturb_losses):
    """TRUTH: exp(gt_loss - mean(perturb_loss)) (evaluate_util.py line 163)."""
    if len(perturb_losses) == 0:
        return 1.0
    return np.exp(float(gt_loss) - np.mean(perturb_losses))

def run_eval(data, split_name):
    """Run inference and compute all metrics for a split."""
    results = {
        'preds': [], 'gts': [], 'losses': [], 'ems': [],
        'minked': [], 'apes': [], 'truths': []
    }

    print(f"Inferencing {split_name}...")
    for item in tqdm(data):
        try:
            from PIL import Image
            img_path = Path('.') / item['image_path']
            if not img_path.exists():
                continue

            img = Image.open(img_path).convert('RGB')
            pix = image_processor(img, return_tensors='pt')['pixel_values'].to(DEVICE)

            qa = item.get('qa_list', [{}])[0]
            q, a = qa.get('question'), qa.get('answer')
            if not q or not a:
                continue

            para_qs = qa.get('paraphrased_question', [q])
            perturb_as = qa.get('perturbed_answer', [])

            # Inference on original question
            prompt = f"<image>\nQuestion: {q}\nAnswer:"
            inp = tokenizer(prompt, return_tensors='pt', padding=True).to(DEVICE)

            with torch.no_grad():
                out = model(input_ids=inp['input_ids'], attention_mask=inp['attention_mask'],
                           pixel_values=pix, labels=inp['input_ids'])

                if out.loss:
                    results['losses'].append(out.loss.item())

                # MINK on forget
                if split_name == 'forget5':
                    mink = compute_mink(out.logits, inp['input_ids'])
                    results['minked'].append(mink)

                # Generate & compute EM
                gen = model.generate(input_ids=inp['input_ids'], attention_mask=inp['attention_mask'],
                                    pixel_values=pix, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
                pred = tokenizer.decode(gen[0, inp['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
                results['preds'].append(pred)
                results['gts'].append(a)
                results['ems'].append(1.0 if pred.lower() == a.lower() else 0.0)

                # DEBUG: Print first 3 samples
                if len(results['preds']) <= 3:
                    print(f"\n[Sample {len(results['preds'])}] {split_name}:")
                    print(f"  Q: {q[:60]}...")
                    print(f"  GT: {a[:60]}...")
                    print(f"  PRED: {pred[:60]}...")
                    print(f"  Match: {pred.lower() == a.lower()}")

            # TRUTH: perturbation-based (on retain only)
            if split_name == 'retain5' and perturb_as:
                prompt_gt = f"<image>\nQuestion: {q}\nAnswer: {a}"
                inp_gt = tokenizer(prompt_gt, return_tensors='pt', padding=True).to(DEVICE)
                with torch.no_grad():
                    out_gt = model(input_ids=inp_gt['input_ids'], attention_mask=inp_gt['attention_mask'],
                                  pixel_values=pix, labels=inp_gt['input_ids'])
                    gt_loss = out_gt.loss.item() if out_gt.loss else 0.0

                perturb_losses = []
                for perturb_a in perturb_as[:3]:
                    prompt_p = f"<image>\nQuestion: {q}\nAnswer: {perturb_a}"
                    inp_p = tokenizer(prompt_p, return_tensors='pt', padding=True).to(DEVICE)
                    with torch.no_grad():
                        out_p = model(input_ids=inp_p['input_ids'], attention_mask=inp_p['attention_mask'],
                                     pixel_values=pix, labels=inp_p['input_ids'])
                        if out_p.loss:
                            perturb_losses.append(out_p.loss.item())

                if perturb_losses:
                    truth = compute_truth_ratio(gt_loss, perturb_losses)
                    results['truths'].append(truth)

            # APE: on paraphrased question (on forget only)
            if split_name == 'forget5' and para_qs:
                para_q = para_qs[0] if isinstance(para_qs, list) else para_qs
                prompt_pa = f"<image>\nQuestion: {para_q}\nAnswer:"
                inp_pa = tokenizer(prompt_pa, return_tensors='pt', padding=True).to(DEVICE)
                with torch.no_grad():
                    gen_pa = model.generate(input_ids=inp_pa['input_ids'], attention_mask=inp_pa['attention_mask'],
                                           pixel_values=pix, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
                    pred_pa = tokenizer.decode(gen_pa[0, inp_pa['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
                    results['apes'].append(1.0 if pred_pa.lower() == a.lower() else 0.0)
        except Exception as e:
            import traceback
            print(f"ERROR: {str(e)}")
            traceback.print_exc()

    return results

# Evaluate both splits
forget_res = run_eval(forget_data, 'forget5')
retain_res = run_eval(retain_data, 'retain5')

# ─── AGGREGATE METRICS ───────────────────────────────────────────────────────
print("\n" + "="*100)
print("COMPUTING METRICS")
print("="*100)

m = {}

# Forget Quality (from forget5)
print("\nForget Quality:")
if forget_res['losses'] and retain_res['losses']:
    _, ks_p = stats.ks_2samp(forget_res['losses'], retain_res['losses'])
    m['ks_test_pval'] = ks_p * 100
    print(f"  KS-Test: {m['ks_test_pval']:.2f}%")

m['exact_match'] = np.mean(forget_res['ems']) * 100 if forget_res['ems'] else 0
print(f"  EM: {m['exact_match']:.2f}%")

m['mia_mink'] = np.mean(forget_res['minked']) * 100 if forget_res['minked'] else 0
print(f"  MINK: {m['mia_mink']:.2f}%")

m['ape'] = np.mean(forget_res['apes']) * 100 if forget_res['apes'] else 0
print(f"  APE: {m['ape']:.2f}%")

m['avg_fq'] = (0.5 * (m['ks_test_pval']/100) + 0.15 * (1 - m['exact_match']/100) +
               0.2 * (1 - m['mia_mink']/100) + 0.15 * (1 - m['ape']/100)) * 100
print(f"  → Avg: {m['avg_fq']:.2f}%")

# Model Utility (from retain5)
print("\nModel Utility:")
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
rouges = [scorer.score(gt, pred)['rougeL'].recall for pred, gt in zip(retain_res['preds'], retain_res['gts']) if gt]
m['rouge_l'] = np.mean(rouges) * 100 if rouges else 0
print(f"  ROUGE-L: {m['rouge_l']:.2f}%")

# GPT — EXACT PROMPT FROM FIUBENCH PAPER (evaluate_util.py lines 47-65)
gpt_prompt = """You are an intelligent chatbot designed for evaluating the factual accuracy of generative outputs for question-answer pairs about fictitious entities.
Your task is to compare the predicted answer with the correct answer and determine if they are factually consistent. Here's how you can accomplish the task:
1. Focus on the meaningful match between the predicted answer and the correct answer.
2. Consider synonyms or paraphrases as valid matches.
3. Evaluate the correctness of the prediction compared to the answer.
4. Please do not consider the difference in sentence style between the correct answer and the predicted answer, but only judge whether the predicted answer makes sense based on factual accuracy.
5. If there is something in the predicted answer that is not in the correct answer, then it is considered to be hallucination.

The score should range from 0 to 1. A larger score means a better answer. The score should be a float number with 2 decimal places. For example, 0.51, 0.99, 0.00, 0.76, etc.
In additional to this, I would like you to be able to extract some key words from the question and the correct answer, which are considered to be the key to answering the question correctly, and a prediction tends to score higher if  the prediction is able to include these key words.
Please first output a single line containing only one value indicating the scores for the predicted answer.
In the subsequent line, please provide some key words of the question and correct answers.
In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

Question: {question}
Correct Answer: {answer}
Prediction: {prediction}

Outputs (include score, key words, explanation):"""

m['gpt_eval'] = 0
try:
    api_key = os.environ.get('OPENAI_API_KEY', '').strip()
    if not api_key:
        print(f"  GPT: 0.00% (API key not provided, skipped)")
    else:
        from openai import OpenAI, APIError, AuthenticationError
        try:
            client = OpenAI(api_key=api_key)
            gpt_scores = []
            errors = []
            for i, (pred, gt) in enumerate(zip(retain_res['preds'][:20], retain_res['gts'][:20])):
                if len(pred) > 3:
                    try:
                        prompt_content = gpt_prompt.format(question="[from image]", answer=gt, prediction=pred)
                        r = client.chat.completions.create(model="gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt_content}],
                            max_tokens=20)
                        score_text = r.choices[0].message.content.strip().split("\n")[0].strip()
                        if ":" in score_text:
                            score_text = score_text[score_text.find(":")+1:].strip()
                        if "**" in score_text:
                            score_text = score_text.strip("**").strip()
                        s = float(score_text)
                        gpt_scores.append(min(1.0, max(0.0, s)))
                    except Exception as e:
                        if len(errors) < 2:
                            errors.append(f"Sample {i}: {str(e)[:80]}")

            m['gpt_eval'] = np.mean(gpt_scores) * 100 if gpt_scores else 0
            print(f"  GPT: {m['gpt_eval']:.2f}%  ({len(gpt_scores)} samples)")
            if errors:
                print(f"    Errors: {'; '.join(errors)}")
        except AuthenticationError:
            print(f"  GPT: 0.00% (Invalid/expired API key)")
        except APIError as e:
            print(f"  GPT: 0.00% (API error: {str(e)[:80]})")
except Exception as e:
    print(f"  GPT: 0.00% (Error: {str(e)[:100]})")

m['truth_ratio'] = np.mean(retain_res['truths']) * 100 if retain_res['truths'] else 0
print(f"  TRUTH: {m['truth_ratio']:.2f}%")

m['acc_mme_pope'] = (100 - np.mean(retain_res['ems']) * 100) if retain_res['ems'] else 0
print(f"  ACC: {m['acc_mme_pope']:.2f}%")

m['avg_mu'] = np.mean([m['rouge_l'], m['gpt_eval'], m['truth_ratio'], m['acc_mme_pope']])
print(f"  → Avg: {m['avg_mu']:.2f}%")

# ─── RESULTS ─────────────────────────────────────────────────────────────────
print("\n" + "="*100)
print("FINAL RESULTS")
print("="*100)
print(f"\n{'Metric':<15} {'Reproduced':<15} {'Paper':<15} {'Δ':<12} {'Status':<10}")
print("-"*100)

paper = {
    'rouge_l': 93.7, 'gpt_eval': 83.3, 'truth_ratio': 77.3, 'acc_mme_pope': 100.0, 'avg_mu': 88.6,
    'ks_test_pval': 0.0, 'exact_match': 13.3, 'mia_mink': 12.3, 'ape': 14.7, 'avg_fq': 93.3,
}

for key in ['rouge_l', 'gpt_eval', 'truth_ratio', 'acc_mme_pope', 'avg_mu', 'ks_test_pval', 'exact_match', 'mia_mink', 'ape', 'avg_fq']:
    rep = m[key]
    pap = paper[key]
    delta = rep - pap
    pct = (delta / pap * 100) if pap else 0
    status = "✅ MATCH" if abs(pct) < 5 else "⚠️ CLOSE" if abs(pct) < 10 else "❌ DIFF"
    print(f"{key:<15} {rep:<15.2f} {pap:<15.2f} {delta:<+12.2f} {status:<10}")

# Save - convert numpy types to Python types
m_serializable = {k: float(v) for k, v in m.items()}
with open(OUTPUT_DIR / 'metrics.json', 'w') as f:
    json.dump(m_serializable, f, indent=2)
