"""
Evaluate a single model checkpoint and save the 4 metrics needed for Figure 2:
  - rouge_l       (Model Utility)
  - gpt_eval      (Model Utility)
  - exact_match   (Forget Quality)
  - mia_mink      (Forget Quality)

Usage (in Colab):
  !python eval_step_level.py \
      --model_path /content/stage2_ga/checkpoint-12 \
      --method ga \
      --step 12 \
      --out_dir /content/drive/MyDrive/fiubench_checkpoints/step_eval
"""

import argparse, json, math, os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, LlavaForConditionalGeneration, CLIPImageProcessor
from peft import LoraConfig, get_peft_model
from PIL import Image

# ── args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--model_path',  required=True)
parser.add_argument('--tokenizer_path', default='/content/stage1_final',
                    help='Stage1 path for tokenizer (stage2 checkpoints are LoRA only)')
parser.add_argument('--method',      required=True, choices=['ga','gd','kl','po','retain','base'])
parser.add_argument('--step',        required=True, type=int)
parser.add_argument('--out_dir',     default='/content/drive/MyDrive/fiubench_checkpoints/step_eval')
parser.add_argument('--dataset_path', default='/content/FIUBench_Reproducing/FIUBench/dataset/full.json')
parser.add_argument('--split_path',   default='/content/FIUBench_Reproducing/FIUBench/dataset/split.json')
parser.add_argument('--openai_key',   default='')
parser.add_argument('--max_new_tokens', type=int, default=50)
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Path(args.out_dir).mkdir(parents=True, exist_ok=True)

# ── model ────────────────────────────────────────────────────────────────────
print(f"Loading model from {args.model_path} ...")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

ckpt_pt = Path(args.model_path) / 'checkpoint.pt'
if ckpt_pt.exists():
    # LoRA checkpoint — load base model from tokenizer_path, apply LoRA, load weights
    base = LlavaForConditionalGeneration.from_pretrained(
        args.tokenizer_path, attn_implementation="sdpa", torch_dtype=torch.bfloat16
    )
    lora_config = LoraConfig(
        r=128, lora_alpha=256,
        target_modules=r'.*language_model.*\.(up_proj|k_proj|down_proj|v_proj|q_proj|o_proj|gate_proj)',
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(base, lora_config)
    checkpoint = torch.load(str(ckpt_pt), map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    model = model.merge_and_unload()
else:
    # Full HuggingFace model directory (step 0 = stage1_final)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path, attn_implementation="sdpa", torch_dtype=torch.bfloat16
    )

model = model.to(DEVICE)
model.eval()
print(f"  ✅ Loaded on {DEVICE}")

os.chdir('/content/FIUBench_Reproducing/FIUBench')

# ── data ─────────────────────────────────────────────────────────────────────
with open(args.dataset_path) as f:
    full_data = [json.loads(line) for line in f if line.strip()]
with open(args.split_path) as f:
    splits = json.load(f)

forget_data = [d for d in full_data if d['unique_id'] in set(splits['forget5'])]
retain_data  = [d for d in full_data if d['unique_id'] in set(splits['retain5'])]
print(f"  forget5={len(forget_data)}, retain5={len(retain_data)}")

# ── helpers ──────────────────────────────────────────────────────────────────
def compute_mink(logits, labels):
    try:
        labels_clean = labels[labels != -100][1:].unsqueeze(0)
        logits_aligned = logits[:, -labels_clean.shape[1]-1: -1, :]
        if logits_aligned.dtype == torch.bfloat16:
            logits_aligned = logits_aligned.float()
        log_probs = F.log_softmax(logits_aligned[0], dim=-1)
        labels_idx = labels_clean[0].unsqueeze(-1)
        token_log_probs = log_probs.gather(dim=-1, index=labels_idx).squeeze(-1)
        weights = [0.3, 0.3, 0.2, 0.1, 0.1]
        scores = []
        for ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
            k = max(1, int(len(token_log_probs) * ratio))
            topk = np.sort(token_log_probs.cpu().numpy())[:k]
            s = np.exp(np.mean(topk))
            scores.append(s if not math.isnan(s) else 0.0)
        return sum(s * w for s, w in zip(scores, weights))
    except Exception:
        return 0.0

def eval_exact_match(pred, gt, keywords):
    if not keywords:
        return 0.0
    score = sum(1.0 / len(keywords) for k in keywords if k.lower() in pred.lower())
    return min(1.0, score)

def run_split(data, split_name):
    preds, gts, ems, minks = [], [], [], []
    gt_losses, perturb_losses_all = [], []

    for item in tqdm(data, desc=split_name):
        try:
            img_path = Path('.') / item['image_path']
            if not img_path.exists(): continue
            img = Image.open(img_path).convert('RGB')
            pix = image_processor(img, return_tensors='pt')['pixel_values'].to(DEVICE, dtype=torch.bfloat16)
            qa = item.get('qa_list', [{}])[0]
            q, a = qa.get('question'), qa.get('answer')
            keywords = qa.get('keywords', [])
            para_qs = qa.get('paraphrased_question', [q])
            perturb_as = qa.get('perturbed_answer', [])
            if not q or not a: continue

            # ── generation + EM ──────────────────────────────────────────
            prompt = f"<|user|>\n<image>\n{q.capitalize()}<|end|>\n<|assistant|>\n"
            inp = tokenizer(prompt, return_tensors='pt', padding=True).to(DEVICE)
            with torch.no_grad():
                out = model(input_ids=inp['input_ids'], attention_mask=inp['attention_mask'],
                            pixel_values=pix, labels=inp['input_ids'])
                if split_name == 'forget5':
                    minks.append(compute_mink(out.logits, inp['input_ids']))
                gen = model.generate(input_ids=inp['input_ids'], attention_mask=inp['attention_mask'],
                                     pixel_values=pix, max_new_tokens=args.max_new_tokens, do_sample=False)
            pred = tokenizer.decode(gen[0, inp['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
            preds.append(pred); gts.append(a)
            ems.append(eval_exact_match(pred, a, keywords))

            # ── truth ratio (paraphrased Q) ───────────────────────────────
            if perturb_as and para_qs:
                pq = para_qs[0] if isinstance(para_qs, list) else para_qs
                inp_gt = tokenizer(
                    f"<|user|>\n<image>\n{pq.capitalize()}<|end|>\n<|assistant|>\n{a.capitalize()}",
                    return_tensors='pt', padding=True).to(DEVICE)
                with torch.no_grad():
                    out_gt = model(input_ids=inp_gt['input_ids'], attention_mask=inp_gt['attention_mask'],
                                   pixel_values=pix, labels=inp_gt['input_ids'])
                    gt_loss = out_gt.loss.item() if out_gt.loss else float('nan')
                p_losses = []
                for pa in perturb_as[:3]:
                    inp_p = tokenizer(
                        f"<|user|>\n<image>\n{pq.capitalize()}<|end|>\n<|assistant|>\n{pa.capitalize()}",
                        return_tensors='pt', padding=True).to(DEVICE)
                    with torch.no_grad():
                        out_p = model(input_ids=inp_p['input_ids'], attention_mask=inp_p['attention_mask'],
                                      pixel_values=pix, labels=inp_p['input_ids'])
                        if out_p.loss: p_losses.append(out_p.loss.item())
                if p_losses and not math.isnan(gt_loss):
                    gt_losses.append(gt_loss)
                    perturb_losses_all.append(p_losses)
        except Exception as e:
            pass

    return dict(preds=preds, gts=gts, ems=ems, minks=minks,
                gt_losses=gt_losses, perturb_losses=perturb_losses_all)

# ── evaluate ─────────────────────────────────────────────────────────────────
forget_res = run_split(forget_data, 'forget5')
retain_res  = run_split(retain_data,  'retain5')

# ── ROUGE-L ──────────────────────────────────────────────────────────────────
scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
rouges = [scorer_obj.score(gt, pred)['rougeL'].recall
          for pred, gt in zip(retain_res['preds'], retain_res['gts']) if gt]
rouge_l = float(np.mean(rouges)) if rouges else 0.0

# ── GPT eval ──────────────────────────────────────────────────────────────────
gpt_eval = 0.0
api_key = args.openai_key or os.environ.get('OPENAI_API_KEY', '')
if api_key:
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
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        scores = []
        for pred, gt in zip(retain_res['preds'][:20], retain_res['gts'][:20]):
            if len(pred) > 3:
                try:
                    r = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": gpt_prompt.format(
                            question="[from image]", answer=gt, prediction=pred)}],
                        max_tokens=20)
                    txt = r.choices[0].message.content.strip().split("\n")[0].strip()
                    if ":" in txt: txt = txt[txt.find(":")+1:].strip()
                    txt = txt.strip("**").strip()
                    scores.append(min(1.0, max(0.0, float(txt))))
                except Exception:
                    pass
        gpt_eval = float(np.mean(scores)) if scores else 0.0
    except Exception:
        pass

# ── Exact Match + MINK ───────────────────────────────────────────────────────
exact_match = float(np.mean(forget_res['ems'])) if forget_res['ems'] else 0.0
mia_mink    = float(np.mean(forget_res['minks'])) if forget_res['minks'] else 0.0

# ── save ─────────────────────────────────────────────────────────────────────
result = {
    'method': args.method,
    'step':   args.step,
    'rouge_l':     rouge_l,
    'gpt_eval':    gpt_eval,
    'exact_match': exact_match,
    'mia_mink':    mia_mink,
}
out_path = Path(args.out_dir) / f"{args.method}_step{args.step:04d}.json"
with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n{'='*60}")
print(f"Method: {args.method}  Step: {args.step}")
print(f"  ROUGE-L    : {rouge_l*100:.1f}%")
print(f"  GPT        : {gpt_eval*100:.1f}%")
print(f"  Exact Match: {exact_match*100:.1f}%")
print(f"  MIA (MINK) : {mia_mink*100:.1f}%")
print(f"Saved → {out_path}")
