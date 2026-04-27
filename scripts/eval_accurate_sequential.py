import argparse
import json
import math
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from scipy.stats import hmean
from rouge_score import rouge_scorer

from transformers import AutoTokenizer, LlavaForConditionalGeneration, CLIPImageProcessor
from peft import get_peft_model, LoraConfig

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_NEW_TOKENS = 50
LORA_TARGET = r'.*language_model.*\.(up_proj|k_proj|linear_2|down_proj|v_proj|q_proj|o_proj|gate_proj|linear_1)'


def _collect_lora_chain(checkpoint_dir):
    """Walk cfg.yaml links to collect [oldest, ..., newest] checkpoint.pt paths."""
    import yaml as _yaml
    chain = []
    current = checkpoint_dir
    for _ in range(20):
        ckpt_pt = os.path.join(current, 'checkpoint.pt')
        if not (os.path.isdir(current) and os.path.exists(ckpt_pt)):
            break
        chain.append(ckpt_pt)
        cfg_yaml = os.path.join(current, 'cfg.yaml')
        if not os.path.exists(cfg_yaml):
            break
        with open(cfg_yaml) as f:
            saved = _yaml.safe_load(f)
        current = saved.get('model_path', '')
    chain.reverse()
    return chain


def _merge_lora_chain(model, lora_chain):
    """Merge all LoRA checkpoints in chain order (oldest first) via delta = B @ A."""
    for ckpt_path in lora_chain:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        for key, param in ckpt.items():
            if 'lora_A' not in key and 'lora_B' not in key:
                continue
            module_key = key.replace('base_model.', '', 1)
            if '.lora_A.default.weight' in key:
                base_key = module_key.replace('.lora_A.default.weight', '')
                is_a = True
            elif '.lora_B.default.weight' in key:
                base_key = module_key.replace('.lora_B.default.weight', '')
                is_a = False
            else:
                continue
            base_key = base_key.replace('language_model.model.', 'language_model.')
            parts = base_key.split('.')
            target = model
            try:
                for part in parts:
                    target = target[int(part)] if part.isdigit() else getattr(target, part)
            except (AttributeError, IndexError, TypeError):
                continue
            if is_a:
                target._lora_a = param.to('cpu')
            else:
                if hasattr(target, '_lora_a') and hasattr(target, 'weight'):
                    A = target._lora_a.to(target.weight.dtype).to(target.weight.device)
                    B = param.to(target.weight.dtype).to(target.weight.device)
                    target.weight.data.add_(B @ A)
                    del target._lora_a
        print(f"  Merged: {ckpt_path}")


def load_model(stage1_path, checkpoint_dir):
    """Load stage1, optionally walk LoRA chain and merge all deltas, return eval model."""
    print("Loading tokenizer and image processor...")
    tokenizer = AutoTokenizer.from_pretrained(stage1_path)
    tokenizer.pad_token = tokenizer.eos_token
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

    print("Loading base model from stage1...")
    base = LlavaForConditionalGeneration.from_pretrained(
        stage1_path, attn_implementation="sdpa", torch_dtype=torch.bfloat16
    )

    if checkpoint_dir:
        lora_chain = _collect_lora_chain(checkpoint_dir)
        print(f"LoRA chain: {len(lora_chain)} checkpoint(s)")
        lora_cfg = LoraConfig(r=128, lora_alpha=256, target_modules=LORA_TARGET,
                              lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(base, lora_cfg)
        _merge_lora_chain(model, lora_chain)
        model = model.merge_and_unload()
    else:
        print("No checkpoint — evaluating baseline (stage1 only)")
        model = base

    model = model.to(DEVICE).eval()
    print(f"Model ready on {DEVICE}")
    return tokenizer, image_processor, model


def compute_mink(logits, labels):
    try:
        labels_clean = labels[labels != -100][1:].unsqueeze(0)
        logits_aligned = logits[:, -labels_clean.shape[1]-1:-1, :].float()
        log_probs = F.log_softmax(logits_aligned[0], dim=-1)
        token_log_probs = log_probs.gather(-1, labels_clean[0].unsqueeze(-1)).squeeze(-1)
        scores, weights = [], [0.3, 0.3, 0.2, 0.1, 0.1]
        for ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
            k = max(1, int(len(token_log_probs) * ratio))
            topk = np.sort(token_log_probs.cpu().numpy())[:k]
            v = np.mean(topk)
            scores.append(0.0 if (np.isnan(v) or np.isinf(v)) else min(np.exp(v), 1.0))
        return sum(s * w for s, w in zip(scores, weights))
    except Exception:
        return 0.0


def eval_exact_match(pred, gt, keywords):
    if not keywords:
        return 0.0
    score = sum(1.0 / len(keywords) for k in keywords if k.lower() in pred.lower())
    return min(1.0, score)


def run_eval(tokenizer, image_processor, model, data, split_name):
    from PIL import Image
    results = dict(preds=[], gts=[], ems=[], minked=[], apes=[],
                   truth_ratios_raw=[], gt_losses=[], perturb_losses=[], truths=[])

    for item in tqdm(data, desc=split_name):
        try:
            img_path = Path(os.getcwd()) / item['image_path']
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert('RGB')
            pix = image_processor(img, return_tensors='pt')['pixel_values'].to(DEVICE, dtype=torch.bfloat16)

            qa = item.get('qa_list', [{}])[0]
            q, a = qa.get('question'), qa.get('answer')
            keywords = qa.get('keywords', [])
            if not q or not a:
                continue
            para_qs = qa.get('paraphrased_question', [q])
            perturb_as = qa.get('perturbed_answer', [])

            prompt = f"<|user|>\n<image>\n{q.capitalize()}<|end|>\n<|assistant|>\n"
            inp = tokenizer(prompt, return_tensors='pt', padding=True).to(DEVICE)

            with torch.no_grad():
                out = model(input_ids=inp['input_ids'], attention_mask=inp['attention_mask'],
                            pixel_values=pix, labels=inp['input_ids'])
                if split_name == 'forget5' and out.logits is not None:
                    results['minked'].append(compute_mink(out.logits, inp['input_ids'][0]))

                gen = model.generate(input_ids=inp['input_ids'], attention_mask=inp['attention_mask'],
                                     pixel_values=pix, max_new_tokens=MAX_NEW_TOKENS,
                                     do_sample=False, pad_token_id=tokenizer.eos_token_id)
            pred = tokenizer.decode(gen[0, inp['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
            results['preds'].append(pred)
            results['gts'].append(a)
            results['ems'].append(eval_exact_match(pred, a, keywords))

            # APE (forget only)
            if split_name == 'forget5' and para_qs:
                para_q = para_qs[0] if isinstance(para_qs, list) else para_qs
                inp_pa = tokenizer(f"<|user|>\n<image>\n{para_q.capitalize()}<|end|>\n<|assistant|>\n",
                                   return_tensors='pt', padding=True).to(DEVICE)
                with torch.no_grad():
                    gen_pa = model.generate(input_ids=inp_pa['input_ids'],
                                            attention_mask=inp_pa['attention_mask'],
                                            pixel_values=pix, max_new_tokens=MAX_NEW_TOKENS,
                                            do_sample=False, pad_token_id=tokenizer.eos_token_id)
                pred_pa = tokenizer.decode(gen_pa[0, inp_pa['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
                results['apes'].append(eval_exact_match(pred_pa, a, keywords))

            # Truth ratio (both splits)
            if perturb_as and para_qs:
                para_q_t = para_qs[0] if isinstance(para_qs, list) else para_qs
                inp_gt = tokenizer(f"<|user|>\n<image>\n{para_q_t.capitalize()}<|end|>\n<|assistant|>\n{a.capitalize()}",
                                   return_tensors='pt', padding=True).to(DEVICE)
                with torch.no_grad():
                    out_gt = model(input_ids=inp_gt['input_ids'], attention_mask=inp_gt['attention_mask'],
                                   pixel_values=pix, labels=inp_gt['input_ids'])
                gt_loss = out_gt.loss.item() if out_gt.loss else float('nan')

                p_losses = []
                for pa in perturb_as[:3]:
                    inp_p = tokenizer(f"<|user|>\n<image>\n{para_q_t.capitalize()}<|end|>\n<|assistant|>\n{pa.capitalize()}",
                                      return_tensors='pt', padding=True).to(DEVICE)
                    with torch.no_grad():
                        out_p = model(input_ids=inp_p['input_ids'], attention_mask=inp_p['attention_mask'],
                                      pixel_values=pix, labels=inp_p['input_ids'])
                    if out_p.loss:
                        p_losses.append(out_p.loss.item())

                if p_losses and not math.isnan(gt_loss):
                    pm = float(np.mean(p_losses))
                    try:
                        results['truth_ratios_raw'].append(float(np.exp(pm - gt_loss)))
                    except Exception:
                        pass
                    results['gt_losses'].append(gt_loss)
                    results['perturb_losses'].append(p_losses)
                    if split_name == 'retain5':
                        try:
                            curr = np.exp(pm - gt_loss)
                            results['truths'].append(float(np.clip(max(0.0, 1.0 - 1.0 / curr), 0, 1)))
                        except Exception:
                            pass

        except Exception as e:
            pass

    return results


def aggregate(forget_res, retain_res):
    m = {}

    # KS-test
    fr = np.array([x for x in forget_res['truth_ratios_raw'] if not math.isnan(x)])
    rr = np.array([x for x in retain_res['truth_ratios_raw'] if not math.isnan(x)])
    if len(fr) > 0 and len(rr) > 0:
        _, pval = stats.ks_2samp(fr, rr)
        m['ks_test_pval'] = pval * 100
    else:
        m['ks_test_pval'] = 0.0

    m['exact_match'] = float(np.mean(forget_res['ems']) * 100) if forget_res['ems'] else 0.0
    m['mia_mink'] = float(np.mean(forget_res['minked']) * 100) if forget_res['minked'] else 0.0
    m['ape'] = float(np.mean(forget_res['apes']) * 100) if forget_res['apes'] else 0.0
    m['avg_fq'] = (0.5 * m['ks_test_pval'] / 100
                   + 0.15 * (1 - m['exact_match'] / 100)
                   + 0.2 * (1 - m['mia_mink'] / 100)
                   + 0.15 * (1 - m['ape'] / 100)) * 100

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouges = [scorer.score(gt, p)['rougeL'].recall
              for p, gt in zip(retain_res['preds'], retain_res['gts']) if gt]
    m['rouge_l'] = float(np.mean(rouges) * 100) if rouges else 0.0

    m['truth_ratio'] = float(np.mean(retain_res['truths']) * 100) if retain_res['truths'] else 0.0

    if retain_res['gt_losses'] and retain_res['perturb_losses']:
        acc_scores = []
        for gl, pls in zip(retain_res['gt_losses'], retain_res['perturb_losses']):
            gt_p = np.exp(-gl)
            acc_scores.append(float(gt_p / (gt_p + np.sum(np.exp(-np.array(pls))))))
        m['acc'] = float(np.mean(acc_scores) * 100)
    else:
        m['acc'] = 0.0

    mu_vals = [m['rouge_l'], m['truth_ratio'], m['acc']]
    m['avg_mu'] = float(hmean([max(v, 1e-9) for v in mu_vals]))

    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1_path', required=True)
    parser.add_argument('--checkpoint_dir', default=None,
                        help='Leave empty for baseline (stage1 only)')
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--split_path', required=True)
    parser.add_argument('--fiubench_dir', required=True)
    args = parser.parse_args()

    os.chdir(args.fiubench_dir)

    tokenizer, image_processor, model = load_model(args.stage1_path, args.checkpoint_dir)

    with open(args.dataset_path) as f:
        raw = f.read().strip()
    try:
        full_data = json.loads(raw)
    except json.JSONDecodeError:
        full_data = [json.loads(line) for line in raw.splitlines() if line.strip()]

    with open(args.split_path) as f:
        splits = json.load(f)

    forget_ids = set(splits['forget5'])
    retain_ids = set(splits['retain5'])
    forget_data = [d for d in full_data if d['unique_id'] in forget_ids]
    retain_data = [d for d in full_data if d['unique_id'] in retain_ids]
    print(f"Dataset: {len(forget_data)} forget, {len(retain_data)} retain")

    forget_res = run_eval(tokenizer, image_processor, model, forget_data, 'forget5')
    retain_res = run_eval(tokenizer, image_processor, model, retain_data, 'retain5')

    m = aggregate(forget_res, retain_res)

    print("\n" + "="*60)
    print(f"RESULTS: {args.checkpoint_dir or 'Baseline (Stage1)'}")
    print("="*60)
    labels = {
        'ks_test_pval': 'KS-Test p-val ↑', 'exact_match': 'Exact Match ↓',
        'mia_mink': 'MINK ↓', 'ape': 'APE ↓', 'avg_fq': 'Avg FQ ↑',
        'rouge_l': 'ROUGE-L (retain) ↑', 'truth_ratio': 'Truth Ratio (retain) ↑',
        'acc': 'ACC (retain) ↑', 'avg_mu': 'Avg MU ↑',
    }
    for k, label in labels.items():
        print(f"  {label:<28} {m.get(k, 0):.2f}%")

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump({'checkpoint_dir': args.checkpoint_dir, 'metrics': m}, f, indent=2)
    print(f"\nSaved to {args.output_path}")


if __name__ == '__main__':
    main()
