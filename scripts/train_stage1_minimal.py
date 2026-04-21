"""
Minimal Stage 1 finetuning script for LLaVA-Phi on FIUBench.
Paper hyperparams: lr=2e-5, batch=8, accum=16, epochs=10, max_length=512.
No config yaml, no bugs from paper's codebase. Pure PyTorch.
"""

import json
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import random
import numpy as np

from transformers import AutoTokenizer, LlavaForConditionalGeneration, CLIPImageProcessor

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATASET_PATH = '/content/FIUBench_Reproducing/FIUBench/dataset/full.json'
SPLIT_PATH = '/content/FIUBench_Reproducing/FIUBench/dataset/split.json'
MODEL_ID = 'xtuner/llava-phi-3-mini-hf'
OUTPUT_DIR = Path('/content/stage1_minimal')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Paper hyperparams
LR = 2e-5
BATCH_SIZE = 8
GRAD_ACCUM = 16
EPOCHS = 10
MAX_LENGTH = 512
SEED = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(SEED)
torch.manual_seed(SEED)

print(f"Device: {DEVICE}")
print(f"Output: {OUTPUT_DIR}\n")

# ─── DATASET ─────────────────────────────────────────────────────────────────
class FIUBenchDataset(Dataset):
    def __init__(self, data_path, split_path, split_name='full', max_length=512):
        """
        split_name: 'full' for Stage 1, 'retain' for retain model, 'forget5' for forget
        """
        with open(data_path) as f:
            self.data = [json.loads(line) for line in f if line.strip()]

        print(f"Total identities in dataset: {len(self.data)}")

        # Limit to 400 identities (match data_module.py behavior)
        self.data = self.data[:400]
        print(f"After 400-identity limit: {len(self.data)} identities")

        if split_name != 'full':
            with open(split_path) as f:
                splits = json.load(f)
            split_ids = set(splits[split_name])
            self.data = [d for d in self.data if d['unique_id'] in split_ids]
            print(f"After split filtering: {len(self.data)} identities")

        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        qa = item.get('qa_list', [{}])[0]
        q = qa.get('question', '')
        a = qa.get('answer', '')
        img_path = item.get('image_path', '')

        return {
            'image_path': img_path,
            'question': q,
            'answer': a,
            'unique_id': item.get('unique_id', '')
        }

# ─── COLLATE FUNCTION ────────────────────────────────────────────────────────
def collate_fn(batch, tokenizer, image_processor, device, max_length):
    """Prepare batch for model: images, input_ids, labels."""
    images, input_ids_list, labels_list = [], [], []

    for item in batch:
        try:
            # Load image
            img_path = Path('.') / item['image_path']
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert('RGB')
            pix = image_processor(img, return_tensors='pt')['pixel_values'][0]
            images.append(pix)

            # Tokenize: "<image>\nQuestion: {q}\nAnswer: {a}"
            prompt = f"<image>\nQuestion: {item['question']}\nAnswer: {item['answer']}"
            tokens = tokenizer(prompt, max_length=max_length, truncation=True,
                             return_tensors='pt', padding=False)
            input_ids = tokens['input_ids'][0]

            input_ids_list.append(input_ids)
            labels_list.append(input_ids.clone())
        except:
            continue

    if not images:
        return None

    # Pad input_ids and labels to same length
    max_len = max(ids.shape[0] for ids in input_ids_list)
    input_ids_padded = torch.zeros(len(input_ids_list), max_len, dtype=torch.long)
    labels_padded = torch.full((len(labels_list), max_len), -100, dtype=torch.long)

    for i, (inp, lbl) in enumerate(zip(input_ids_list, labels_list)):
        input_ids_padded[i, :len(inp)] = inp
        labels_padded[i, :len(lbl)] = lbl

    # Stack images
    pixel_values = torch.stack(images)  # [batch, 3, 336, 336]

    return {
        'pixel_values': pixel_values.to(device),
        'input_ids': input_ids_padded.to(device),
        'labels': labels_padded.to(device),
        'attention_mask': (input_ids_padded != 0).to(device),
    }

# ─── LOAD MODEL ──────────────────────────────────────────────────────────────
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, attn_implementation="sdpa", torch_dtype=torch.bfloat16
).to(DEVICE)
print("✅ Model loaded\n")

# ─── DATASET & DATALOADER ────────────────────────────────────────────────────
print("Loading dataset...")
dataset = FIUBenchDataset(DATASET_PATH, SPLIT_PATH, split_name='full', max_length=MAX_LENGTH)
dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=lambda b: collate_fn(b, tokenizer, image_processor, DEVICE, MAX_LENGTH),
    num_workers=0
)
print(f"✅ {len(dataloader)} batches\n")

# ─── OPTIMIZER ───────────────────────────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.0)
num_steps = len(dataloader) * EPOCHS // GRAD_ACCUM
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

# ─── TRAINING LOOP ───────────────────────────────────────────────────────────
print("="*80)
print(f"STAGE 1 FINETUNING (Minimal)")
print(f"LR={LR}, Batch={BATCH_SIZE}, Accum={GRAD_ACCUM}, Epochs={EPOCHS}")
print("="*80 + "\n")

model.train()
optimizer.zero_grad()
global_step = 0

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    num_updates = 0
    batch_losses = []
    accum_step = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in pbar:
        if batch is None:
            continue

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
            )
            loss = outputs.loss / GRAD_ACCUM

        loss.backward()
        batch_loss = outputs.loss.item()
        batch_losses.append(batch_loss)
        epoch_loss += batch_loss
        accum_step += 1

        if accum_step % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            num_updates += 1
            avg_batch_loss = np.mean(batch_losses[-GRAD_ACCUM:])
            pbar.set_postfix({'loss': f"{avg_batch_loss:.6f}"})

    avg_loss = epoch_loss / len(batch_losses) if batch_losses else 0
    print(f"Epoch {epoch+1} — Avg Loss: {avg_loss:.6f}  ({num_updates} updates)\n")

# ─── SAVE CHECKPOINT ─────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"Saving checkpoint to {OUTPUT_DIR}")
print(f"{'='*80}\n")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Stage 1 finetuning complete!")
print(f"Checkpoint saved to: {OUTPUT_DIR}")

# ─── EVALUATE ON RETAIN5 SPLIT ───────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"EVALUATING ON RETAIN5 SPLIT (ROUGE-L)")
print(f"{'='*80}\n")

try:
    from rouge_score import rouge_scorer

    # Load splits
    with open(SPLIT_PATH) as f:
        splits = json.load(f)

    # Load full data and filter to retain5
    with open(DATASET_PATH) as f:
        full_data = [json.loads(line) for line in f if line.strip()]

    full_data = full_data[:400]  # Match training data limit
    retain_data = [d for d in full_data if d['unique_id'] in set(splits.get('retain5', []))]

    print(f"Evaluating on {len(retain_data)} retain5 samples\n")

    # Reload model in eval mode
    eval_model = LlavaForConditionalGeneration.from_pretrained(
        OUTPUT_DIR, attn_implementation="sdpa", torch_dtype=torch.bfloat16
    ).to(DEVICE)
    eval_model.eval()

    # Generate predictions
    preds, gts = [], []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = []

    for item in tqdm(retain_data, desc="Evaluating"):
        try:
            # Load image
            img_path = Path('.') / item['image_path']
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert('RGB')
            pix = image_processor(img, return_tensors='pt')['pixel_values'][0]

            # Get question/answer
            qa = item.get('qa_list', [{}])[0]
            q = qa.get('question', '')
            a = qa.get('answer', '')
            if not q or not a:
                continue

            # Generate prediction
            prompt = f"<image>\nQuestion: {q}\nAnswer:"
            inp = tokenizer(prompt, return_tensors='pt', padding=True).to(DEVICE)

            with torch.no_grad():
                gen = eval_model.generate(
                    input_ids=inp['input_ids'],
                    attention_mask=inp['attention_mask'],
                    pixel_values=pix.unsqueeze(0).to(DEVICE),
                    max_new_tokens=50,
                    do_sample=False
                )
                pred = tokenizer.decode(gen[0, inp['input_ids'].shape[-1]:], skip_special_tokens=True).strip()

            # Compute ROUGE-L
            if a:
                rouge = scorer.score(a, pred)['rougeL'].recall
                rouge_scores.append(rouge)
                preds.append(pred)
                gts.append(a)
        except:
            continue

    # Print results
    if rouge_scores:
        mean_rouge = np.mean(rouge_scores) * 100
        print(f"\n{'='*80}")
        print(f"ROUGE-L: {mean_rouge:.2f}%")
        print(f"Samples evaluated: {len(rouge_scores)}")
        print(f"{'='*80}\n")
    else:
        print("⚠️  No valid samples evaluated")

except ImportError:
    print("⚠️  rouge_score not installed. Skipping ROUGE-L evaluation.")
except Exception as e:
    print(f"⚠️  Evaluation failed: {str(e)}")
