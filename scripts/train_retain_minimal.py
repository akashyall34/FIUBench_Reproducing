"""
Minimal Retain model finetuning script for LLaVA-Phi on FIUBench.
Same as Stage 1 but trains only on retain5 subset.
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

from transformers import AutoTokenizer, LlavaForConditionalGeneration, CLIPImageProcessor

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATASET_PATH = '/content/FIUBench_Reproducing/FIUBench/dataset/full.json'
SPLIT_PATH = '/content/FIUBench_Reproducing/FIUBench/dataset/split.json'
MODEL_ID = 'xtuner/llava-phi-3-mini-hf'
OUTPUT_DIR = Path('/content/retain_model_minimal')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Paper hyperparams (same as Stage 1)
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
    def __init__(self, data_path, split_path, split_name='retain5', max_length=512):
        """Load FIUBench data filtered by split."""
        with open(data_path) as f:
            self.data = [json.loads(line) for line in f if line.strip()]

        if split_name != 'full':
            with open(split_path) as f:
                splits = json.load(f)
            split_ids = set(splits[split_name])
            self.data = [d for d in self.data if d['unique_id'] in split_ids]

        self.max_length = max_length
        print(f"Loaded {len(self.data)} samples for split '{split_name}'")

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
    """Prepare batch for model."""
    images, input_ids_list, labels_list = [], [], []

    for item in batch:
        try:
            img_path = Path('.') / item['image_path']
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert('RGB')
            pix = image_processor(img, return_tensors='pt')['pixel_values'][0]
            images.append(pix)

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

    max_len = max(ids.shape[0] for ids in input_ids_list)
    input_ids_padded = torch.zeros(len(input_ids_list), max_len, dtype=torch.long)
    labels_padded = torch.full((len(labels_list), max_len), -100, dtype=torch.long)

    for i, (inp, lbl) in enumerate(zip(input_ids_list, labels_list)):
        input_ids_padded[i, :len(inp)] = inp
        labels_padded[i, :len(lbl)] = lbl

    pixel_values = torch.stack(images)

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
dataset = FIUBenchDataset(DATASET_PATH, SPLIT_PATH, split_name='retain5', max_length=MAX_LENGTH)
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
print(f"RETAIN MODEL FINETUNING (Minimal)")
print(f"LR={LR}, Batch={BATCH_SIZE}, Accum={GRAD_ACCUM}, Epochs={EPOCHS}")
print("="*80 + "\n")

model.train()
optimizer.zero_grad()
global_step = 0

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for step, batch in enumerate(pbar):
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
        epoch_loss += loss.item() * GRAD_ACCUM
        num_batches += 1

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            pbar.set_postfix({'loss': f"{epoch_loss/num_batches:.4f}"})

    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
    print(f"Epoch {epoch+1} — Avg Loss: {avg_loss:.4f}\n")

# ─── SAVE CHECKPOINT ─────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"Saving checkpoint to {OUTPUT_DIR}")
print(f"{'='*80}\n")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Retain model finetuning complete!")
print(f"Checkpoint saved to: {OUTPUT_DIR}")
