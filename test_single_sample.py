#!/usr/bin/env python3
"""
Test pipeline: Load 1 sample → Run inference → Compute metrics (ROUGE-L + GPT-eval)
"""
import os
import sys
import torch
import json
from pathlib import Path
from rouge_score import rouge_scorer

# Setup
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "FIUBench"))
os.chdir(str(PROJECT_ROOT / "FIUBench"))

print("\n" + "="*80)
print("VERIFY PIPELINE: One Sample → Inference → Metrics")
print("="*80)

# Load model & processor
print("\n[1/5] Loading model...")
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image

model_id = "xtuner/llava-phi-3-mini-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)
device = model.device
print(f"  ✅ Model loaded on {device}")

# Load dataset
print("\n[2/5] Loading dataset...")
from data_module import MMDatasetQA

data_path = PROJECT_ROOT / "FIUBench" / "dataset" / "full.json"
dataset = MMDatasetQA(
    data_path=str(data_path),
    split="full",
    max_length=512,
    model_family="llava-phi"
)
print(f"  ✅ Dataset: {len(dataset)} samples")

# Get first sample
print("\n[3/5] Getting first sample...")
with open(data_path) as f:
    json_items = [json.loads(line) for line in f if line.strip()]
first_item = json_items[0]

# Load image
image_path = PROJECT_ROOT / "FIUBench" / first_item['image_path']
image = Image.open(image_path).convert('RGB')
question = first_item['conversations'][0]['value'].replace('<image>\n', '').strip()
ground_truth = first_item['conversations'][1]['value'].strip()

print(f"  ✅ Image: {image_path.name}")
print(f"     Question: {question[:80]}...")
print(f"     Ground truth: {ground_truth[:100]}...")

# Run inference
print("\n[4/5] Running inference...")
inputs = processor(images=image, text=question, return_tensors="pt").to(device)
inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        use_cache=True
    )

prediction = processor.decode(output_ids[0], skip_special_tokens=True)
if question in prediction:
    prediction = prediction.split(question)[-1].strip()

print(f"  ✅ Prediction: {prediction[:100]}...")

# Compute ROUGE-L
print("\n[5/5] Computing metrics...")
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
rouge_score = scorer.score(ground_truth, prediction)['rougeL'].recall

print(f"  ✅ ROUGE-L recall: {rouge_score:.3f}")

# Optional: GPT-Eval
api_key = os.environ.get("OPENAI_API_KEY", "")
if api_key:
    print("\n     Running GPT-Eval...")
    from openai import OpenAI
    import re

    prompt = f"""Score factual accuracy 0-1.
Question: {question}
Correct: {ground_truth}
Prediction: {prediction}
Score:"""

    try:
        resp = OpenAI(api_key=api_key).chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        gpt_score = float(re.findall(r'\d+\.\d+', resp.choices[0].message.content)[0])
        print(f"  ✅ GPT-Eval score: {gpt_score:.2f}/1.00")
    except Exception as e:
        print(f"  ⚠️  GPT-Eval failed: {e}")
else:
    print("\n     (Set OPENAI_API_KEY to run GPT-Eval)")

print("\n" + "="*80)
print("✅ PIPELINE VERIFIED")
print("="*80)
print(f"\nSummary:")
print(f"  Question:     {question[:60]}...")
print(f"  Ground truth: {ground_truth[:60]}...")
print(f"  Prediction:   {prediction[:60]}...")
print(f"  ROUGE-L:      {rouge_score:.3f}")
if api_key:
    print(f"  GPT-Eval:     {gpt_score:.2f}")
