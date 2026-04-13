"""
Monitoring script for Stage 1 fine-tuning.
Loads checkpoint at specified step and evaluates Rouge-L + GPT-Eval on forget set.
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm

# Add FIUBench to path
sys.path.insert(0, str(Path(__file__).parent.parent / "FIUBench"))

from transformers import AutoTokenizer, LlavaForConditionalGeneration, AutoProcessor
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer
from data_generation.api import GPTEvaluator
from data_module import MMDatasetQA, custom_data_collator
from utils import get_model_identifiers_from_yaml
import json as json_module


def load_checkpoint(checkpoint_path, model_id, device="cuda"):
    """Load model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None, None

    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None


def compute_rouge_l(predictions, references):
    """Compute Rouge-L scores between predictions and references."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []

    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores.append(score['rougeL'].fmeasure)

    return sum(scores) / len(scores) if scores else 0.0


def compute_gpt_eval(predictions, references, api_key, sample_size=50):
    """Compute GPT-Eval scores (sampled to reduce API calls)."""
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Skipping GPT-Eval.")
        return None

    agent = GPTEvaluator(api_key=api_key, model="gpt-4o-mini", max_tokens=20)

    # Sample for efficiency
    indices = list(range(min(sample_size, len(predictions))))
    sampled_preds = [predictions[i] for i in indices]
    sampled_refs = [references[i] for i in indices]

    scores = []
    for pred, ref in tqdm(zip(sampled_preds, sampled_refs), total=len(sampled_preds), desc="GPT-Eval"):
        try:
            score = agent.score(predicted_answer=pred, correct_answer=ref)
            scores.append(score)
        except Exception as e:
            print(f"Error in GPT-Eval: {e}")
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0


def monitor_checkpoint(checkpoint_path, step, model_id, forget_data_path, output_path, sample_size=100):
    """Monitor a single checkpoint."""
    print(f"\n{'='*60}")
    print(f"Step {step}: Monitoring {checkpoint_path}")
    print(f"{'='*60}")

    # Load checkpoint
    model, tokenizer = load_checkpoint(checkpoint_path, model_id)
    if model is None:
        return None

    # Load forget set sample
    with open(forget_data_path, 'r') as f:
        all_data = json.load(f)

    # Sample data
    import random
    random.seed(0)
    sample_indices = random.sample(range(len(all_data)), min(sample_size, len(all_data)))
    sample_data = [all_data[i] for i in sample_indices]

    # Create dataloader
    dataset = MMDatasetQA(
        data=sample_data,
        image_folder="./dataset/images",
        model_id=model_id,
        split="test"
    )
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=custom_data_collator,
        num_workers=0
    )

    # Generate predictions
    predictions = []
    references = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            # Extract question and answer
            questions = batch.pop("question", [])
            answers = batch.pop("answer", [])

            # Move to device
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(model.device)

            # Generate
            outputs = model.generate(
                **batch,
                max_new_tokens=50,
                do_sample=False
            )

            # Decode
            for output in outputs:
                pred_text = tokenizer.decode(output, skip_special_tokens=True)
                predictions.append(pred_text)

            references.extend(answers)

    # Compute metrics
    print("\nComputing metrics...")
    rouge_l = compute_rouge_l(predictions, references)
    print(f"Rouge-L: {rouge_l:.4f}")

    gpt_eval = compute_gpt_eval(
        predictions,
        references,
        api_key=os.getenv("OPENAI_API_KEY", ""),
        sample_size=min(50, len(predictions))
    )
    if gpt_eval is not None:
        print(f"GPT-Eval: {gpt_eval:.4f}")

    # Log results
    result = {
        "step": step,
        "checkpoint": checkpoint_path,
        "rouge_l": rouge_l,
        "gpt_eval": gpt_eval,
        "sample_size": len(predictions)
    }

    # Append to results file
    results = []
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            results = json.load(f)

    results.append(result)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Monitor Stage 1 fine-tuning")
    parser.add_argument("--checkpoint_dir", type=str, default="models", help="Directory containing checkpoints")
    parser.add_argument("--model_id", type=str, default="xtuner/llava-phi-3-mini-hf", help="Base model ID")
    parser.add_argument("--forget_data", type=str, default="dataset/forget1.json", help="Forget set data path")
    parser.add_argument("--output", type=str, default="results/stage1_monitoring.json", help="Output results path")
    parser.add_argument("--sample_size", type=int, default=100, help="Sample size for monitoring")
    parser.add_argument("--step", type=int, help="Specific step to monitor (if not specified, monitors latest)")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # If step specified, monitor that specific checkpoint
    if args.step:
        checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint-{args.step}")
        monitor_checkpoint(
            checkpoint_path,
            args.step,
            args.model_id,
            args.forget_data,
            args.output,
            args.sample_size
        )
    else:
        # Monitor all checkpoints in directory
        checkpoints = sorted(
            [d for d in os.listdir(args.checkpoint_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1])
        )

        for checkpoint in checkpoints:
            step = int(checkpoint.split("-")[1])
            checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint)
            monitor_checkpoint(
                checkpoint_path,
                step,
                args.model_id,
                args.forget_data,
                args.output,
                args.sample_size
            )


if __name__ == "__main__":
    main()
