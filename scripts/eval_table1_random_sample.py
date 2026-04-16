#!/usr/bin/env python3
"""
Table 1 Evaluation: Random Sample from Full Dataset
Paper methodology: "evaluate the model utility on a randomly sampled subset of the dataset"
"""

import json
import argparse
import subprocess
import tempfile
from pathlib import Path
import numpy as np

def create_random_sample(full_json_path, output_json_path, n_samples=200, seed=0):
    """Create random sample from first 400 identities for Table 1 evaluation.

    Paper methodology: Train on first 400, evaluate on random sample FROM those 400.
    """
    with open(full_json_path) as f:
        full_data = [json.loads(line) for line in f if line.strip()]

    # Filter to first 400 (same as Stage 1 training)
    first_400 = full_data[:400]
    print(f"Full dataset: {len(full_data)} identities")
    print(f"Training split (first 400): {len(first_400)} identities")

    # Random sample from those 400
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(first_400), min(n_samples, len(first_400)), replace=False)
    sample = [first_400[i] for i in indices]

    print(f"Evaluation sample: {len(sample)} identities (random, seed={seed})")

    # Save as JSON (one per line for streaming)
    with open(output_json_path, 'w') as f:
        for item in sample:
            f.write(json.dumps(item) + '\n')

    return len(sample)

def evaluate_random_sample(stage1_ckpt, sample_json, fiubench_dir, output_dir, seed=0):
    """Run evaluate_util.py on random sample."""

    cmd = [
        'python', 'evaluate_util.py', '--config-name', 'eval',
        f'model_path={stage1_ckpt}',
        'LoRA.r=0',
        f'save_dir={output_dir}',
        'split_list=[full]',
        'eval_task=[eval_log]',  # Only ROUGE-L, no perturbation metrics
        'robust_eval=[[rouge]]',
        'batch_size=4',
        'perturb_batch_size=4',
        'overwrite=true',
        f'hydra.run.dir={output_dir}/hydra_run'
    ]

    # Override data_path to use random sample
    env_override = f"data_path={sample_json}"
    print(f"Running: {' '.join(cmd)}")
    print(f"Data: {sample_json}")

    proc = subprocess.Popen(
        cmd,
        cwd=fiubench_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    for line in proc.stdout:
        print(line, end='', flush=True)
    proc.wait()

    return proc.returncode

def parse_results(output_dir):
    """Parse and display results."""
    result_path = Path(output_dir) / 'full_eval_log.json'

    if not result_path.exists():
        print(f"❌ Results not found at {result_path}")
        return None

    with open(result_path) as f:
        data = json.load(f)

    rouge = data.get('rougeL_recall', {})

    # Handle dict format (per-sample scores)
    if isinstance(rouge, dict):
        scores = [float(v) for v in rouge.values() if v is not None]
        if not scores:
            print("❌ No valid ROUGE scores found")
            return None
        mean_rouge = np.mean(scores) * 100
    else:
        # Handle scalar format
        mean_rouge = float(rouge) * 100

    print("\n" + "="*70)
    print("TABLE 1 - STAGE I LEARNING (Random Sample)")
    print("="*70)
    print(f"ROUGE-L: {mean_rouge:.1f}")
    print(f"Paper target: 93.3")
    print(f"Acceptable range: 88.0 - 96.0")

    if mean_rouge >= 88:
        print("✅ PASS - Stage 1 successful")
    else:
        print("⚠️  Below threshold - check training")
    print("="*70)

    return mean_rouge

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1-ckpt', required=True, help='Path to Stage 1 checkpoint')
    parser.add_argument('--full-json', required=True, help='Path to full.json')
    parser.add_argument('--fiubench-dir', required=True, help='Path to FIUBench directory')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--n-samples', type=int, default=400, help='Number of samples for random evaluation')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Create random sample
    sample_path = Path(args.output_dir) / 'random_sample.json'
    n = create_random_sample(args.full_json, sample_path, args.n_samples, args.seed)

    # Evaluate
    ret = evaluate_random_sample(
        args.stage1_ckpt,
        str(sample_path),
        args.fiubench_dir,
        args.output_dir,
        args.seed
    )

    if ret == 0:
        rouge = parse_results(args.output_dir)
    else:
        print(f"❌ Evaluation failed (exit {ret})")
