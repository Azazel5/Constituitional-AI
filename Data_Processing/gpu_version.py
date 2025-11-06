#!/usr/bin/env python3
"""
Constitutional AI Training Data Generation
Optimized for HPC batch jobs with A100 GPU
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path
import random
from datetime import datetime
import time
import argparse

from huggingface_hub import login
from config import HF_TOKEN

# ============================================
# Setup
# ============================================


# Paths
BASE_DIR = Path("CAI")
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


current_script_dir = Path(__file__).resolve().parent
parent_dir = current_script_dir.parent


CHECKPOINT_FILE = DATA_DIR / 'constitutional_training_data_mistral.jsonl'
PROGRESS_FILE = DATA_DIR / 'progress_mistral.json'
ERROR_LOG = DATA_DIR / 'errors_mistral.log'

# Create logs directory for batch outputs
LOG_DIR = DATA_DIR / 'logs'
LOG_DIR.mkdir(exist_ok=True)

print("="*80)
print("Constitutional AI Training Data Generation")
print("="*80)
print(f"\nConfiguration:")
print(f"  Data directory: {DATA_DIR}")
print(f"  Checkpoint file: {CHECKPOINT_FILE}")
print(f"  Progress file: {PROGRESS_FILE}")
print()

# ============================================
# GPU Check
# ============================================

print("Checking GPU...")
if not torch.cuda.is_available():
    print("❌ ERROR: No GPU available!")
    sys.exit(1)

print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"  CUDA Version: {torch.version.cuda}")
print()

# ============================================
# HuggingFace Authentication
# ============================================

if HF_TOKEN:
    print("Authenticating with HuggingFace...")
    
    try:
        login(token=HF_TOKEN)
        print("✓ Authenticated with HuggingFace")
        
    except Exception as e:
        print(f"⚠️  HuggingFace auth failed: {e}")

else:
    print("⚠️  No HF token provided, skipping authentication")

print()

# ============================================
# Load Model
# ============================================

print("Loading model...")
start_time = time.time()

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token
print("  ✓ Tokenizer loaded")

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float16,
).to('cuda:0')

print(f"  ✓ Model loaded in {time.time() - start_time:.1f}s")
print(f"  Device: {model.device}")
print()

# ============================================
# Generation Function
# ============================================

def generate_response(messages, max_tokens=512):
    """Generate response using loaded model"""
    # Format messages as text
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            prompt += f"{content}\n\n"
        elif role == "assistant":
            prompt += f"{content}\n\n"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    return response.strip()

# ============================================
# Checkpoint Functions
# ============================================

def load_progress():
    """Load progress"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"completed": 0, "timestamp": str(datetime.now())}

def save_progress(completed, total):
    """Save progress"""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({
            "completed": completed,
            "total": total,
            "timestamp": str(datetime.now()),
            "last_index": completed - 1
        }, f, indent=2)
    os.sync()

def save_example(example_data):
    """Append example to checkpoint file"""
    with open(CHECKPOINT_FILE, 'a') as f:
        json.dump(example_data, f)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())

def log_error(index, prompt, error):
    """Log errors"""
    with open(ERROR_LOG, 'a') as f:
        f.write(f"[{datetime.now()}] {index}: {prompt[:50]}... | {str(error)}\n")
        f.flush()

# ============================================
# Load Data
# ============================================

print("Loading training data...")
data_file = parent_dir / "data.json"

if not data_file:
    print(f"❌ ERROR: Data file not found: {data_file}")
    sys.exit(1)

with open(data_file, 'r') as f:
    data = json.load(f)

# Extract the three arrays (matching your Colab structure)
harmful_prompts = data["red_team"]
critiques_list = data["critiques_or_principles"]
revisions_list = data["revision_request"]

# Create principle pairs
principles = list(zip(critiques_list, revisions_list))
print(f"✓ Data loaded:")
print(f"  Prompts: {len(harmful_prompts)}")
print(f"  Principles: {len(principles)}")

# Sanity checks
assert len(critiques_list) == len(revisions_list), "Critique/revision count mismatch!"
assert len(harmful_prompts) > 0, "No prompts found!"
assert len(principles) > 0, "No principles found!"

print(f"\n🔍 Sample data:")
print(f"  Prompt: {harmful_prompts[0][:80]}...")
print(f"  Critique: {principles[0][0][:80]}...")
print(f"  Revision: {principles[0][1][:80]}...")
print()

# ============================================
# Speed Test
# ============================================

print("Running speed test...")
test_start = time.time()
test_response = generate_response(
    [{"role": "user", "content": "Say hello in 10 words"}],
    max_tokens=50
)
test_time = time.time() - test_start

print(f"✓ Test generation: {test_time:.2f}s")
print(f"  Estimated time for {len(harmful_prompts)} examples: {(test_time * 3 * len(harmful_prompts)) / 3600:.1f} hours")
print(f"  (Assuming 3 generations per example)")
print()

# ============================================
# Main Generation Loop
# ============================================

progress = load_progress()
start_idx = progress["completed"]

print("="*80)
print("Starting Generation")
print("="*80)
print(f"Starting from: {start_idx}/{len(harmful_prompts)}")
print(f"Checkpoint file: {CHECKPOINT_FILE}")
print()

generation_start = time.time()
successful = 0
failed = 0

for i in range(start_idx, len(harmful_prompts)):
    prompt = harmful_prompts[i]
    
    try:
        iter_start = time.time()
        print(f"[{i+1}/{len(harmful_prompts)}] Processing...", end=' ', flush=True)
        
        # Select random principle
        principle, revision = random.choice(principles)
        revision += " Do not provide meta-commentary; respond as the assistant would."
        
        # Step 1: Initial response
        initial_response = generate_response([{"role": "user", "content": prompt}])
        
        # Step 2: Critique
        critique_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": initial_response},
            {"role": "user", "content": f"CritiqueRequest: {principle}\n\nCritique:"}
        ]
        critique = generate_response(critique_messages)
        
        # Step 3: Revision
        revision_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": initial_response},
            {"role": "user", "content": f"CritiqueRequest: {principle}"},
            {"role": "assistant", "content": critique},
            {"role": "user", "content": f"RevisionRequest: {revision}\n\nRevision:"}
        ]
        revised_response = generate_response(revision_messages)
        
        # Save
        example_data = {
            "prompt": prompt,
            "initial_response": initial_response,
            "principle_used": principle,
            "critique": critique,
            "revision": revised_response,
            "index": i,
            "timestamp": str(datetime.now())
        }
        
        save_example(example_data)
        
        # Update progress periodically
        if (i + 1) % 10 == 0:
            save_progress(i + 1, len(harmful_prompts))
        
        iter_time = time.time() - iter_start
        successful += 1
        
        # Estimate remaining time
        avg_time = (time.time() - generation_start) / (i - start_idx + 1)
        remaining = (len(harmful_prompts) - i - 1) * avg_time
        
        print(f"✓ ({iter_time:.1f}s) | ETA: {remaining/3600:.1f}h")
        
    except Exception as e:
        failed += 1
        print(f"✗ Error: {e}")
        log_error(i, prompt, e)
        continue

# Final save
save_progress(len(harmful_prompts), len(harmful_prompts))

# ============================================
# Summary
# ============================================

total_time = time.time() - generation_start

print()
print("="*80)
print("Generation Complete!")
print("="*80)
print(f"Total time: {total_time/3600:.2f} hours")
print(f"Successful: {successful}/{len(harmful_prompts)}")
print(f"Failed: {failed}")
print(f"Average per example: {total_time/successful:.1f}s")
print()
print(f"Output file: {CHECKPOINT_FILE}")
print(f"Progress file: {PROGRESS_FILE}")
if failed > 0:
    print(f"Error log: {ERROR_LOG}")
print("="*80)