import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path
import random
from datetime import datetime
import time

from huggingface_hub import login
from config import HF_TOKEN

DRIVE_BASE = Path('CAI')

DATA_DIR = DRIVE_BASE / 'data'
DATA_DIR.mkdir(exist_ok=True)

CHECKPOINT_FILE = DATA_DIR / 'constitutional_training_data.jsonl'
PROGRESS_FILE = DATA_DIR / 'progress.json'
ERROR_LOG = DATA_DIR / 'errors.log'


# Check GPU
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")


print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


# Login to HuggingFace (if using gated models)
try:
    login(token=HF_TOKEN)
    print("✓ Authenticated with HuggingFace")
except:
    print("⚠️  No HF token found, skipping auth")


# Load model (takes ~5 minutes first time)
model_name = "Qwen/Qwen2.5-3B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,  # Use FP16 for speed
).to('cuda')

print("Model loaded successfully!")
print(f"Model device: {model.device}")


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
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def load_progress():
    """Load progress from Drive"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"completed": 0, "total": len(harmful_prompts), "timestamp": str(datetime.now())}

def save_progress(completed):
    """Save progress to Drive (instant persistence)"""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({
            "completed": completed,
            "total": len(harmful_prompts),
            "timestamp": str(datetime.now()),
            "last_index": completed - 1
        }, f, indent=2)

    # Force sync to Drive
    os.sync()

def save_example(example_data):
    """Append example to Drive checkpoint (instant write)"""
    with open(CHECKPOINT_FILE, 'a') as f:
        json.dump(example_data, f)
        f.write('\n')
        f.flush()  # Force write
        os.fsync(f.fileno())  # Force sync to Drive

def log_error(index, prompt, error):
    """Log errors to Drive"""
    with open(ERROR_LOG, 'a') as f:
        f.write(f"{index}: {prompt[:50]}... | {str(error)}\n")
        f.flush()
        os.fsync(f.fileno())

with open(DATA_DIR / "data.json", 'r') as f:
    data = json.load(f)

# Extract the three arrays
harmful_prompts = data["red_team"]
critiques_list = data["critiques_or_principles"]
revisions_list = data["revision_request"]

# Create principle pairs
principles = list(zip(critiques_list, revisions_list))

# Summary
print(f"\n📊 Data Summary:")
print(f"  • Prompts: {len(harmful_prompts)}")
print(f"  • Principles: {len(principles)}")
print(f"\n✓ Data loaded successfully!")

# Quick sanity check
assert len(critiques_list) == len(revisions_list), "Critique and revision counts don't match!"
assert len(harmful_prompts) > 0, "No prompts found!"
assert len(principles) > 0, "No principles found!"

print("\n🔍 Sample data:")
print(f"  Prompt: {harmful_prompts[0][:80]}...")
print(f"  Critique: {principles[0][0][:80]}...")
print(f"  Revision: {principles[0][1][:80]}...")


# Load progress
progress = load_progress()
start_idx = progress["completed"]

print(f"📊 Starting from {start_idx}/{len(harmful_prompts)}")
print(f"💾 All data saves to: {DRIVE_BASE}")
print(f"🔄 Auto-saves after each example")
print("\n" + "="*80 + "\n")

# Main loop
for i in range(start_idx, len(harmful_prompts)):
    prompt = harmful_prompts[i]

    try:
        print(f"\n[{i+1}/{len(harmful_prompts)}] Processing...")

        # Select random principle
        principle, revision = random.choice(principles)
        revision += " Do not provide meta-commentary; respond as the assistant would."

        # Step 1: Initial response
        print("  Generating initial response...")
        initial_response = generate_response([{"role": "user", "content": prompt}])

        # Step 2: Critique
        print("  Generating critique...")
        critique_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": initial_response},
            {"role": "user", "content": f"CritiqueRequest: {principle}\n\nCritique:"}
        ]
        critique = generate_response(critique_messages)

        # Step 3: Revision
        print("  Generating revision...")
        revision_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": initial_response},
            {"role": "user", "content": f"CritiqueRequest: {principle}"},
            {"role": "assistant", "content": critique},
            {"role": "user", "content": f"RevisionRequest: {revision}\n\nRevision:"}
        ]
        revised_response = generate_response(revision_messages)

        # Save to Drive IMMEDIATELY
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
        save_progress(i + 1)

        print(f"✓ Saved {i+1}/{len(harmful_prompts)} to Drive")

    except Exception as e:
        print(f"✗ Error on prompt {i}: {e}")
        log_error(i, prompt, e)
        continue

print("\n" + "="*80)
print("✅ Generation complete!")
print(f"📁 Data saved to: {CHECKPOINT_FILE}")
print(f"📊 Progress file: {PROGRESS_FILE}")