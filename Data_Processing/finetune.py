#!/usr/bin/env python3
"""
Fine-tune models on Constitutional AI data
Usage: python finetune.py --model qwen --selection random --data-file CAI/data/constitutional_training_data.jsonl
"""

import argparse
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from pathlib import Path
import numpy as np

# ============================================
# Configuration
# ============================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['qwen', 'mistral'],
                        help='Base model to fine-tune')
    parser.add_argument('--selection', type=str, required=True,
                        choices=['random', 'contextual'],
                        help='Which dataset (random or contextual selection)')
    parser.add_argument('--data-file', type=str, required=True,
                        help='Path to training data JSONL')
    parser.add_argument('--output-dir', type=str, default='models/finetuned',
                        help='Where to save fine-tuned model')
    parser.add_argument('--val-size', type=int, default=79,
                        help='Number of examples for validation')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Training batch size')
    return parser.parse_args()

args = parse_args()
SEED = 42

# Model mapping
MODEL_NAMES = {
    'qwen': 'Qwen/Qwen2.5-3B-Instruct',
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.3'
}

base_model_name = MODEL_NAMES[args.model]
SPLIT_DIR =  Path(args.output_dir)
print(f"Output directory is: {SPLIT_DIR}")
output_dir = SPLIT_DIR / f"{args.model}_{args.selection}"
output_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print(f"Constitutional AI Fine-Tuning")
print("="*80)
print(f"Base model: {base_model_name}")
print(f"Selection method: {args.selection}")
print(f"Data file: {args.data_file}")
print(f"Output directory: {output_dir}")
print(f"Validation size: {args.val_size}")
print()

# ============================================
# Load Data
# ============================================

print("Loading training data...")
examples = []
with open(args.data_file, 'r') as f:
    for line in f:
        examples.append(json.loads(line))

print(f"Loaded {len(examples)} examples")

# Split train/val
split_file = SPLIT_DIR / 'train_val_split.json'

# Use existing split (ensures consistency across all 4 models)
print(f"Loading existing split from {split_file}...")
with open(split_file, 'r') as f:
    split_info = json.load(f)
    
    train_indices = split_info['train_indices']
    val_indices = split_info['val_indices']
    print(f"✓ Using seed {split_info['seed']}")
    

# Apply split
train_examples = [examples[i] for i in train_indices]
val_examples = [examples[i] for i in val_indices]

print(f"Train: {len(train_examples)} examples")
print(f"Validation: {len(val_examples)} examples")


val_prompts_expected = set(split_info['val_prompts'])
val_prompts_actual = set(ex['prompt'] for ex in val_examples)
if val_prompts_expected == val_prompts_actual:
    print("✓ Validation prompts verified consistent across datasets")
else:
    print("⚠️  WARNING: Validation prompt mismatch detected!")


print()

# ============================================
# Prepare Training Data
# ============================================

print("Preparing training data...")

def format_example(example):
    """
    Format: prompt -> revised_response
    We train the model to directly produce the safe, revised response
    """
    prompt = example['prompt']
    revision = example['revision']
    
    # Format as instruction-following
    text = f"Human: {prompt}\n\nAssistant: {revision}"
    return text

train_texts = [format_example(ex) for ex in train_examples]
val_texts = [format_example(ex) for ex in val_examples]

# Create HuggingFace datasets
train_dataset = Dataset.from_dict({'text': train_texts})
val_dataset = Dataset.from_dict({'text': val_texts})

print(f"✓ Training dataset: {len(train_dataset)} examples")
print(f"✓ Validation dataset: {len(val_dataset)} examples")
print()


# ============================================
# Load Model & Tokenizer
# ============================================

print(f"Loading {base_model_name}...")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available! This script requires GPU.")
    
    
print(f"✓ CUDA available")
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print()


tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    dtype=torch.float16,
)

model = model.to("cuda:0")

print(f"✓ Model loaded: {model.device}")
print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
print()

# ============================================
# Tokenize Data
# ============================================

print("Tokenizing data...")

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=512,
        padding='max_length'
    )

tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['text']
)

tokenized_val = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['text']
)

print("✓ Tokenization complete")
print()

# ============================================
# Training Configuration
# ============================================

training_args = TrainingArguments(
    output_dir=str(output_dir),
    num_train_epochs=args.epochs,
   per_device_train_batch_size=1,  
    per_device_eval_batch_size=1,  
    learning_rate=2e-5,
    warmup_steps=100,
    logging_steps=50,
    eval_steps=200,
    save_steps=200,
    eval_strategy='steps',
    save_strategy='steps',
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    bf16=True,  # Use mixed precision
    gradient_accumulation_steps=16,  # Increase from 2 to 16 (keeps effective batch size = 16)
    gradient_checkpointing=True,  # Add this - trades speed for memory
    report_to='none'  # Disable wandb
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM, not masked LM
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator
)

# ============================================
# Train
# ============================================

print("="*80)
print("Starting Training")
print("="*80)
print()

trainer.train()

print()
print("="*80)
print("Training Complete!")
print("="*80)
print()

# ============================================
# Save Model
# ============================================

print(f"Saving model to {output_dir}...")
trainer.save_model(str(output_dir))
tokenizer.save_pretrained(str(output_dir))

# Save training info
info = {
    'base_model': base_model_name,
    'selection_method': args.selection,
    'train_examples': len(train_examples),
    'val_examples': len(val_examples),
    'epochs': args.epochs,
    'learning_rate': 2e-5,
    'final_train_loss': trainer.state.log_history[-2]['loss'],
    'final_eval_loss': trainer.state.log_history[-1]['eval_loss']
}

with open(output_dir / 'training_info.json', 'w') as f:
    json.dump(info, f, indent=2)

print(f"✓ Model saved to {output_dir}")
print(f"✓ Training info saved")
print()
print("="*80)
print("Fine-tuning Complete!")
print("="*80)