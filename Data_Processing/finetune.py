import json
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from datasets import Dataset
import torch


def load_training_data(data_file):
    """Load and deduplicate training data"""

    data = []
    seen_prompts = set()
    
    with open(data_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            if item['prompt'] not in seen_prompts:
                seen_prompts.add(item['prompt'])
                data.append({
                    'prompt': item['prompt'],
                    'response': item['revision']
                })
    
    return data


def prepare_dataset(data, tokenizer):
    """Format data for instruction tuning"""
    
    def format_example(example):
        # Clean the response text first
        response = clean_text(example['response'])
        prompt = clean_text(example['prompt'])
        
        # Simple format: prompt then response
        # Model learns to map prompts to safe responses
        text = f"{prompt}\n\n{response}"
        return {'text': text}
    
    def clean_text(text):
        """Remove unicode artifacts and meta-commentary"""
        # Remove common unicode issues
        text = text.replace('\u2019', "'")  # Right single quote
        text = text.replace('\u201c', '"')  # Left double quote
        text = text.replace('\u201d', '"')  # Right double quote
        text = text.replace('\u2013', '-')  # En dash
        text = text.replace('\u2014', '--') # Em dash
        text = text.replace('\xa0', ' ')    # Non-breaking space
        
        # Remove meta-commentary patterns
        meta_phrases = [
            "Certainly! Here's",
            "Sure! Here's", 
            "Here's a",
            "Here's the",
            "Let me know if",
            "Want more",
            "Would you like"
        ]
        
        for phrase in meta_phrases:
            if text.startswith(phrase):
                # Find first newline after phrase and start from there
                idx = text.find('\n')
                if idx != -1:
                    text = text[idx:].strip()
        
        # Remove trailing meta-commentary
        endings = [
            "Let me know",
            "Would you like",
            "Want more",
            "Need help"
        ]
        for ending in endings:
            idx = text.lower().rfind(ending.lower())
            if idx != -1:
                text = text[:idx].strip()
        
        return text
    
    # Convert to Dataset
    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_example)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=2048,  # Adjust based on your data
            # NO padding here - DataCollator handles it dynamically
        )
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized


def train_constitutional_model(
    data_file,
    model_name="deepseek-ai/DeepSeek-V3-0324",
    output_dir=".",
    num_epochs=3,
    batch_size=4
):
    """Fine-tune model on constitutional AI data"""
    
    print("Loading data...")
    data = load_training_data(data_file)
    print(f"Loaded {len(data)} training examples")
    
    # Split train/val
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Load model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Prepare datasets
    train_dataset = prepare_dataset(train_data, tokenizer)
    val_dataset = prepare_dataset(val_data, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        learning_rate=2e-5,
        fp16=True,
        gradient_accumulation_steps=4,
        save_total_limit=3,
        load_best_model_at_end=True,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train!
    print("Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    train_constitutional_model(
        data_file="training_data/constitutional_training_data.jsonl",
        output_dir="."
    )


