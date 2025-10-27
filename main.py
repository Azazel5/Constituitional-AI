from Data_Processing import (
    get_principles, get_redteam_prompts, response_form,
    critique_form, revision_form
)

from pdb import set_trace as breakpoint


from arc.llm_client import query_llm, query_llm_openrouter

import json
from pathlib import Path
import random
from datetime import datetime
import time


def load_progress(harmful_prompts):
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {"completed": 0, "total": len(harmful_prompts), "timestamp": str(datetime.now())}


def save_progress(completed, harmful_prompts):
    with open(progress_file, 'w') as f:
        json.dump({
            "completed": completed,
            "total": len(harmful_prompts),
            "timestamp": str(datetime.now())
        }, f)

def query_with_backoff(query_structure, use_openrouter=False, max_retries=5):
    """
    Query LLM with exponential backoff on rate limits.
    Alternates between HuggingFace and OpenRouter on failures.
    """

    for attempt in range(max_retries):
        try:
            if use_openrouter:
                response = query_llm_openrouter(query_structure)
            else:
                response = query_llm(query_structure)
            
            time.sleep(2)
            return response
            
        except Exception as e:
            error_str = str(e).lower()
            
            if "rate" in error_str or "429" in error_str or "limit" in error_str:
                wait_time = min(30 * (2 ** attempt), 600)
                print(f"Rate limited on {'OpenRouter' if use_openrouter else 'HuggingFace'}. "
                      f"Waiting {wait_time}s (attempt {attempt+1}/{max_retries})...")
                
                time.sleep(wait_time)
                
                if attempt >= 2:
                    print(f"Switching to {'HuggingFace' if use_openrouter else 'OpenRouter'}...")
                    use_openrouter = not use_openrouter
            else:
                raise e
    
    
    raise Exception(f"Max retries ({max_retries}) exceeded for query")


if __name__ == "__main__":
    output_dir = Path("training_data")
    output_dir.mkdir(exist_ok=True)

    checkpoint_file = output_dir / "constitutional_training_data.jsonl"
    progress_file = output_dir / "progress.json"


    critiques, revisions = get_principles()
    red_team = get_redteam_prompts()

    breakpoint()


    # progress = load_progress(red_team)
    # start_idx = progress["completed"]

    # print(f"Starting from {start_idx}/{len(red_team)}")

    # use_openrouter = True
    # consecutive_failures = 0
    
    # BATCH_SIZE = 50
    # BATCH_COOLING_TIME = 120  

    # for i in range(start_idx, len(red_team)):
    #     prompt = red_team[i]

    #     try:
    #         print(f"\n[{i+1}/{len(red_team)}] Processing...")
    #         principle, revision = random.choice(principles)
    #         revision += " Do not provide meta-commentary; respond as the assistant would."

    #         response_structure = response_form(prompt)
    #         response_from_base = query_with_backoff(response_structure, use_openrouter)

    #         criticism_structure = critique_form(prompt, response_from_base, principle)
    #         critique_from_base = query_with_backoff(criticism_structure, use_openrouter)

    #         revision_structure = revision_form(
    #         prompt, response_from_base, principle, critique_from_base, revision)
    #         revised_from_base = query_with_backoff(revision_structure, use_openrouter) 

             
    #         with open(checkpoint_file, 'a') as f:
    #             json.dump({
    #                 "prompt": prompt,
    #                 "initial_response": response_from_base,
    #                 "principle_used": principle,
    #                 "critique": critique_from_base,
    #                 "revision": revised_from_base,
    #                 "index": i
    #             }, f)


    #             f.write('\n')
            

    #         save_progress(i + 1, red_team)
    #         print(f"✓ Saved {i+1}/{len(red_team)}")      

    #         consecutive_failures = 0
            

    #         if (i + 1) % BATCH_SIZE == 0 and (i + 1) < len(red_team):
    #             print(f"\nBatch complete ({i+1}/{len(red_team)}). "
    #                   f"Cooling down for {BATCH_COOLING_TIME}s...\n")
                
    #             time.sleep(BATCH_COOLING_TIME)


    #     except Exception as e:
    #         print(f"✗ Error on prompt {i}: {e}")

    #         with open(output_dir / "errors.log", 'a') as f:
    #             f.write(f"{i}: {prompt[:50]}... | {str(e)}\n")
            
            
    #         consecutive_failures += 1
            
    #         if consecutive_failures >= 3:
    #             print(f"{consecutive_failures} consecutive failures. "
    #                   f"Switching provider and waiting 5 minutes...")
                
    #             use_openrouter = not use_openrouter
    #             consecutive_failures = 0
    #             time.sleep(300)
            
    #         continue