from Data_Processing import (
    get_principles, get_redteam_prompts, response_form,
    critique_form, revision_form
)


from arc.llm_client import query_llm

import json
from pathlib import Path
import random
from datetime import datetime


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


if __name__ == "__main__":
    output_dir = Path("training_data")
    output_dir.mkdir(exist_ok=True)

    checkpoint_file = output_dir / "constitutional_training_data.jsonl"
    progress_file = output_dir / "progress.json"


    principles = get_principles()
    red_team = get_redteam_prompts()

    progress = load_progress(red_team)
    start_idx = progress["completed"]

    print(f"Starting from {start_idx}/{len(red_team)}")

    for i in range(start_idx, len(red_team)):
        prompt = red_team[i]

        try:
            print(f"\n[{i+1}/{len(red_team)}] Processing...")
            principle, revision = random.choice(principles)
            revision += " Do not provide meta-commentary; respond as the assistant would."

            response_structure = response_form(prompt)
            response_from_base = query_llm(response_structure)

            criticism_structure = critique_form(prompt, response_from_base, principle)
            critique_from_base = query_llm(criticism_structure)

            revision_structure = revision_form(
            prompt, response_from_base, principle, critique_from_base, revision)
            revised_from_base = query_llm(revision_structure) 

             
            with open(checkpoint_file, 'a') as f:
                json.dump({
                    "prompt": prompt,
                    "initial_response": response_from_base,
                    "principle_used": principle,
                    "critique": critique_from_base,
                    "revision": revised_from_base,
                    "index": i
                }, f)


                f.write('\n')
            

            save_progress(i + 1, red_team)
            print(f"✓ Saved {i+1}/{len(red_team)}")      

        except Exception as e:
            print(f"✗ Error on prompt {i}: {e}")

            with open(output_dir / "errors.log", 'a') as f:
                f.write(f"{i}: {prompt[:50]}... | {str(e)}\n")
            
            continue