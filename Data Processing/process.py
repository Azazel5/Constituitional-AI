import json
from pathlib import Path
from pdb import set_trace


current_script_path = Path(__file__).resolve()

PARENT_DIR = current_script_path.parent.parent
EVALS_DIR = PARENT_DIR / "Data/evals"


file_name = "438HHHEvaluations.jsonl"  
file_path = EVALS_DIR / file_name


prompts = []
with open(file_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        # Extract just the human's harmful question
        prompt = data['prompt'].split('\n\nHuman: ')[1].split('\n\n')[0]
        prompts.append(prompt)


set_trace()
print(prompts)