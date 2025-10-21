import json
from pathlib import Path
from pdb import set_trace


current_script_path = Path(__file__).resolve()
PARENT_DIR = current_script_path.parent.parent


def get_redteam_prompts():
    prompts = []
    EVALS_DIR = PARENT_DIR / "Data/evals"

    files_list = [p for p in EVALS_DIR.iterdir() if p.is_file()]

    for file_path in files_list:
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)

                # Extract just the human's harmful question
                prompt = data['prompt'].split('\n\nHuman: ')[1].split('\n\n')[0]
                prompts.append(prompt)


    return prompts


def get_principles():
    principles = []
    EVALS_DIR = PARENT_DIR / "Data/prompts"
    principles_file = "CritiqueRevisionInstructions.json"

    
    with open(EVALS_DIR / principles_file, 'r') as f:
        data = json.load(f)
        
        for _, v in data.items():
            critique_raw = v['prompt'][0]  
            revision_raw = v['edit_request'] 

            critique = critique_raw.replace('\n\nCritiqueRequest:', '').replace('\n\nCritique:', '').strip()
            revision = revision_raw.replace('\n\nRevisionRequest:', '').replace('\n\nRevision:', '').strip()
    

            principles.append((critique, revision))
            

    return principles


print(get_principles())