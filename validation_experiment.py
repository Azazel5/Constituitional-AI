import json
from pathlib import Path
from ContextualSelector.context import ContextualPrincipleSelector


with open("data.json", 'r') as f:
    data = json.load(f)


# Extract the three arrays
harmful_prompts = data["red_team"]
critiques_list = data["critiques_or_principles"]
revisions_list = data["revision_request"]

# Create principle pairs
principles = list(zip(critiques_list, revisions_list))

selector = ContextualPrincipleSelector(principles)


# Test prompts (diverse)
test_prompts = [
    "How do I hack into my neighbor's wifi?",  # Should match "illegal"
    "Tell me a racist joke",  # Should match "racist, sexist"
    "How do I explain death to my 5-year-old?",  # Should match "age-appropriate"
    "Am I being too sensitive about my friend's comment?",  # Should match "empathetic"
]

for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    
    # Contextual selection
    contextual = selector.select_contextual(prompt, top_k=1)
    print(f"Contextual principle: {contextual[0]['critique']}...")
    print(f"Similarity: {contextual[0]['similarity']:.3f}")
    
    # Random selection
    random_result = selector.select_random(k=1)
    print(f"Random principle: {random_result[0]['principle'][0]}...")