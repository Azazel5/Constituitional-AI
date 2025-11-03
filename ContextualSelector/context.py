import random
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class ContextualPrincipleSelector:
    def __init__(self, principles):
        """
        principles: list of (critique, revision) tuples
        """

        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.principles = principles
        
        self.critiques = [p[0] for p in principles]
        self.critique_embeddings = self.model.encode(self.critiques)
    

    def select_contextual(self, prompt, top_k=1):
        """Select most relevant principle for this prompt"""

        prompt_embedding = self.model.encode([prompt])
        
        similarities = cosine_similarity(
            prompt_embedding,
            self.critique_embeddings
        )[0]
        
        # Get top-k most similar

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []

        for idx in top_indices:
            results.append({
                'principle': self.principles[idx],
                'critique': self.critiques[idx],
                'similarity': float(similarities[idx])
            })

        return results
    
    def select_random(self, k=1):
        """Baseline: random selection"""

        selected = random.sample(self.principles, k)
        return [{'principle': p, 'similarity': None} for p in selected]