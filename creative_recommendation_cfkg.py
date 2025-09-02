# creativity_metrics/creative_recommendation_cfkg.py
import os
import pickle
import numpy as np

class CreativeRecommendationCFKG:
    def __init__(self, dataset_name='ml1m', alpha=0.5):
        self.dataset_name = dataset_name
        self.alpha = alpha
        self.recs = self._load_recommendations()
        self.user_history = self._load_user_history()
        self.item_embeddings = self._load_item_embeddings()

    def _load_recommendations(self):
        path = f"results/{self.dataset_name}/cfkg/item_topk.pkl"
        if not os.path.exists(path):
            raise FileNotFoundError("TopK CFKG file non trovato!")
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_user_history(self):
        path = f"data/{self.dataset_name}/train.txt"
        user_history = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    user = int(parts[0])
                    items = list(map(int, parts[1:]))
                    user_history[user] = set(items)
        return user_history

    def _load_item_embeddings(self):
        # TODO: qui mettiamo caricamento reale degli embedding del modello CFKG
        # Per ora: generiamo embedding random per ogni item
        all_items = {i for items in self.recs.values() for i in items}
        return {item: np.random.rand(16) for item in all_items}  # 16-dim casuali

    def novelty(self, user, recs):
        seen = self.user_history.get(user, set())
        new_items = [i for i in recs if i not in seen]
        return len(new_items) / len(recs) if recs else 0

    def diversity(self, recs):
        if len(recs) < 2:
            return 0
        vecs = [self.item_embeddings[i] for i in recs if i in self.item_embeddings]
        if len(vecs) < 2:
            return 0
        dists = []
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                d = np.linalg.norm(vecs[i] - vecs[j])
                dists.append(d)
        return np.mean(dists)

    def creativity(self, user):
        recs = self.recs.get(user, [])
        if not recs:
            return 0
        n = self.novelty(user, recs)
        d = self.diversity(recs)
        return self.alpha * n + (1 - self.alpha) * d

