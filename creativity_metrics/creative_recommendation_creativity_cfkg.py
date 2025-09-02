# creativity_metrics/creative_recommendation_cfkg.py

import os
import pickle
import numpy as np

class CreativeRecommendationCFKG:
    def __init__(self, dataset_name='ml1m'):
        self.dataset_name = dataset_name
        self.topk_file = os.path.join("results", dataset_name, "cfkg", "item_topk.pkl")
        self.recs = self._load_recommendations()

    def _load_recommendations(self):
        if not os.path.exists(self.topk_file):
            raise FileNotFoundError("TopK CFKG file non trovato!")
        with open(self.topk_file, "rb") as f:
            recs = pickle.load(f)
        return recs

    def calculate_creativity_for_user(self, user_id):
        """Esempio semplice: creatività = numero di item unici non presenti tra i più popolari"""
        topk_items = self.recs.get(user_id, [])
        if not topk_items:
            return 0.0
        # qui puoi definire logica più avanzata
        # esempio: più l'utente riceve items rari, maggiore la creatività
        unique_items = len(set(topk_items))
        creativity_score = unique_items / len(topk_items)
        return creativity_score

    def calculate_creativity_all_users(self, max_users=None):
        users = list(self.recs.keys())
        if max_users:
            users = users[:max_users]
        scores = {uid: self.calculate_creativity_for_user(uid) for uid in users}
        return scores

    def get_summary_statistics(self, max_users=None):
        scores = list(self.calculate_creativity_all_users(max_users=max_users).values())
        if not scores:
            return {"avg_creativity": 0.0, "std_creativity": 0.0}
        return {
            "avg_creativity": np.mean(scores),
            "std_creativity": np.std(scores),
            "min_creativity": np.min(scores),
            "max_creativity": np.max(scores)
        }

