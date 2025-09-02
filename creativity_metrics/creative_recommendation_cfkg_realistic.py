import pickle
import os
import numpy as np

DATA_DIR = "data/ml1m/preprocessed/cfkg"
RESULTS_DIR = "results/ml1m/cfkg"

class CreativeRecommendationCFKG:
    def __init__(self, dataset_name="ml1m", k=10):
        self.dataset_name = dataset_name
        self.k = k
        self.topk_file = os.path.join(RESULTS_DIR, "item_topk.pkl")
        self.train_file = os.path.join(DATA_DIR, "train.txt")
        self._load_data()

    def _load_data(self):
        # Carico top-k
        if not os.path.exists(self.topk_file):
            raise FileNotFoundError("TopK CFKG file non trovato!")
        with open(self.topk_file, "rb") as f:
            self.topk = pickle.load(f)  # dict: user_id -> list of item_ids

        # Carico cronologia utenti
        if not os.path.exists(self.train_file):
            raise FileNotFoundError("Train file non trovato!")
        self.user_history = {}
        with open(self.train_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                uid = int(parts[0])
                items = list(map(int, parts[1:]))
                self.user_history[uid] = set(items)

    def evaluate_creativity(self, sample_users=None):
        user_ids = list(self.topk.keys())
        if sample_users:
            user_ids = user_ids[:sample_users]

        creativity_scores = {}
        for uid in user_ids:
            recs = self.topk.get(uid, [])[:self.k]
            history = self.user_history.get(uid, set())
            novel_count = sum(1 for item in recs if item not in history)
            creativity = novel_count / len(recs) if recs else 0.0
            creativity_scores[uid] = creativity
        return creativity_scores

if __name__ == "__main__":
    metric = CreativeRecommendationCFKG(dataset_name="ml1m", k=10)
    scores = metric.evaluate_creativity(sample_users=100)

    values = np.array(list(scores.values()))
    print("=== Creatività CFKG su ml1m ===")
    print(f"Utenti valutati: {len(scores)}")
    print(f"Media: {values.mean():.4f}")
    print(f"Std:   {values.std():.4f}")
    print(f"Min:   {values.min():.4f}")
    print(f"Max:   {values.max():.4f}")
    print("Dettaglio utenti:")
    for uid, val in scores.items():
        print(f"User {uid}: Creativity = {val:.4f}")

