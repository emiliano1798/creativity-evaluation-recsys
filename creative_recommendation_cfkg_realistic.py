import pickle
import os

class CreativeRecommendationCFKG:
    def __init__(self, dataset_name="ml1m", k=10):
        self.dataset_name = dataset_name
        self.k = k
        self.topk_file = f"results/{dataset_name}/cfkg/item_topk.pkl"
        self.train_file = f"data/{dataset_name}/preprocessed/cfkg/train.txt"
        self.recommendations = {}
        self.user_history = {}
        self._load_data()

    def _load_data(self):
        # Carica top-K CFKG
        if not os.path.exists(self.topk_file):
            raise FileNotFoundError(f"TopK CFKG file non trovato: {self.topk_file}")
        with open(self.topk_file, "rb") as f:
            self.recommendations = pickle.load(f)

        # Carica cronologia utenti
        if not os.path.exists(self.train_file):
            raise FileNotFoundError(f"File train non trovato: {self.train_file}")
        with open(self.train_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                uid = int(parts[0])
                items = set(map(int, parts[1:]))
                self.user_history[uid] = items

    def compute_user_creativity(self, user_id):
        recs = self.recommendations.get(user_id, [])[:self.k]
        history = self.user_history.get(user_id, set())
        if not recs:
            return 0.0
        new_items = sum(1 for item in recs if item not in history)
        return new_items / len(recs)

    def evaluate_all_users(self, sample_users=None):
        users = list(self.recommendations.keys())
        if sample_users:
            users = users[:sample_users]
        results = {}
        for uid in users:
            results[uid] = self.compute_user_creativity(uid)
        return results

