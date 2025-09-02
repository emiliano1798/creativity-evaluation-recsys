import os
import pickle
import numpy as np

class CreativeRecommendationCFKG:
    """
    Calcola la creatività delle raccomandazioni CFKG.
    Combina novità (vs top popolari) e unexpectedness (vs storico utente).
    """
    def __init__(self, dataset_name="ml1m", alpha=0.5, k=10):
        self.dataset_name = dataset_name
        self.alpha = alpha  # peso tra novelty e unexpectedness
        self.k = k
        self.topk_file = f"results/{dataset_name}/cfkg/item_topk.pkl"
        self.most_pop_file = f"results/{dataset_name}/most_pop/item_topks.pkl"
        self._load_recommendations()
        self._load_most_pop()

    def _load_recommendations(self):
        if not os.path.exists(self.topk_file):
            raise FileNotFoundError("TopK CFKG file non trovato!")
        with open(self.topk_file, "rb") as f:
            self.recs = pickle.load(f)

    def _load_most_pop(self):
        if not os.path.exists(self.most_pop_file):
            raise FileNotFoundError("TopK most popular file non trovato!")
        with open(self.most_pop_file, "rb") as f:
            self.most_pop = pickle.load(f)

    def _novelty_score(self, topk_items, most_pop):
        """Quanto le raccomandazioni si discostano dagli item più popolari."""
        topk_set = set(topk_items[:self.k])
        pop_set = set(most_pop[:self.k])
        novel_items = topk_set - pop_set
        return len(novel_items) / self.k

    def _unexpectedness_score(self, topk_items, user_history):
        """Quanto le raccomandazioni si discostano dallo storico dell'utente."""
        hist_set = set(user_history)
        topk_set = set(topk_items[:self.k])
        unexpected_items = topk_set - hist_set
        return len(unexpected_items) / self.k

    def compute_user_creativity(self, user_id, user_history=None):
        topk_items = self.recs.get(user_id, [])
        most_pop = self.most_pop.get(user_id, [])
        if user_history is None:
            user_history = []  # se non c'è storico, unexpectedness = 1
        novelty = self._novelty_score(topk_items, most_pop)
        unexpectedness = self._unexpectedness_score(topk_items, user_history)
        creativity = self.alpha * novelty + (1 - self.alpha) * unexpectedness
        return creativity

    def compute_all_users(self, user_histories=None):
        """
        Calcola la creatività per tutti gli utenti.
        user_histories: dict {user_id: list of historical items}
        """
        if user_histories is None:
            user_histories = {}
        creativity_scores = {}
        for uid in self.recs.keys():
            hist = user_histories.get(uid, [])
            creativity_scores[uid] = self.compute_user_creativity(uid, hist)
        return creativity_scores

