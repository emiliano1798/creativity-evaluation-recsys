import pickle
import numpy as np
import os

class CreativeRecommendationCFKG:
    """
    Calcola la creatività delle raccomandazioni CFKG per ciascun utente.
    La creatività combina:
        - Novità: quanto gli item raccomandati sono nuovi rispetto alla cronologia dell'utente
        - Serendipità: quanto gli item raccomandati differiscono dai più popolari
        - Diversità: quanto gli item raccomandati sono diversi fra loro
    """
    def __init__(self, dataset_name="ml1m", k=10):
        self.dataset_name = dataset_name
        self.k = k
        self.topk_file = f"results/{dataset_name}/cfkg/item_topk.pkl"
        self.train_file = f"data/{dataset_name}/preprocessed/cfkg/train.txt"
        self.pop_file = f"results/{dataset_name}/most_pop/item_topks.pkl"
        self._load_data()

    def _load_data(self):
        # Carica TopK CFKG
        if not os.path.exists(self.topk_file):
            raise FileNotFoundError("TopK CFKG file non trovato!")
        with open(self.topk_file, "rb") as f:
            self.topk = pickle.load(f)

        # Carica cronologia utenti
        if not os.path.exists(self.train_file):
            raise FileNotFoundError("File train.txt non trovato!")
        self.user_history = {}
        with open(self.train_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                uid, pid = int(parts[0]), int(parts[1])
                self.user_history.setdefault(uid, set()).add(pid)

        # Carica TopK popolari
        if os.path.exists(self.pop_file):
            with open(self.pop_file, "rb") as f:
                self.most_pop = pickle.load(f)
        else:
            self.most_pop = {}

    def _novelty(self, user_id, recommended):
        history = self.user_history.get(user_id, set())
        new_items = [i for i in recommended if i not in history]
        return len(new_items) / len(recommended)

    def _serendipity(self, user_id, recommended):
        popular = set(self.most_pop.get(user_id, []))
        ser_items = [i for i in recommended if i not in popular]
        return len(ser_items) / len(recommended)

    def _diversity(self, recommended):
        # Diversità basata sul numero di item unici
        return len(set(recommended)) / len(recommended)

    def compute_creativity(self, sample_users=None):
        creativity_scores = {}
        users = list(self.topk.keys())
        if sample_users is not None:
            users = users[:sample_users]

        for uid in users:
            recommended = self.topk[uid][:self.k]
            nov = self._novelty(uid, recommended)
            ser = self._serendipity(uid, recommended)
            div = self._diversity(recommended)
            # Combina i tre fattori in un punteggio 0-1
            creativity_scores[uid] = (nov + ser + div) / 3.0
        return creativity_scores

