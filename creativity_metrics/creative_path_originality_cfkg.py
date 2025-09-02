# creativity_metrics/creative_path_originality_cfkg.py

import os
import pickle
from collections import Counter
from creativity_metrics.creative_path_originality_base import CreativePathOriginalityBase

class CreativePathOriginalityCFKG(CreativePathOriginalityBase):
    """
    Calcolo della CPO (Creative Path Originality) per il modello CFKG.
    Per CFKG non ci sono percorsi espliciti: definiamo un pattern come la sequenza dei top-K item raccomandati.
    """

    def __init__(self, dataset_name='ml1m', topk=5):
        super().__init__(dataset_name=dataset_name, model_name='cfkg')
        self.topk = topk
        self.pattern_counts = self._compute_pattern_counts()

    def _compute_pattern_counts(self):
        """Conta le occorrenze dei pattern di top-K items per tutti gli utenti."""
        patterns = Counter()
        for user_id, rec_items in self.topk_data.items():
            # Estrai solo i primi `topk` item
            topk_items = rec_items[:self.topk]
            # Trasforma in stringa per creare pattern univoco
            pattern_str = "-".join(map(str, topk_items))
            patterns[pattern_str] += 1
        return patterns

    def calculate_cpo_for_user(self, user_id):
        """
        Calcola la CPO per un singolo utente come:
        1 - (media frequenza dei pattern dell'utente / numero pattern)
        """
        if user_id not in self.topk_data:
            raise ValueError(f"Utente {user_id} non trovato nei dati CFKG")

        user_patterns = []
        topk_items = self.topk_data[user_id][:self.topk]
        pattern_str = "-".join(map(str, topk_items))
        user_patterns.append(pattern_str)

        # Frequenze globali dei pattern
        freq_sum = sum(self.pattern_counts.get(p, 0) for p in user_patterns)
        cpo_score = 1 - (freq_sum / len(user_patterns))
        return cpo_score

    def calculate_cpo_all_users(self, max_users=None):
        """Calcola CPO per tutti gli utenti o per un campione limitato."""
        user_ids = list(self.topk_data.keys())
        if max_users is not None:
            user_ids = user_ids[:max_users]

        cpo_scores = {}
        for user_id in user_ids:
            try:
                cpo_scores[user_id] = self.calculate_cpo_for_user(user_id)
            except Exception:
                cpo_scores[user_id] = 0.0
        return cpo_scores

