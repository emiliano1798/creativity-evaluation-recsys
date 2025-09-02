# creativity_metrics/creative_path_originality_base.py

import os
import pickle
import numpy as np
from collections import Counter

class CreativePathOriginalityBase:
    def __init__(self, dataset_name='ml1m', model_name='pgpr'):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.results_path = f'results/{dataset_name}/{model_name}'
        
        print(f"Loading {model_name.upper()} data from {self.results_path}...")
        
        # Identifica tipo di modello e carica dati appropriati
        self.model_type = self._identify_model_type()
        self._load_model_data()
        
        print(f"Loaded data for {len(self.topk_data)} users")
        
        # Estrai pattern appropriati per il tipo di modello
        self.patterns = self._extract_patterns()
        print(f"Found {len(self.patterns)} unique patterns")

    def _identify_model_type(self):
        """Identifica il tipo di modello basandosi sui file disponibili"""
        # Controlla quali file esistono
        has_pred_paths = os.path.exists(os.path.join(self.results_path, 'pred_paths.pkl'))
        has_path_topk = os.path.exists(os.path.join(self.results_path, 'path_topk.pkl'))
        has_item_topk = os.path.exists(os.path.join(self.results_path, 'item_topk.pkl'))
        
        if has_pred_paths and has_path_topk:
            return 'path_based'  # PGPR, CAFE, UCPR
        elif has_item_topk:
            return 'knowledge_aware'  # CFKG, KGAT, CKE
        else:
            return 'embedding'  # TransE

    def _load_model_data(self):
        """Carica dati specifici per tipo di modello"""
        if self.model_type == 'path_based':
            self._load_path_based_data()
        elif self.model_type == 'knowledge_aware':
            self._load_knowledge_aware_data()
        elif self.model_type == 'embedding':
            self._load_embedding_data()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _load_path_based_data(self):
        """Carica dati per modelli path-based (PGPR, CAFE, UCPR)"""
        self.topk_data = self._load_pickle_file('path_topk.pkl')
        self.paths_data = self._load_pickle_file('pred_paths.pkl')

    def _load_knowledge_aware_data(self):
        """Carica dati per modelli knowledge-aware (CFKG, KGAT, CKE)"""
        self.topk_data = self._load_pickle_file('item_topk.pkl')
        self.paths_data = None  # Non hanno path espliciti

    def _load_embedding_data(self):
        """Carica dati per modelli embedding (TransE)"""
        # TransE potrebbe non avere file topk, gestiamo il caso
        topk_files = ['item_topk.pkl', 'topk.pkl', 'recommendations.pkl']
        self.topk_data = None
        
        for filename in topk_files:
            try:
                self.topk_data = self._load_pickle_file(filename)
                print(f"Found topk data in {filename}")
                break
            except FileNotFoundError:
                continue
        
        if self.topk_data is None:
            # Se non trova file topk, crea dati vuoti per ora
            print("Warning: No topk file found for TransE, using empty data")
            self.topk_data = {}
        
        self.paths_data = None  # Non hanno path espliciti

    def _load_pickle_file(self, filename):
        """Carica file pickle con gestione errori"""
        file_path = os.path.join(self.results_path, filename)
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def _extract_patterns(self):
        """Estrai pattern appropriati per il tipo di modello"""
        if self.model_type == 'path_based':
            return self._extract_path_patterns()
        else:
            return self._extract_recommendation_patterns()

    def _extract_path_patterns(self):
        """Estrai pattern dai path espliciti (per modelli path-based)"""
        all_patterns = []
        for user_paths_dict in self.paths_data.values():
            for path_info_list in user_paths_dict.values():
                for path_info in path_info_list:
                    if len(path_info) >= 3:
                        pattern = self._get_path_pattern(path_info[2])
                        all_patterns.append(pattern)
        
        pattern_counts = Counter(all_patterns)
        total_patterns = len(all_patterns)
        return {p: c / total_patterns for p, c in pattern_counts.items()}

    def _extract_recommendation_patterns(self):
        """Estrai pattern dalle raccomandazioni (per modelli non path-based)"""
        all_patterns = []
        
        for user_id, recommendations in self.topk_data.items():
            # Converti recommendations in lista se necessario
            if isinstance(recommendations, dict):
                rec_list = list(recommendations.values())
            else:
                rec_list = recommendations
            
            # Analizza pattern di diversità
            pattern = self._get_recommendation_pattern(rec_list)
            all_patterns.append(pattern)
        
        pattern_counts = Counter(all_patterns)
        total_patterns = len(all_patterns)
        return {p: c / total_patterns for p, c in pattern_counts.items()}

    def _get_path_pattern(self, path_sequence):
        """Estrai pattern dal path (per modelli path-based)"""
        if len(path_sequence) <= 1:
            return "direct"
        
        relations = [step[0] for step in path_sequence if step[0] != 'self_loop']
        
        if len(relations) == 0:
            return "self_loop_only"
        elif len(relations) == 1:
            return f"single_{relations[0]}"
        else:
            return f"multi_{len(relations)}_{relations[0]}_{relations[-1]}"

    def _get_recommendation_pattern(self, recommendations):
        """Estrai pattern dalle raccomandazioni (per modelli non path-based)"""
        if not recommendations:
            return "empty"
        
        # Analizza diversità
        unique_items = len(set(recommendations))
        total_items = len(recommendations)
        
        if unique_items == 1:
            return "concentrated"
        elif unique_items == total_items:
            return "fully_diverse"
        elif unique_items / total_items > 0.8:
            return "highly_diverse"
        elif unique_items / total_items > 0.5:
            return "moderately_diverse"
        else:
            return "low_diverse"

    def _get_path_length(self, path_sequence):
        """Calcola lunghezza path (solo per modelli path-based)"""
        return len([step for step in path_sequence if step[0] != 'self_loop'])

    def calculate_cpo_for_user(self, user_id):
        """Calcola CPO per un utente - metodo unificato"""
        if self.model_type == 'path_based':
            return self._calculate_cpo_path_based(user_id)
        else:
            return self._calculate_cpo_non_path_based(user_id)

    def _calculate_cpo_path_based(self, user_id):
        """CPO per modelli path-based"""
        if user_id not in self.paths_data:
            return 0.0
        
        cpo_scores = []
        for path_info_list in self.paths_data[user_id].values():
            for path_info in path_info_list:
                if len(path_info) >= 3:
                    path_seq = path_info[2]
                    unexpectedness = self._calculate_unexpectedness_path(path_seq)
                    coherence = self._calculate_coherence_path(path_seq)
                    relevance = self._calculate_relevance_path(path_info)
                    cpo_scores.append(unexpectedness * coherence * relevance)
        
        return np.mean(cpo_scores) if cpo_scores else 0.0

    def _calculate_cpo_non_path_based(self, user_id):
        """CPO per modelli non path-based"""
        if user_id not in self.topk_data:
            return 0.0
        
        recommendations = self.topk_data[user_id]
        if isinstance(recommendations, dict):
            rec_list = list(recommendations.values())
        else:
            rec_list = recommendations
        
        # Limita a top-10
        rec_list = rec_list[:10]
        
        unexpectedness = self._calculate_unexpectedness_recommendations(rec_list)
        coherence = self._calculate_coherence_recommendations(rec_list)
        relevance = self._calculate_relevance_recommendations(rec_list)
        
        return unexpectedness * coherence * relevance

    # Metodi per calcolo componenti CPO - PATH BASED
    def _calculate_unexpectedness_path(self, path_sequence):
        """Unexpectedness per path"""
        pattern = self._get_path_pattern(path_sequence)
        expected_prob = self.patterns.get(pattern, 0.001)
        return max(min(1 - expected_prob, 0.98), 0.02)

    def _calculate_coherence_path(self, path_sequence):
        """Coherence per path"""
        length = self._get_path_length(path_sequence)
        if length == 0: return 0.1
        elif length == 1: return 0.6
        elif length == 2: return 1.0
        elif length == 3: return 0.8
        else: return max(0.3, 1.0 - (length - 3) * 0.2)

    def _calculate_relevance_path(self, path_info):
        """Relevance per path"""
        if len(path_info) >= 2:
            score, prob = path_info[0], path_info[1]
            normalized_score = min(max(score, 0), 1)
            return min(0.7 * normalized_score + 0.3 * min(prob * 100000, 1), 1.0)
        return 0.5

    # Metodi per calcolo componenti CPO - NON PATH BASED
    def _calculate_unexpectedness_recommendations(self, recommendations):
        """Unexpectedness per raccomandazioni"""
        pattern = self._get_recommendation_pattern(recommendations)
        expected_prob = self.patterns.get(pattern, 0.001)
        return max(min(1 - expected_prob, 0.98), 0.02)

    def _calculate_coherence_recommendations(self, recommendations):
        """Coherence per raccomandazioni"""
        if not recommendations:
            return 0.1
        
        # Coerenza basata su diversità ottimale
        unique_ratio = len(set(recommendations)) / len(recommendations)
        
        if 0.5 <= unique_ratio <= 0.8:
            return 1.0  # Diversità ottimale
        elif 0.3 <= unique_ratio < 0.5 or 0.8 < unique_ratio <= 1.0:
            return 0.8  # Buona diversità
        else:
            return 0.5  # Diversità subottimale

    def _calculate_relevance_recommendations(self, recommendations):
        """Relevance per raccomandazioni"""
        if not recommendations:
            return 0.5
        
        # Assumiamo che l'ordine nella lista indichi rilevanza
        relevance_scores = []
        for i, item in enumerate(recommendations):
            position_score = 1 - (i / len(recommendations))
            relevance_scores.append(position_score)
        
        return np.mean(relevance_scores)

    # Metodi comuni
    def calculate_cpo_all_users(self, max_users=None):
        """Calcola CPO per tutti gli utenti"""
        users = list(self.topk_data.keys())
        if max_users:
            users = users[:max_users]
        
        print(f"Calculating CPO for {len(users)} users...")
        return {user_id: self.calculate_cpo_for_user(user_id) for user_id in users}

    def get_summary_statistics(self, max_users=1000):
        """Statistiche riassuntive"""
        print("Computing summary statistics...")
        cpo_scores = self.calculate_cpo_all_users(max_users)
        
        if not cpo_scores:
            return {
                'total_users_analyzed': 0,
                'avg_cpo': 0,
                'std_cpo': 0,
                'min_cpo': 0,
                'max_cpo': 0,
                'median_cpo': 0,
                'total_patterns': len(self.patterns),
                'most_common_patterns': {},
                'least_common_patterns': {}
            }
        
        sorted_patterns = sorted(self.patterns.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'model_type': self.model_type,
            'total_users_analyzed': len(cpo_scores),
            'avg_cpo': np.mean(list(cpo_scores.values())),
            'std_cpo': np.std(list(cpo_scores.values())),
            'min_cpo': min(cpo_scores.values()),
            'max_cpo': max(cpo_scores.values()),
            'median_cpo': np.median(list(cpo_scores.values())),
            'total_patterns': len(self.patterns),
            'most_common_patterns': dict(sorted_patterns[:10]),
            'least_common_patterns': dict(sorted_patterns[-5:])
        }

    def analyze_creativity_distribution(self, max_users=1000):
        """Analizza distribuzione creatività"""
        cpo_scores = self.calculate_cpo_all_users(max_users)
        
        if not cpo_scores:
            return {
                'percentiles': {},
                'most_creative_users': [],
                'least_creative_users': []
            }
        
        scores_list = list(cpo_scores.values())
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        percentile_values = np.percentile(scores_list, percentiles)
        sorted_users = sorted(cpo_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'percentiles': dict(zip(percentiles, percentile_values)),
            'most_creative_users': sorted_users[:10],
            'least_creative_users': sorted_users[-10:]
        }
