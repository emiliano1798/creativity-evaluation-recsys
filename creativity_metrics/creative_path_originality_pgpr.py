# creativity_metrics/creative_path_originality_pgpr.py

import pickle
import numpy as np
from collections import defaultdict, Counter
import os

class CreativePathOriginalityPGPR:
    """CPO specifico per PGPR con la struttura dati reale"""
    
    def __init__(self, dataset_name='ml1m'):
        self.dataset_name = dataset_name
        self.results_path = f'results/{dataset_name}/pgpr'
        
        print(f"Loading PGPR data from {self.results_path}...")
        
        # Carica dati PGPR
        self.topk_data = self._load_topk_data()
        self.paths_data = self._load_paths_data()
        
        print(f"Loaded data for {len(self.topk_data)} users")
        
        # Analizza pattern
        print("Extracting path patterns...")
        self.path_patterns = self._extract_path_patterns()
        
        print(f"Found {len(self.path_patterns)} unique path patterns")
        
    def _load_topk_data(self):
        """Carica raccomandazioni TopK"""
        topk_file = os.path.join(self.results_path, 'path_topk.pkl')
        with open(topk_file, 'rb') as f:
            return pickle.load(f)
    
    def _load_paths_data(self):
        """Carica path predetti"""
        paths_file = os.path.join(self.results_path, 'pred_paths.pkl')
        with open(paths_file, 'rb') as f:
            return pickle.load(f)
    
    def _extract_path_patterns(self):
        """Estrai pattern comuni dai path di tutti gli utenti"""
        all_patterns = []
        
        for user_id, user_paths_dict in self.paths_data.items():
            for item_id, path_info_list in user_paths_dict.items():
                # path_info_list = [(score, prob, path_sequence)]
                for path_info in path_info_list:
                    if len(path_info) >= 3:
                        path_sequence = path_info[2]  # Terzo elemento è la sequenza
                        pattern = self._get_path_pattern(path_sequence)
                        all_patterns.append(pattern)
        
        # Calcola frequenze pattern
        pattern_counts = Counter(all_patterns)
        total_paths = len(all_patterns)
        
        # Converti in probabilità
        pattern_probs = {
            pattern: count / total_paths 
            for pattern, count in pattern_counts.items()
        }
        
        return pattern_probs
    
    def _get_path_pattern(self, path_sequence):
        """Estrai pattern del path dalla sequenza di relation-entity"""
        if len(path_sequence) <= 1:
            return "direct"
        
        # Estrai sequenza di relazioni
        relations = [step[0] for step in path_sequence if step[0] != 'self_loop']
        
        # Crea pattern basato su relazioni e lunghezza
        if len(relations) == 0:
            return "self_loop_only"
        elif len(relations) == 1:
            return f"single_{relations[0]}"
        else:
            return f"multi_{len(relations)}_{relations[0]}_{relations[-1]}"
    
    def _get_path_length(self, path_sequence):
        """Calcola lunghezza significativa del path (esclusi self_loop)"""
        return len([step for step in path_sequence if step[0] != 'self_loop'])
    
    def _calculate_unexpectedness(self, path_sequence):
        """Calcola quanto è inaspettato il path"""
        pattern = self._get_path_pattern(path_sequence)
        expected_prob = self.path_patterns.get(pattern, 0.001)  # Small default
        
        # Unexpectedness = 1 - P(pattern)
        # Applica smoothing per evitare valori estremi
        unexpectedness = min(1 - expected_prob, 0.98)
        
        return max(unexpectedness, 0.02)  # Minimo 2%
    
    def _calculate_coherence(self, path_sequence):
        """Calcola coerenza semantica del path"""
        path_length = self._get_path_length(path_sequence)
        
        # Penalizza path troppo lunghi o troppo corti
        if path_length == 0:
            return 0.1  # Solo self-loop
        elif path_length == 1:
            return 0.6  # Path diretto
        elif path_length == 2:
            return 1.0  # Lunghezza ottimale
        elif path_length == 3:
            return 0.8  # Ancora buono
        else:
            return max(0.3, 1.0 - (path_length - 3) * 0.2)  # Penalità per lunghi
    
    def _calculate_relevance(self, path_info):
        """Calcola rilevanza basata su score e probabilità del path"""
        if len(path_info) >= 2:
            score = path_info[0]  # Score del path
            prob = path_info[1]   # Probabilità del path
            
            # Combina score e probabilità (normalizzati)
            # Score alto = più rilevante, Prob alta = più sicuro
            normalized_score = min(max(score, 0), 1)  # Clamp tra 0 e 1
            
            # Weighted combination
            relevance = 0.7 * normalized_score + 0.3 * min(prob * 100000, 1)
            return min(relevance, 1.0)
        else:
            return 0.5  # Default
    
    def calculate_cpo_for_user(self, user_id):
        """Calcola CPO per un utente specifico"""
        if user_id not in self.paths_data:
            return 0.0
        
        user_paths_dict = self.paths_data[user_id]
        cpo_scores = []
        
        for item_id, path_info_list in user_paths_dict.items():
            for path_info in path_info_list:
                if len(path_info) >= 3:
                    path_sequence = path_info[2]
                    
                    # Calcola componenti CPO
                    unexpectedness = self._calculate_unexpectedness(path_sequence)
                    coherence = self._calculate_coherence(path_sequence)
                    relevance = self._calculate_relevance(path_info)
                    
                    # CPO = unexpectedness * coherence * relevance
                    cpo_score = unexpectedness * coherence * relevance
                    cpo_scores.append(cpo_score)
        
        return np.mean(cpo_scores) if cpo_scores else 0.0
    
    def calculate_cpo_all_users(self, max_users=None):
        """Calcola CPO per tutti gli utenti (o un subset)"""
        results = {}
        users = list(self.paths_data.keys())
        
        if max_users:
            users = users[:max_users]
        
        print(f"Calculating CPO for {len(users)} users...")
        
        for i, user_id in enumerate(users):
            if i % 500 == 0:
                print(f"  Progress: {i}/{len(users)} users")
            
            results[user_id] = self.calculate_cpo_for_user(user_id)
        
        return results
    
    def get_summary_statistics(self, max_users=1000):
        """Restituisce statistiche sui pattern e creatività"""
        print("Computing summary statistics...")
        
        cpo_scores = self.calculate_cpo_all_users(max_users)
        
        # Pattern statistics
        sorted_patterns = sorted(self.path_patterns.items(), 
                               key=lambda x: x[1], reverse=True)
        
        return {
            'total_users_analyzed': len(cpo_scores),
            'avg_cpo': np.mean(list(cpo_scores.values())),
            'std_cpo': np.std(list(cpo_scores.values())),
            'min_cpo': min(cpo_scores.values()),
            'max_cpo': max(cpo_scores.values()),
            'median_cpo': np.median(list(cpo_scores.values())),
            'total_patterns': len(self.path_patterns),
            'most_common_patterns': dict(sorted_patterns[:10]),
            'least_common_patterns': dict(sorted_patterns[-5:])
        }
    
    def analyze_creativity_distribution(self, max_users=1000):
        """Analizza distribuzione creatività"""
        cpo_scores = self.calculate_cpo_all_users(max_users)
        scores_list = list(cpo_scores.values())
        
        # Percentili
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        percentile_values = np.percentile(scores_list, percentiles)
        
        # Utenti più creativi
        sorted_users = sorted(cpo_scores.items(), key=lambda x: x[1], reverse=True)
        most_creative = sorted_users[:10]
        least_creative = sorted_users[-10:]
        
        return {
            'percentiles': dict(zip(percentiles, percentile_values)),
            'most_creative_users': most_creative,
            'least_creative_users': least_creative
        }

# test_cpo_real.py
def test_cpo_implementation():
    """Test della CPO implementation sui dati reali"""
    
    print("=== TEST CPO SU DATI REALI PGPR ===\n")
    
    try:
        # Inizializza CPO
        cpo_metric = CreativePathOriginalityPGPR('ml1m')
        
        print("\n" + "="*50)
        print("1. TEST SU UTENTI CAMPIONE")
        print("="*50)
        
        # Test su primi 5 utenti
        sample_users = list(cpo_metric.paths_data.keys())[:5]
        
        print("CPO scores per utenti campione:")
        for user_id in sample_users:
            cpo_score = cpo_metric.calculate_cpo_for_user(user_id)
            print(f"  User {user_id}: CPO = {cpo_score:.4f}")
        
        print("\n" + "="*50)
        print("2. STATISTICHE GENERALI")
        print("="*50)
        
        stats = cpo_metric.get_summary_statistics(max_users=500)  # Solo 500 per velocità
        
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
            else:
                print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        
        print("\n" + "="*50)
        print("3. ANALISI DISTRIBUZIONE CREATIVITÀ")
        print("="*50)
        
        distribution = cpo_metric.analyze_creativity_distribution(max_users=500)
        
        print("Percentili CPO:")
        for percentile, value in distribution['percentiles'].items():
            print(f"  {percentile}%: {value:.4f}")
        
        print("\nUtenti più creativi:")
        for user_id, score in distribution['most_creative_users'][:5]:
            print(f"  User {user_id}: {score:.4f}")
        
        print("\nUtenti meno creativi:")  
        for user_id, score in distribution['least_creative_users'][:5]:
            print(f"  User {user_id}: {score:.4f}")
            
    except Exception as e:
        print(f"Errore durante il test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cpo_implementation()
