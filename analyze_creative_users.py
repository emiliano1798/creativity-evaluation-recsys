# analyze_creative_users.py
from creativity_metrics.creative_path_originality_pgpr import CreativePathOriginalityPGPR
import os
import numpy as np

def analyze_most_creative_users():
    print("=== ANALISI UTENTI PIÙ CREATIVI ===\n")
    
    cpo_metric = CreativePathOriginalityPGPR('ml1m')
    
    # Focus sui top creative users
    creative_users = [163, 74, 499, 45, 157]
    
    for user_id in creative_users:
        print(f"USER {user_id} (CPO: {cpo_metric.calculate_cpo_for_user(user_id):.4f}):")
        
        user_paths = cpo_metric.paths_data[user_id]
        print(f"  Totale raccomandazioni: {len(user_paths)}")
        
        # Analizza pattern utilizzati
        patterns_used = []
        creative_paths = []
        
        for item_id, path_info_list in user_paths.items():
            for path_info in path_info_list:
                if len(path_info) >= 3:
                    path_sequence = path_info[2]
                    pattern = cpo_metric._get_path_pattern(path_sequence)
                    patterns_used.append(pattern)
                    
                    # Se pattern raro, salva per analisi
                    if pattern != 'multi_3_watched_watched':
                        creative_paths.append((item_id, pattern, path_sequence))
        
        # Statistiche pattern
        from collections import Counter
        pattern_counts = Counter(patterns_used)
        print(f"  Pattern utilizzati:")
        for pattern, count in pattern_counts.most_common():
            percentage = count / len(patterns_used) * 100
            print(f"    {pattern}: {count} ({percentage:.1f}%)")
        
        # Mostra path creativi (non-standard)
        if creative_paths:
            print(f"  Path creativi (primi 3):")
            for i, (item_id, pattern, path_seq) in enumerate(creative_paths[:3]):
                print(f"    Item {item_id} [{pattern}]: {path_seq}")
        
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    analyze_most_creative_users()
