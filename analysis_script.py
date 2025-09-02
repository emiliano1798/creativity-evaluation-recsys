import pickle
import pandas as pd
import numpy as np
from collections import Counter

def analyze_pgpr_results():
    print("=== ANALISI RISULTATI PGPR ===\n")
    
    # 1. Analizza raccomandazioni TopK
    print("1. RACCOMANDAZIONI (path_topk.pkl):")
    try:
        with open('results/ml1m/pgpr/path_topk.pkl', 'rb') as f:
            topk_data = pickle.load(f)
        
        print(f"   Tipo: {type(topk_data)}")
        if isinstance(topk_data, dict):
            print(f"   Numero utenti: {len(topk_data)}")
            first_user = list(topk_data.keys())[0]
            print(f"   Esempio utente {first_user}: {topk_data[first_user][:3]}...")
            
    except Exception as e:
        print(f"   Errore: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # 2. Analizza path predetti
    print("2. PATH PREDETTI (pred_paths.pkl):")
    try:
        with open('results/ml1m/pgpr/pred_paths.pkl', 'rb') as f:
            paths_data = pickle.load(f)
        
        print(f"   Tipo: {type(paths_data)}")
        if isinstance(paths_data, dict):
            print(f"   Numero utenti: {len(paths_data)}")
            
            first_user = list(paths_data.keys())[0]
            user_paths = paths_data[first_user]
            print(f"   Esempio utente {first_user}:")
            print(f"     Numero path: {len(user_paths)}")
            print(f"     Primo path: {user_paths[0]}")
            print(f"     Secondo path: {user_paths[1] if len(user_paths) > 1 else 'N/A'}")
        
    except Exception as e:
        print(f"   Errore: {e}")

if __name__ == "__main__":
    analyze_pgpr_results()
