# analysis_script_v2.py
import pickle
import pandas as pd
import numpy as np

def analyze_pgpr_results_v2():
    print("=== ANALISI RISULTATI PGPR V2 ===\n")
    
    # 1. Analizza raccomandazioni TopK
    print("1. RACCOMANDAZIONI (path_topk.pkl):")
    try:
        with open('results/ml1m/pgpr/path_topk.pkl', 'rb') as f:
            topk_data = pickle.load(f)
        
        print(f"   Tipo: {type(topk_data)}")
        print(f"   Numero utenti: {len(topk_data)}")
        
        # Analizza primo utente
        first_user = list(topk_data.keys())[0]
        topk_obj = topk_data[first_user]
        
        print(f"   Utente {first_user}:")
        print(f"   Tipo oggetto: {type(topk_obj)}")
        print(f"   Attributi: {dir(topk_obj)}")
        
        # Prova ad accedere agli attributi
        if hasattr(topk_obj, 'topk_items'):
            print(f"   TopK items: {topk_obj.topk_items[:5]}")
        if hasattr(topk_obj, 'items'):
            print(f"   Items: {topk_obj.items[:5]}")
        if hasattr(topk_obj, 'scores'):
            print(f"   Scores: {topk_obj.scores[:5]}")
            
    except Exception as e:
        print(f"   Errore: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50 + "\n")
    
    # 2. Analizza path predetti
    print("2. PATH PREDETTI (pred_paths.pkl):")
    try:
        with open('results/ml1m/pgpr/pred_paths.pkl', 'rb') as f:
            paths_data = pickle.load(f)
        
        print(f"   Tipo: {type(paths_data)}")
        print(f"   Numero utenti: {len(paths_data)}")
        
        first_user = list(paths_data.keys())[0]
        user_paths = paths_data[first_user]
        
        print(f"   Utente {first_user}:")
        print(f"   Numero path: {len(user_paths)}")
        print(f"   Tipo path list: {type(user_paths)}")
        
        if len(user_paths) > 0:
            print(f"   Primo path:")
            print(f"     Tipo: {type(user_paths[0])}")
            print(f"     Contenuto: {user_paths[0]}")
            
        if len(user_paths) > 1:
            print(f"   Secondo path:")
            print(f"     Tipo: {type(user_paths[1])}")  
            print(f"     Contenuto: {user_paths[1]}")
        
    except Exception as e:
        print(f"   Errore: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_pgpr_results_v2()
