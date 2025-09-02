# analysis_script_v3.py
import pickle

def analyze_pgpr_results_v3():
    print("=== ANALISI FINALE PGPR ===\n")
    
    # 1. Analizza raccomandazioni TopK
    print("1. RACCOMANDAZIONI (path_topk.pkl):")
    with open('results/ml1m/pgpr/path_topk.pkl', 'rb') as f:
        topk_data = pickle.load(f)
    
    first_user = list(topk_data.keys())[0]
    topk_obj = topk_data[first_user]
    
    print(f"   Utente {first_user}:")
    print(f"   topk attribute: {topk_obj.topk}")
    print(f"   topk type: {type(topk_obj.topk)}")
    print()
    
    # 2. Analizza path predetti
    print("2. PATH PREDETTI (pred_paths.pkl):")
    with open('results/ml1m/pgpr/pred_paths.pkl', 'rb') as f:
        paths_data = pickle.load(f)
    
    first_user = list(paths_data.keys())[0]
    user_paths = paths_data[first_user]
    
    print(f"   Utente {first_user}:")
    print(f"   Chiavi paths: {list(user_paths.keys())[:10]}...")  # Prime 10 chiavi
    print(f"   Tipo chiavi: {type(list(user_paths.keys())[0])}")
    
    # Prendi primo path
    first_path_key = list(user_paths.keys())[0]
    first_path = user_paths[first_path_key]
    
    print(f"   Primo path (chiave {first_path_key}):")
    print(f"     Contenuto: {first_path}")
    print(f"     Tipo: {type(first_path)}")
    print(f"     Lunghezza: {len(first_path) if hasattr(first_path, '__len__') else 'N/A'}")
    
    # Secondo path
    if len(user_paths) > 1:
        second_path_key = list(user_paths.keys())[1]
        second_path = user_paths[second_path_key]
        print(f"   Secondo path (chiave {second_path_key}):")
        print(f"     Contenuto: {second_path}")

if __name__ == "__main__":
    analyze_pgpr_results_v3()
