# test_cpo_transe.py
import pickle
import numpy as np
from creativity_metrics.creative_path_originality_transe import CreativePathOriginalityTransE

dataset_name = "ml1m"  # o "lfm1m"
result_dir = f"results/{dataset_name}/transe"

# --- CARICA TOP-K E PATHS ---
topk_items_file = f"{result_dir}/topk_items.pkl"
topk_paths_file = f"{result_dir}/topk_reasoning_paths.pkl"

try:
    with open(topk_paths_file, "rb") as f:
        topk_paths = pickle.load(f)
except FileNotFoundError:
    print(f"Errore: non trovato {topk_paths_file}. Devi eseguire generate_transe_topk_paths.py prima.")
    exit(1)

# --- INIZIALIZZA METRICA CPO ---
cpo_metric = CreativePathOriginalityTransE(dataset_name)

# --- Sostituisci dati interni con quelli appena generati ---
cpo_metric.topk_data = topk_paths  # compatibile con la metrica

# --- CALCOLA CPO per tutti gli utenti (o un campione) ---
max_users = 10
cpo_scores = cpo_metric.calculate_cpo_all_users(max_users=max_users)

# --- STATISTICHE ---
print(f"=== CPO TransE ({dataset_name}) ===")
for uid, score in cpo_scores.items():
    print(f"User {uid}: CPO = {score:.4f}")

avg_cpo = np.mean(list(cpo_scores.values()))
print(f"\nCPO medio su {max_users} utenti: {avg_cpo:.4f}")

