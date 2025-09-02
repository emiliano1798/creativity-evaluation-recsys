# generate_transe_topk_paths.py
import os
import pickle
import torch
from reasoning_path import TopkReasoningPaths

# --- CONFIGURAZIONE ---
dataset_name = "ml1m"   # o "lfm1m"
result_dir = f"results/{dataset_name}/transe"
K = 10  # top-K items per utente

# --- PERCORSI FILE ---
topk_items_file = os.path.join(result_dir, "topk_items.pkl")
topk_paths_file = os.path.join(result_dir, "topk_reasoning_paths.pkl")

# --- CARICA DATI --- 
# Supponiamo che tu abbia un dizionario pre-calcolato: user_id -> {item_id: score}
# Se non esiste, qui dovresti inserire il codice per calcolare i punteggi TransE
# Ad esempio: user_item_scores = {...}
try:
    with open(os.path.join(result_dir, "user_item_scores.pkl"), "rb") as f:
        user_item_scores = pickle.load(f)
except FileNotFoundError:
    print("Errore: non trovato user_item_scores.pkl. Devi generarlo con il modello TransE.")
    exit(1)

# --- GENERA TOP-K ITEMS ---
topk_items = {}
for uid, scores_dict in user_item_scores.items():
    # ordina item per score decrescente e prendi i primi K
    sorted_items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    topk_items[uid] = [item_id for item_id, _ in sorted_items[:K]]

# salva topk items
os.makedirs(result_dir, exist_ok=True)
with open(topk_items_file, "wb") as f:
    pickle.dump(topk_items, f)
print(f"Salvati top-K items per utente in {topk_items_file}")

# --- GENERA TOP-K REASONING PATHS ---
# Qui costruisci oggetti TopkReasoningPaths compatibili con la metrica CPO
topk_reasoning_paths = {}
for uid, items in topk_items.items():
    # placeholder paths, puoi inserire logica di ragionamento reale se disponibile
    paths_for_user = []
    for item_id in items:
        # struttura minima compatibile con TopkReasoningPaths
        paths_for_user.append((item_id, None, None, None))
    topk_reasoning_paths[uid] = TopkReasoningPaths(dataset_name, paths_for_user, K)

# salva i reasoning paths
with open(topk_paths_file, "wb") as f:
    pickle.dump(topk_reasoning_paths, f)
print(f"Salvati TopkReasoningPaths in {topk_paths_file}")

