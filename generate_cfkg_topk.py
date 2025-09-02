# generate_cfkg_topk.py
import os
import pickle
import torch
from models.knowledge_aware.CFKG.CFKG import CFKG
from models.knowledge_aware.CFKG.loader_cfkg import load_cfkg_dataset

DATASET = "lfm1m"  # o "ml1m"
RESULTS_DIR = f"results/{DATASET}/cfkg/"
TOPK_FILE = os.path.join(RESULTS_DIR, "item_topk.pkl")
K = 10  # top-K raccomandazioni per utente

os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"=== Generazione top-K CFKG ({DATASET}) ===")

# 1. Carica dataset
print("Caricamento dataset CFKG...")
train_data, test_data, user2id, item2id = load_cfkg_dataset(DATASET)
print(f"Utenti: {len(user2id)}, Item: {len(item2id)}")

# 2. Inizializza modello CFKG
print("Inizializzazione modello CFKG...")
model = CFKG(dataset=DATASET)
model.eval()

# 3. Genera raccomandazioni Top-K
topk_dict = {}
print("Calcolo Top-K per ogni utente...")
for uid in user2id.keys():
    scores = model.predict_user(uid)  # restituisce dict {item_id: score}
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    topk_items = [item for item, score in sorted_items[:K]]
    topk_dict[uid] = topk_items

# 4. Salva su file
with open(TOPK_FILE, "wb") as f:
    pickle.dump(topk_dict, f)

print(f"File Top-K salvato in {TOPK_FILE}")
print(f"Generati Top-{K} items per {len(topk_dict)} utenti.")

