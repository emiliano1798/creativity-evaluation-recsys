import os
import pickle
import torch

# ⚠️ Importa il modello CFKG dal codice del repo
from models.cfkg import CFKG  # se il modulo sta in un'altra cartella, adattiamo

# Percorsi
DATASET = "ml1m"
MODEL_DIR = f"results/{DATASET}/cfkg"
SAVE_PATH = os.path.join(MODEL_DIR, "item_embeddings.pkl")

# Parametri di esempio, se hai un file best_hparams_cfg possiamo caricarli lì
hparams_path = os.path.join(MODEL_DIR, "hparams_cfg", "test_metrics.json")

print(f"=== Esportazione embeddings CFKG ({DATASET}) ===")

# 1. Caricare modello
# NB: qui devi sostituire con il metodo usato normalmente per inizializzare CFKG
# Ad esempio:
embedding_dim = 64   # da adattare se diverso
num_users = 6040     # da adattare per ml1m
num_items = 3952     # da adattare per ml1m
num_entities = num_users + num_items
num_relations = 10   # placeholder

model = CFKG(
    n_users=num_users,
    n_items=num_items,
    n_entities=num_entities,
    n_relations=num_relations,
    embed_dim=embedding_dim
)

# 2. Caricare i pesi addestrati
checkpoint_path = os.path.join(MODEL_DIR, "best_hparams_cfg", "checkpoint.pt")
if os.path.exists(checkpoint_path):
    print(f"Carico pesi da {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
else:
    print("⚠️ Nessun checkpoint trovato, uso modello non allenato")

# 3. Estrarre embeddings item
if hasattr(model, "item_embeddings"):
    item_embeds = model.item_embeddings.weight.detach().cpu().numpy()
elif hasattr(model, "entity_embeddings"):
    # se gli item sono un subset degli entity embeddings
    item_offset = num_users  # spesso gli item iniziano dopo gli utenti
    item_embeds = model.entity_embeddings.weight[item_offset:item_offset+num_items].detach().cpu().numpy()
else:
    raise AttributeError("Non ho trovato né item_embeddings né entity_embeddings nel modello")

# 4. Salvare su file
with open(SAVE_PATH, "wb") as f:
    pickle.dump(item_embeds, f)

print(f"✅ Embeddings salvati in {SAVE_PATH} con shape {item_embeds.shape}")

