import pickle

# Carica le raccomandazioni
with open('results/ml1m/cfkg/item_topk.pkl', 'rb') as f:
    topk_data = pickle.load(f)

# Mostra la struttura
print("Tipo di dato:", type(topk_data))
print("Numero di utenti:", len(topk_data))

# Mostra i primi 3 utenti e le loro raccomandazioni
for i, (user_id, recommended_items) in enumerate(list(topk_data.items())[:3]):
    print(f"User {user_id}: {recommended_items[:10]}")
