import pickle
import numpy as np
import os

def load_recommendations(results_dir):
    """Carica le raccomandazioni Top-K salvate da CFKG."""
    rec_file = os.path.join(results_dir, "item_topk.pkl")
    with open(rec_file, "rb") as f:
        recs = pickle.load(f)
    return recs  # {user: [item1, item2, ...]}

def load_user_history(data_dir):
    """Carica la cronologia utenti dal train.txt."""
    history = {}
    train_file = os.path.join(data_dir, "train.txt")
    with open(train_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                user = int(parts[0])
                items = list(map(int, parts[1:]))
                history[user] = set(items)
    return history  # {user: {item1, item2, ...}}

def compute_creativity(user_recs, user_history):
    """Creatività = frazione di raccomandazioni nuove (non viste in training)."""
    if not user_recs:
        return 0.0
    new_items = [i for i in user_recs if i not in user_history]
    return len(new_items) / len(user_recs)

def main():
    dataset = "ml1m"
    results_dir = f"results/{dataset}/cfkg"
    data_dir = f"data/{dataset}/preprocessed/cfkg"

    # carica dati
    recs = load_recommendations(results_dir)
    history = load_user_history(data_dir)

    creativity_scores = []
    for user, user_recs in recs.items():
        hist = history.get(user, set())
        c = compute_creativity(user_recs, hist)
        creativity_scores.append(c)

    creativity_scores = np.array(creativity_scores)

    print(f"=== Creatività CFKG su {dataset} ===")
    print(f"Utenti valutati: {len(creativity_scores)}")
    print(f"Media: {creativity_scores.mean():.4f}")
    print(f"Std:   {creativity_scores.std():.4f}")
    print(f"Min:   {creativity_scores.min():.4f}")
    print(f"Max:   {creativity_scores.max():.4f}")

if __name__ == "__main__":
    main()

