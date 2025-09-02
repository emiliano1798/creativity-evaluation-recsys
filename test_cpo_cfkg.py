# test_cpo_cfkg.py

from creativity_metrics.creative_path_originality_cfkg import CreativePathOriginalityCFKG

def main():
    # Inizializza la metrica CPO per CFKG
    cfkg_metric = CreativePathOriginalityCFKG('lfm1m', topk=5)

    # Calcola la CPO per i primi 10 utenti
    scores = cfkg_metric.calculate_cpo_all_users(max_users=10)

    # Stampa la media
    avg_cpo = sum(scores.values()) / len(scores)
    print("CFKG CPO (media su 10 utenti):", avg_cpo)

    # Stampa anche i singoli valori per ogni utente
    for user_id, score in scores.items():
        print(f"User {user_id}: CPO = {score:.4f}")

if __name__ == "__main__":
    main()

