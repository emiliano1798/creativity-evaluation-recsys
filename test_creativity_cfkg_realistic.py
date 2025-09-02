# test_creativity_cfkg_realistic.py
from creativity_metrics.creative_recommendation_cfkg_realistic import CreativeRecommendationCFKG
import numpy as np

def main():
    print("=== Test Creatività CFKG (versione realistica) ===")
    
    # Inizializzo la metrica
    metric = CreativeRecommendationCFKG(dataset_name="ml1m", k=10)
    
    # Valutazione su un campione di utenti
    sample_users = 100
    scores = metric.evaluate_creativity(sample_users=sample_users)
    
    # Statistiche
    values = np.array(list(scores.values()))
    print(f"Utenti valutati: {len(scores)}")
    print(f"Media: {values.mean():.4f}")
    print(f"Std:   {values.std():.4f}")
    print(f"Min:   {values.min():.4f}")
    print(f"Max:   {values.max():.4f}")
    
    # Dettaglio utenti
    print("Dettaglio utenti:")
    for uid, val in scores.items():
        print(f"User {uid}: Creativity = {val:.4f}")

if __name__ == "__main__":
    main()

