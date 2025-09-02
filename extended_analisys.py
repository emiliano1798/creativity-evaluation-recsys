import os
import numpy as np
# extended_analysis.py
from creativity_metrics.creative_path_originality_pgpr import CreativePathOriginalityPGPR

def detailed_analysis():
    print("=== ANALISI DETTAGLIATA CPO ===\n")
    
    cpo_metric = CreativePathOriginalityPGPR('ml1m')
    
    # 1. Analizza i pattern più comuni
    print("1. PATTERN PIÙ COMUNI:")
    sorted_patterns = sorted(cpo_metric.path_patterns.items(), 
                           key=lambda x: x[1], reverse=True)
    
    for pattern, probability in sorted_patterns:
        print(f"   {pattern}: {probability:.4f} ({probability*100:.2f}%)")
    
    print("\n" + "="*50)
    
    # 2. Analisi distribuzione completa
    print("2. DISTRIBUZIONE CREATIVITÀ (500 utenti):")
    distribution = cpo_metric.analyze_creativity_distribution(max_users=500)
    
    print("Percentili CPO:")
    for percentile, value in distribution['percentiles'].items():
        print(f"   {percentile}%: {value:.4f}")
    
    print(f"\nUtenti PIÙ creativi:")
    for i, (user_id, score) in enumerate(distribution['most_creative_users'][:5]):
        print(f"   #{i+1}: User {user_id} = {score:.4f}")
    
    print(f"\nUtenti MENO creativi:")
    for i, (user_id, score) in enumerate(distribution['least_creative_users'][:5]):
        print(f"   #{i+1}: User {user_id} = {score:.4f}")
    
    print("\n" + "="*50)
    
    # 3. Confronto con metriche esistenti
    print("3. CONFRONTO CON METRICHE ESISTENTI:")
    compare_with_existing_metrics(cpo_metric)

def compare_with_existing_metrics(cpo_metric):
    """Confronta CPO con serendipity, novelty, etc."""
    try:
        import pandas as pd
        
        # Carica metriche esistenti
        rec_quality_file = 'results/ml1m/pgpr/rec_quality_group_avg_values.csv'
        path_quality_file = 'results/ml1m/pgpr/path_quality_group_avg_values.csv'
        
        if os.path.exists(rec_quality_file):
            rec_metrics = pd.read_csv(rec_quality_file)
            print("Metriche recommendation quality disponibili:")
            print(f"   Colonne: {list(rec_metrics.columns)}")
            
        if os.path.exists(path_quality_file):
            path_metrics = pd.read_csv(path_quality_file)
            print("Metriche path quality disponibili:")
            print(f"   Colonne: {list(path_metrics.columns)}")
            
        # Calcola CPO per confronto
        sample_cpo = cpo_metric.calculate_cpo_all_users(max_users=100)
        avg_cpo = np.mean(list(sample_cpo.values()))
        
        print(f"\nConfronto:")
        print(f"   CPO medio (nostro): {avg_cpo:.4f}")
        
        # Se disponibili, mostra altre metriche per confronto
        if os.path.exists(rec_quality_file):
            rec_data = pd.read_csv(rec_quality_file)
            if 'serendipity' in rec_data.columns:
                print(f"   Serendipity: {rec_data['serendipity'].mean():.4f}")
            if 'novelty' in rec_data.columns:
                print(f"   Novelty: {rec_data['novelty'].mean():.4f}")
        
    except Exception as e:
        print(f"   Errore nel confronto: {e}")

if __name__ == "__main__":
    detailed_analysis()
