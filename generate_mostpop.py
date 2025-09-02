#!/usr/bin/env python3
"""
Script per generare il file most_pop/item_topks.pkl mancante
"""

import os
import pickle
import pandas as pd
import csv
from collections import defaultdict, Counter
from utils import *

def generate_most_popular_items(dataset_name, k=100):
    """
    Genera le raccomandazioni most-popular per tutti gli utenti
    """
    print(f"Generating most popular items for {dataset_name}...")

    # Crea la directory se non esiste
    result_dir = get_result_dir(dataset_name)
    mostpop_dir = os.path.join(result_dir, "most_pop")
    ensure_dir(mostpop_dir)

    # Carica i dati di training per calcolare la popolarità
    data_dir = get_data_dir(dataset_name)

    # Conta la popolarità degli item dai dati di training
    item_popularity = Counter()

    # Leggi le interazioni di training
    train_file = os.path.join(data_dir, "train.txt")
    if os.path.exists(train_file):
        with open(train_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 2:
                    user_id = row[0]
                    item_id = row[1] 
                    item_popularity[item_id] += 1
    else:
        # Prova a leggere da products.txt se train.txt non esiste
        products_file = os.path.join(data_dir, "products.txt")
        if os.path.exists(products_file):
            df_products = pd.read_csv(products_file, sep='\t')
            if 'pop_item' in df_products.columns:
                # Usa la popolarità pre-calcolata
                for _, row in df_products.iterrows():
                    item_popularity[str(row['pid'])] = row['pop_item']
            else:
                print("Warning: No popularity data found, using uniform distribution")
                for _, row in df_products.iterrows():
                    item_popularity[str(row['pid'])] = 1

    # Ordina gli item per popolarità (dal più al meno popolare)
    most_popular_items = [item for item, count in item_popularity.most_common()]

    print(f"Found {len(most_popular_items)} items")

    # Carica tutti gli utenti
    users_file = os.path.join(data_dir, "users.txt")
    users = []
    if os.path.exists(users_file):
        with open(users_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)  # Skip header
            for row in reader:
                users.append(row[0])

    # Genera top-k most popular per ogni utente
    most_pop_topks = {}
    for user_id in users:
        # Per semplicità, diamo gli stessi item più popolari a tutti gli utenti
        # In una implementazione più sofisticata potresti filtrare gli item già visti
        most_pop_topks[user_id] = most_popular_items[:k]

    # Salva il risultato
    output_file = os.path.join(mostpop_dir, "item_topks.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(most_pop_topks, f)

    print(f"Saved most popular items to {output_file}")
    print(f"Generated recommendations for {len(most_pop_topks)} users")
    return most_pop_topks

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate most popular items file')
    parser.add_argument('--dataset', '-d', default='ml1m', help='Dataset name')
    parser.add_argument('--k', '-k', type=int, default=100, help='Top-k items')

    args = parser.parse_args()

    generate_most_popular_items(args.dataset, args.k)

if __name__ == '__main__':
    main()
