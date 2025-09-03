import numpy as np
import pickle
import json
import os
from collections import defaultdict, Counter
from scipy.spatial.distance import cosine
from scipy.stats import entropy
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class CFKGCreativityMetric:
    """
    Metrica di creatività composita per il modello CFKG che combina:
    1. Path-based Novelty - connessioni creative tramite KG paths
    2. Entity Diversity - diversità semantica degli item raccomandati
    3. Popularity-based Surprise - quanto sono inaspettati gli item
    4. Knowledge Graph Distance - distanza semantica dal profilo utente
    """
    
    def __init__(self, dataset='ml1m', base_path='rep-path-reasoning-recsys'):
        """
        Inizializza la metrica di creatività
        
        Args:
            dataset: 'ml1m' o 'lfm1m'
            base_path: percorso base del repository
        """
        self.dataset = dataset
        self.base_path = base_path
        
        # Percorsi ai dati
        self.data_path = f"{base_path}/data/{dataset}/preprocessed/cfkg"
        self.results_path = f"{base_path}/results/{dataset}/cfkg"
        self.model_weights_path = f"{self.results_path}/best_params_cfg/weights"
        
        print(f"Inizializzando metrica creatività per dataset: {dataset}")
        print(f"Data path: {self.data_path}")
        print(f"Results path: {self.results_path}")
        
        # Carica i dati del knowledge graph
        self.kg_triples = self._load_kg_triples()
        self.relations = self._load_relations()
        self.entities = self._load_entities()
        self.n_users, self.n_items, self.n_entities = self._load_dataset_stats()
        
        # Costruisce il grafo per l'analisi dei path
        print("Costruendo grafo KG per analisi path...")
        self.kg_graph = self._build_kg_graph()
        
        # Calcola statistiche per le metriche
        print("Calcolando statistiche item...")
        self.item_popularity = self._calculate_item_popularity()
        
        # Modello e embedding
        self.model = None
        self.sess = None
        self.entity_embeddings = None
        
        # Pesi per la combinazione delle metriche (personalizzabili)
        self.weights = {
            'path_novelty': 0.25,
            'entity_diversity': 0.25,
            'popularity_surprise': 0.25,
            'kg_distance': 0.25
        }
    
    def _load_dataset_stats(self):
        """Carica statistiche base del dataset"""
        # Conta users dal file train.txt
        users = set()
        items = set()
        
        with open(f"{self.data_path}/train.txt", 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    user_id = int(parts[0])
                    user_items = list(map(int, parts[1:]))
                    users.add(user_id)
                    items.update(user_items)
        
        # Conta entità dal file entity_list.txt
        n_entities = len(self.entities)
        
        return len(users), len(items), n_entities
    
    def _load_kg_triples(self):
        """Carica le triple del knowledge graph"""
        triples = []
        kg_file = f"{self.data_path}/kg_final.txt"
        print(f"Caricando KG da: {kg_file}")
        
        with open(kg_file, 'r') as f:
            for line in f:
                h, r, t = map(int, line.strip().split())
                triples.append((h, r, t))
        
        print(f"Caricate {len(triples)} triple del KG")
        return triples
    
    def _load_relations(self):
        """Carica la mappatura delle relazioni"""
        relations = {}
        rel_file = f"{self.data_path}/relation_list.txt"
        
        with open(rel_file, 'r') as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    rel_name = parts[0]
                    rel_id = int(parts[1])
                    relations[rel_id] = rel_name
        
        print(f"Caricate {len(relations)} relazioni")
        return relations
    
    def _load_entities(self):
        """Carica la mappatura delle entità"""
        entities = {}
        ent_file = f"{self.data_path}/entity_list.txt"
        
        with open(ent_file, 'r') as f:
            next(f)  # skip header
            next(f)  # skip second header
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    entity_name = parts[0]
                    entity_id = int(parts[1])
                    entities[entity_id] = entity_name
        
        print(f"Caricate {len(entities)} entità")
        return entities
    
    def _build_kg_graph(self):
        """Costruisce un grafo NetworkX dal KG per l'analisi dei path"""
        G = nx.MultiDiGraph()
        for h, r, t in self.kg_triples:
            G.add_edge(h, t, relation=r)
        
        print(f"Grafo KG creato: {G.number_of_nodes()} nodi, {G.number_of_edges()} archi")
        return G
    
    def _calculate_item_popularity(self):
        """Calcola la popolarità degli item dal training set"""
        item_counts = Counter()
        
        # Carica i dati di training
        with open(f"{self.data_path}/train.txt", 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    user_id = int(parts[0])
                    items = list(map(int, parts[1:]))
                    for item in items:
                        item_counts[item] += 1
        
        # Normalizza in probabilità
        total_interactions = sum(item_counts.values())
        item_popularity = {item: count/total_interactions 
                          for item, count in item_counts.items()}
        
        print(f"Calcolata popolarità per {len(item_popularity)} item")
        return item_popularity
    
    def load_cfkg_model(self):
        """Carica il modello CFKG trainato per ottenere embedding reali"""
        try:
            from models.knowledge_aware.CFKG.CFKG import CFKG
            from models.knowledge_aware.CFKG.parser import parse_args
            
            print("Caricando modello CFKG trainato...")
            
            # Configurazione del modello (usa parametri standard)
            args = parse_args()
            args.dataset = self.dataset
            if self.dataset == 'ml1m':
                args.embed_size = 128
                args.kge_size = 64
            else:  # lfm1m
                args.embed_size = 64
                args.kge_size = 64
            
            config = {
                'n_users': self.n_users,
                'n_items': self.n_items,
                'n_entities': self.n_entities,
                'n_relations': len(self.relations)
            }
            
            # Crea modello
            self.model = CFKG(data_config=config, pretrain_data=None, args=args)
            
            # Crea sessione TensorFlow
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=tf_config)
            
            # Carica i pesi trainati
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.model_weights_path + '/checkpoint'))
            
            if ckpt and ckpt.model_checkpoint_path:
                self.sess.run(tf.global_variables_initializer())
                saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("Modello CFKG caricato con successo!")
                
                # Estrai embedding
                self._extract_embeddings()
                return True
            else:
                print(f"Checkpoint non trovato in: {self.model_weights_path}")
                print("Usando embedding casuali...")
                self._generate_random_embeddings()
                return False
                
        except Exception as e:
            print(f"Errore nel caricamento del modello: {e}")
            print("Usando embedding casuali...")
            self._generate_random_embeddings()
            return False
    
    def _extract_embeddings(self):
        """Estrae gli embeddings delle entità dal modello CFKG"""
        if self.sess is None or self.model is None:
            print("Modello non disponibile, usando embedding casuali")
            self._generate_random_embeddings()
            return
        
        try:
            # Estrae embedding dal modello CFKG
            embeddings = tf.concat([self.model.weights['user_embed'], 
                                   self.model.weights['entity_embed']], axis=0)
            self.entity_embeddings = self.sess.run(embeddings)
            print(f"Estratti embedding reali: shape {self.entity_embeddings.shape}")
            
        except Exception as e:
            print(f"Errore nell'estrazione degli embedding: {e}")
            self._generate_random_embeddings()
    
    def _generate_random_embeddings(self):
        """Genera embedding casuali per demo/fallback"""
        embed_dim = 64
        total_entities = self.n_users + self.n_entities
        self.entity_embeddings = np.random.randn(total_entities, embed_dim)
        print(f"Generati embedding casuali: shape {self.entity_embeddings.shape}")
    
    def _get_user_profile_entities(self, user_id):
        """Ottiene le entità del profilo utente dal training set"""
        user_items = []
        with open(f"{self.data_path}/train.txt", 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and int(parts[0]) == user_id:
                    user_items = list(map(int, parts[1:]))
                    break
        
        # Converti item IDs in entity IDs (gli item sono mappati come entity_id = item_id + n_users)
        user_items_as_entities = [item + self.n_users for item in user_items]
        
        # Trova entità connesse agli item del profilo utente nel KG
        profile_entities = set(user_items_as_entities)
        for item_entity in user_items_as_entities:
            for h, r, t in self.kg_triples:
                if h == item_entity:
                    profile_entities.add(t)
                elif t == item_entity:
                    profile_entities.add(h)
        
        return list(profile_entities)
    
    def calculate_path_novelty(self, user_id, recommended_items, max_path_length=3):
        """
        1. Path-based Novelty: Misura la creatività basata sui path nel KG
        """
        user_profile = self._get_user_profile_entities(user_id)
        path_novelty_scores = []
        
        # Converti recommended_items in entity IDs
        recommended_entities = [item + self.n_users for item in recommended_items]
        
        for item_entity in recommended_entities:
            item_novelty = 0
            path_count = 0
            
            # Trova path dal profilo utente all'item raccomandato
            for profile_entity in user_profile[:10]:  # limita per performance
                try:
                    if self.kg_graph.has_node(profile_entity) and self.kg_graph.has_node(item_entity):
                        if nx.has_path(self.kg_graph, profile_entity, item_entity):
                            paths = list(nx.all_simple_paths(
                                self.kg_graph, profile_entity, item_entity, 
                                cutoff=max_path_length))[:5]  # limita numero di path
                            
                            for path in paths:
                                path_count += 1
                                # Calcola novelty del path
                                path_relations = []
                                for i in range(len(path)-1):
                                    edge_data = self.kg_graph.get_edge_data(path[i], path[i+1])
                                    if edge_data:
                                        relation = list(edge_data.values())[0]['relation']
                                        path_relations.append(relation)
                                
                                # Novelty basata su rarità delle relazioni
                                relation_counts = Counter([r for _, r, _ in self.kg_triples])
                                relation_rarity = sum([1/max(1, relation_counts[rel]) 
                                                     for rel in path_relations])
                                
                                # Bonus per path di lunghezza inaspettata
                                length_bonus = 1 / (len(path) ** 0.5) if len(path) > 1 else 0
                                
                                item_novelty += relation_rarity + length_bonus
                                
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
            
            # Normalizza per numero di path trovati
            if path_count > 0:
                path_novelty_scores.append(item_novelty / path_count)
            else:
                path_novelty_scores.append(0)
        
        return np.mean(path_novelty_scores) if path_novelty_scores else 0
    
    def calculate_entity_diversity(self, recommended_items):
        """
        2. Entity Diversity: Misura la diversità semantica degli item raccomandati
        """
        if self.entity_embeddings is None:
            return 0
        
        if len(recommended_items) < 2:
            return 0
        
        # Converti item IDs in entity IDs e ottieni embedding
        try:
            recommended_entities = [item + self.n_users for item in recommended_items]
            valid_entities = [ent for ent in recommended_entities 
                            if ent < len(self.entity_embeddings)]
            
            if len(valid_entities) < 2:
                return 0
                
            item_embeddings = self.entity_embeddings[valid_entities]
        except IndexError:
            return 0
        
        # Calcola diversità come media delle distanze coseno
        similarities = cosine_similarity(item_embeddings)
        
        # Diversità intra-lista (1 - similarità media)
        n_items = len(item_embeddings)
        total_similarity = 0
        count = 0
        
        for i in range(n_items):
            for j in range(i+1, n_items):
                total_similarity += similarities[i][j]
                count += 1
        
        avg_similarity = total_similarity / count if count > 0 else 0
        diversity = 1 - avg_similarity
        
        return max(0, diversity)  # assicura non negatività
    
    def calculate_popularity_surprise(self, recommended_items):
        """
        3. Popularity-based Surprise: Misura quanto sono inaspettati gli item
        """
        if not recommended_items:
            return 0
        
        item_surprises = []
        for item in recommended_items:
            popularity = self.item_popularity.get(item, 0.0001)  # default basso per item sconosciuti
            # Surprise inversamente proporzionale alla popolarità
            surprise = -np.log(popularity + 1e-10)  # evita log(0)
            item_surprises.append(surprise)
        
        # Media delle surprise + bonus per distribuzione uniforme
        mean_surprise = np.mean(item_surprises)
        
        # Bonus per diversità delle popolarità
        popularities = [self.item_popularity.get(item, 0.0001) for item in recommended_items]
        popularity_entropy = entropy(popularities) if len(popularities) > 1 else 0
        
        return mean_surprise + 0.1 * popularity_entropy
    
    def calculate_kg_distance(self, user_id, recommended_items):
        """
        4. Knowledge Graph Distance: Misura la distanza semantica dal profilo utente
        """
        if self.entity_embeddings is None:
            return 0
            
        user_profile = self._get_user_profile_entities(user_id)
        
        if not user_profile or not recommended_items:
            return 0
        
        kg_distances = []
        
        try:
            # Limita profilo per performance
            valid_profile = [ent for ent in user_profile[:20] 
                           if ent < len(self.entity_embeddings)]
            if not valid_profile:
                return 0
                
            profile_embeddings = self.entity_embeddings[valid_profile]
            profile_centroid = np.mean(profile_embeddings, axis=0)
            
            # Converti item in entity IDs
            recommended_entities = [item + self.n_users for item in recommended_items]
            
            for entity in recommended_entities:
                if entity < len(self.entity_embeddings):
                    item_embedding = self.entity_embeddings[entity]
                    
                    # Distanza coseno dal centroide del profilo
                    distance = cosine(profile_centroid, item_embedding)
                    
                    # Creatività ottimale a distanza media (non troppo vicino/lontano)
                    optimal_distance = 0.5
                    creativity = 1 - abs(distance - optimal_distance)
                    kg_distances.append(max(0, creativity))
                    
        except (IndexError, ValueError):
            return 0
        
        return np.mean(kg_distances) if kg_distances else 0
    
    def calculate_creativity_score(self, user_id, recommended_items):
        """
        Calcola il punteggio di creatività composito
        """
        # Calcola le singole componenti
        path_novelty = self.calculate_path_novelty(user_id, recommended_items)
        entity_diversity = self.calculate_entity_diversity(recommended_items)
        popularity_surprise = self.calculate_popularity_surprise(recommended_items)
        kg_distance = self.calculate_kg_distance(user_id, recommended_items)
        
        # Punteggio composito pesato
        composite_score = (
            self.weights['path_novelty'] * path_novelty +
            self.weights['entity_diversity'] * entity_diversity +
            self.weights['popularity_surprise'] * popularity_surprise +
            self.weights['kg_distance'] * kg_distance
        )
        
        return {
            'path_novelty': path_novelty,
            'entity_diversity': entity_diversity,
            'popularity_surprise': popularity_surprise,
            'kg_distance': kg_distance,
            'composite_creativity': composite_score,
            'individual_scores': {
                'path_novelty': path_novelty,
                'entity_diversity': entity_diversity,
                'popularity_surprise': popularity_surprise,
                'kg_distance': kg_distance
            }
        }
    
    def evaluate_recommendations(self, top_k=10, save_detailed=True):
        """
        Valuta la creatività per tutte le raccomandazioni salvate
        
        Args:
            top_k: numero di item da considerare per utente
            save_detailed: salva risultati dettagliati per ogni utente
            
        Returns:
            dict con statistiche di creatività
        """
        # Percorso al file delle raccomandazioni
        recommendations_file = f"{self.results_path}/item_topk.pkl"
        
        if not os.path.exists(recommendations_file):
            print(f"File raccomandazioni non trovato: {recommendations_file}")
            return None
        
        # Carica le raccomandazioni
        print(f"Caricando raccomandazioni da: {recommendations_file}")
        with open(recommendations_file, 'rb') as f:
            all_recommendations = pickle.load(f)
        
        print(f"Trovate raccomandazioni per {len(all_recommendations)} utenti")
        
        creativity_scores = []
        detailed_results = {}
        
        print(f"Valutando creatività per {len(all_recommendations)} utenti (top-{top_k})...")
        
        for i, (user_id, recommended_items) in enumerate(all_recommendations.items()):
            if i % 100 == 0:
                print(f"Processati {i}/{len(all_recommendations)} utenti")
            
            # Considera solo i top-k item
            top_items = recommended_items[:top_k]
            
            # Calcola creatività
            creativity_result = self.calculate_creativity_score(user_id, top_items)
            creativity_scores.append(creativity_result['composite_creativity'])
            
            if save_detailed:
                detailed_results[user_id] = creativity_result
        
        # Statistiche finali
        results = {
            'dataset': self.dataset,
            'top_k': top_k,
            'n_users_evaluated': len(all_recommendations),
            'overall_creativity': {
                'mean': float(np.mean(creativity_scores)),
                'std': float(np.std(creativity_scores)),
                'median': float(np.median(creativity_scores)),
                'min': float(np.min(creativity_scores)),
                'max': float(np.max(creativity_scores))
            },
            'component_averages': {
                'path_novelty': float(np.mean([r['path_novelty'] for r in detailed_results.values()])),
                'entity_diversity': float(np.mean([r['entity_diversity'] for r in detailed_results.values()])),
                'popularity_surprise': float(np.mean([r['popularity_surprise'] for r in detailed_results.values()])),
                'kg_distance': float(np.mean([r['kg_distance'] for r in detailed_results.values()]))
            },
            'weights_used': self.weights
        }
        
        if save_detailed:
            results['detailed_results'] = detailed_results
        
        return results
    
    def set_weights(self, **kwargs):
        """Permette di modificare i pesi delle componenti"""
        for component, weight in kwargs.items():
            if component in self.weights:
                self.weights[component] = weight
        
        # Normalizza i pesi
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for component in self.weights:
                self.weights[component] /= total_weight
    
    def save_results(self, results, filename=None):
        """Salva i risultati in un file JSON"""
        if filename is None:
            filename = f"cfkg_creativity_results_{self.dataset}.json"
        
        # Rimuovi detailed_results se troppo grande
        save_results = results.copy()
        if 'detailed_results' in save_results:
            print(f"Salvando risultati dettagliati per {len(save_results['detailed_results'])} utenti")
        
        with open(filename, 'w') as f:
            json.dump(save_results, f, indent=2)
        
        print(f"Risultati salvati in: {filename}")
        return filename


# Script principale per l'esecuzione
def main():
    """Funzione principale per eseguire la valutazione di creatività"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Valuta creatività delle raccomandazioni CFKG")
    parser.add_argument('--dataset', type=str, default='ml1m', choices=['ml1m', 'lfm1m'],
                       help='Dataset da utilizzare')
    parser.add_argument('--base_path', type=str, default='.',
                       help='Percorso base del repository')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Numero di item da considerare per utente')
    parser.add_argument('--load_model', action='store_true',
                       help='Carica il modello CFKG per embedding reali')
    parser.add_argument('--save_detailed', action='store_true', default=True,
                       help='Salva risultati dettagliati per ogni utente')
    
    args = parser.parse_args()
    
    # Inizializza la metrica
    print(f"=== VALUTAZIONE CREATIVITÀ CFKG - Dataset: {args.dataset} ===")
    metric = CFKGCreativityMetric(dataset=args.dataset, base_path=args.base_path)
    
    # Carica il modello se richiesto
    if args.load_model:
        metric.load_cfkg_model()
    else:
        print("Usando embedding casuali (modello non caricato)")
        metric._generate_random_embeddings()
    
    # Valuta le raccomandazioni
    print("\nIniziando valutazione creatività...")
    results = metric.evaluate_recommendations(top_k=args.top_k, save_detailed=args.save_detailed)
    
    if results is None:
        print("Errore: impossibile completare la valutazione")
        return
    
    # Mostra risultati
    print("\n=== RISULTATI CREATIVITÀ CFKG ===")
    print(f"Dataset: {results['dataset']}")
    print(f"Utenti valutati: {results['n_users_evaluated']}")
    print(f"Top-K considerati: {results['top_k']}")
    print(f"\nCreatività complessiva:")
    print(f"  Media: {results['overall_creativity']['mean']:.4f}")
    print(f"  Mediana: {results['overall_creativity']['median']:.4f}")
    print(f"  Deviazione standard: {results['overall_creativity']['std']:.4f}")
    print(f"  Min-Max: [{results['overall_creativity']['min']:.4f}, {results['overall_creativity']['max']:.4f}]")
    
    print(f"\nComponenti individuali (media):")
    for component, score in results['component_averages'].items():
        print(f"  {component}: {score:.4f}")
    
    print(f"\nPesi utilizzati:")
    for component, weight in results['weights_used'].items():
        print(f"  {component}: {weight:.2f}")
    
    # Salva risultati
    output_file = metric.save_results(results)
    print(f"\nRisultati completi salvati in: {output_file}")
    
    # Chiudi sessione se aperta
    if metric.sess is not None:
        metric.sess.close()


if __name__ == "__main__":
    main()
