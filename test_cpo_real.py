import argparse
import traceback

# Importa la metrica CPO (per ora solo PGPR)
from creativity_metrics.creative_path_originality_pgpr import CreativePathOriginalityPGPR


def test_cpo_implementation(dataset, model):
    """Test della CPO implementation sui dati reali"""

    print(f"=== TEST CPO SU DATI REALI ({model.upper()} - {dataset}) ===\n")

    try:
        # Inizializza CPO
        if model == "pgpr":
            cpo_metric = CreativePathOriginalityPGPR(dataset)
        else:
            raise NotImplementedError(
                f"Modello {model} non ancora supportato. Implementa la classe corrispondente!"
            )

        # Test su primi 5 utenti
        sample_users = list(cpo_metric.paths_data.keys())[:5]

        print("CPO scores per utenti campione:")
        for user_id in sample_users:
            cpo_score = cpo_metric.calculate_cpo_for_user(user_id)
            print(f"  User {user_id}: CPO = {cpo_score:.4f}")

        # Statistiche su campione
        stats = cpo_metric.get_summary_statistics(max_users=100)
        print("\nStatistiche (primi 100 utenti):")
        print(f"  CPO medio: {stats['avg_cpo']:.4f}")
        print(f"  CPO std: {stats['std_cpo']:.4f}")
        print(f"  Pattern totali: {stats['total_patterns']}")

    except Exception as e:
        print(f"Errore: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml1m", help="Dataset (ml1m o lfm1m)")
    parser.add_argument("--model", type=str, default="pgpr", help="Modello (pgpr, cfkg, transe)")
    args = parser.parse_args()

    test_cpo_implementation(args.dataset, args.model)

