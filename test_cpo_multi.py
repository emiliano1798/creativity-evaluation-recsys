import argparse
from creativity_metrics.creative_path_originality_pgpr import CreativePathOriginalityPGPR
from creativity_metrics.creative_path_originality_base import CreativePathOriginalityBase

def test_cpo(dataset, model):
    print(f"=== TEST CPO ({model.upper()} - {dataset}) ===")

    if model.lower() == "pgpr":
        cpo_metric = CreativePathOriginalityPGPR(dataset)
    else:
        cpo_metric = CreativePathOriginalityBase(dataset, model_name=model.lower())

    # Calcola CPO per primi 5 utenti
    users = list(cpo_metric.topk_data.keys())[:5]
    for user_id in users:
        score = cpo_metric.calculate_cpo_for_user(user_id)
        print(f"User {user_id}: CPO = {score:.4f}")

    # Statistiche generali
    stats = cpo_metric.get_summary_statistics(max_users=100)
    print(f"\nCPO medio: {stats['avg_cpo']:.4f}, Std: {stats['std_cpo']:.4f}, Tot pattern: {stats['total_patterns']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml1m", help="Dataset (ml1m o lfm1m)")
    parser.add_argument("--model", type=str, default="pgpr", help="Modello (pgpr, cfkg, transe)")
    args = parser.parse_args()

    test_cpo(args.dataset, args.model)

