from creativity_metrics.creative_path_originality_cfkg import CreativePathOriginalityCFKG
from creativity_metrics.creative_path_originality_transe import CreativePathOriginalityTransE

print("=== TEST CPO per CFKG (lfm1m) ===")
cfkg_metric = CreativePathOriginalityCFKG('lfm1m')
cfkg_scores = cfkg_metric.calculate_cpo_all_users(max_users=10)
print("CFKG CPO (media su 10 utenti):", sum(cfkg_scores.values()) / len(cfkg_scores))

print("\n=== TEST CPO per TransE (ml1m) ===")
transe_metric = CreativePathOriginalityTransE('ml1m')
transe_scores = transe_metric.calculate_cpo_all_users(max_users=10)
print("TransE CPO (media su 10 utenti):", sum(transe_scores.values()) / len(transe_scores))

