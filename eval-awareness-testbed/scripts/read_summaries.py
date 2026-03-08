import json, glob, os
for f in sorted(glob.glob("/workspace/eval-awareness/eval-awareness-testbed/results/*/evaluation_summary.json")):
    model = os.path.basename(os.path.dirname(f))
    with open(f) as fh:
        d = json.load(fh)
    print(f"=== {model} === layer={d['layer']} threshold={d['threshold']:.4f} cross_auroc={d['cross_dataset_auroc']:.4f}")
    for k, v in d['datasets'].items():
        auroc = v.get('auroc', 'N/A')
        acc = v.get('accuracy', 'N/A')
        mean = v.get('mean_score', 0)
        pct = v.get('pct_classified_testing', 'N/A')
        print(f"  {k:25s}: auroc={auroc}, acc={acc}, mean={mean:.4f}, pct_test={pct}")
    print()
