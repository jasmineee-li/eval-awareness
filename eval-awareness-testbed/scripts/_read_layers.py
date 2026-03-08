import json, glob, os
for f in sorted(glob.glob("probes/trained/*/main/probe_info.json")):
    model_dir = os.path.basename(os.path.dirname(os.path.dirname(f)))
    d = json.load(open(f))
    print(f"{model_dir:<30} best_layer={d['best_layer']}  num_layers={d['num_layers']}  model={d.get('model','?')}")
