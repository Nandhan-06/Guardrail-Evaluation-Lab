import json
for r in json.load(open("policy_layer_live_results.json", encoding="utf-8")):
    print(f"\n[{r['layer']} | {r['label']} | {r['config']}]")
    print("PROMPT:  ", r["prompt"])
    print("EXPECTED:", r["expected_behaviour"])
    print("RESPONSE:", r["model_response"])
    print("-" * 70)
