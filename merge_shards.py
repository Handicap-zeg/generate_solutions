"""合并所有 shard 结果，生成最终训练数据"""
import os
import json

OUTPUT_DIR = ""

all_results = []
for fname in sorted(os.listdir(OUTPUT_DIR)):
    if fname.startswith("shard_") and fname.endswith(".json"):
        fpath = os.path.join(OUTPUT_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            shard = json.load(f)
        print(f"{fname}: {len(shard)} samples")
        all_results.extend(shard)

print(f"\nTotal: {len(all_results)}")

no_boxed = sum(1 for r in all_results if not r["_has_boxed"])
print(f"Missing boxed: {no_boxed} ({no_boxed/len(all_results)*100:.1f}%)")

full_path = os.path.join(OUTPUT_DIR, "mathfusion_math_qwq32b_solutions.json")
with open(full_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"Full results -> {full_path}")

alpaca_data = []
for r in all_results:
    if r["_has_boxed"]:
        alpaca_data.append({
            "instruction": r["instruction"],
            "input": "",
            "output": r["output"],
        })

alpaca_path = os.path.join(OUTPUT_DIR, "train_alpaca.json")
with open(alpaca_path, "w", encoding="utf-8") as f:
    json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
print(f"Alpaca format: {len(alpaca_data)} samples -> {alpaca_path}")
