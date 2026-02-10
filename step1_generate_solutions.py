"""
Step 1: 用 QwQ-32B 为 MathFusionQA (MATH seed) 重新生成长 CoT solution
严格对齐 MathMixup 论文 Appendix C.1 参数配置

支持分片并行: python step1.py --shard 0 --total_shards 3
"""
import os
import json
import argparse
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

QWQ_MODEL_PATH = ""
DATASET_DIR = ""
OUTPUT_DIR = "./mathfusionqa_math_qwq_results"

SAMPLING_PARAMS = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    top_k=40,
    min_p=0.0,
    repetition_penalty=1.0,
    max_tokens=32768,
    stop=["<|im_end|>", "<|endoftext|>"],
    stop_token_ids=[151645, 151643],
)

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
# =====================================================


def load_all_parquets(dataset_dir):
    all_data = []
    for fname in sorted(os.listdir(dataset_dir)):
        if not fname.endswith(".parquet"):
            continue
        fpath = os.path.join(dataset_dir, fname)
        df = pd.read_parquet(fpath)
        subset_name = fname.replace("-00000-of-00001.parquet", "")
        for _, row in df.iterrows():
            item = row.to_dict()
            item["_subset"] = subset_name
            all_data.append(item)
    print(f"Loaded {len(all_data)} samples from {dataset_dir}")
    return all_data


def get_question_field(item):
    for key in ["query", "question", "problem", "instruction", "input"]:
        if key in item and item[key]:
            return str(item[key])
    raise ValueError(f"Cannot find question field in: {list(item.keys())}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--total_shards", type=int, default=1)
    parser.add_argument("--tp", type=int, default=2, help="tensor_parallel_size")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_all_parquets(DATASET_DIR)
    shard_size = len(data) // args.total_shards
    start = args.shard * shard_size
    end = len(data) if args.shard == args.total_shards - 1 else start + shard_size
    data = data[start:end]
    print(f"Shard {args.shard}/{args.total_shards}: samples [{start}, {end}), count={len(data)}")

    tokenizer = AutoTokenizer.from_pretrained(QWQ_MODEL_PATH, trust_remote_code=True)
    prompts = []
    for item in data:
        question = get_question_field(item)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)
    print(f"Built {len(prompts)} prompts")
    llm = LLM(
        model=QWQ_MODEL_PATH,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        max_model_len=32768,
    )

    print(f"Generating solutions with QwQ-32B (shard {args.shard})...")
    outputs = llm.generate(prompts, SAMPLING_PARAMS)

    results = []
    no_boxed_count = 0
    for item, output in zip(data, outputs):
        solution = output.outputs[0].text
        question = get_question_field(item)
        has_boxed = "\\boxed{" in solution and "\\boxed{}" not in solution
        if not has_boxed:
            no_boxed_count += 1

        result = {
            "instruction": question,
            "output": solution,
            "_subset": item.get("_subset", ""),
            "_has_boxed": has_boxed,
        }
        for key in ["answer", "solution", "response"]:
            if key in item:
                result[f"_original_{key}"] = str(item[key])
        results.append(result)

    print(f"\nShard {args.shard} complete! Total: {len(results)}, "
          f"Missing boxed: {no_boxed_count} ({no_boxed_count/len(results)*100:.1f}%)")

    shard_path = os.path.join(OUTPUT_DIR, f"shard_{args.shard}.json")
    with open(shard_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved shard to {shard_path}")


if __name__ == "__main__":
    main()
