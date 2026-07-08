"""Few-shot LLM baseline: Qwen2.5-7B-Instruct (ungated, multilingual),
4-bit quantized — fits a Colab T4. Classifies a stratified subset of each
test set with a fixed 5-shot Bangla prompt and writes standard prediction
dumps + results rows, so it appears in stats_summary.csv like any model.

The subset row-ids are saved so identical rows can be compared across
models (see stats.py). On CPU or without bitsandbytes, pass a smaller
--model-name (e.g. Qwen/Qwen2.5-0.5B-Instruct) and --no-quant for smoke.

Usage:
    python -m src.models.llm_baseline                       # full (GPU)
    python -m src.models.llm_baseline --smoke --no-quant \
        --model-name Qwen/Qwen2.5-0.5B-Instruct             # CPU check
"""
import argparse
import re

import numpy as np
import pandas as pd
import torch

from src.evaluation.evaluate import (
    PREDICTIONS_DIR,
    load_test_set,
)
from src.evaluation.metrics import record_results
from src.utils.common import ensure_dirs, set_seed, setup_utf8_stdout

# 5 fixed few-shot examples (composed for the prompt — not from any test set).
FEW_SHOTS = [
    ("খেলাটা আজ দারুণ হয়েছে, সবাই ভালো খেলেছে", "না"),
    ("তুই একটা কুত্তার বাচ্চা, তোরে পিটাইয়া মারমু", "হ্যাঁ"),
    ("ভাই ভিডিওটা অনেক সুন্দর হইছে, চালিয়ে যান", "না"),
    ("এই মাগীরে ধইরা জুতা দিয়া পিটানো দরকার", "হ্যাঁ"),
    ("ওর গানগুলা আমার তেমন ভালো লাগে না", "না"),
]

SYSTEM = ("তুমি একজন বাংলা কনটেন্ট মডারেটর। নিচের মন্তব্যটি ঘৃণাত্মক বক্তব্য "
          "(hate speech) কি না নির্ধারণ করো। ঘৃণাত্মক বক্তব্য = কোনো ব্যক্তি বা "
          "গোষ্ঠীকে আক্রমণ, অপমান, হুমকি বা সহিংসতার আহ্বান। শুধু 'হ্যাঁ' বা "
          "'না' লিখে উত্তর দাও।")


def build_messages(text: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}]
    for ex, ans in FEW_SHOTS:
        msgs.append({"role": "user", "content": f"মন্তব্য: {ex}\nঘৃণাত্মক?"})
        msgs.append({"role": "assistant", "content": ans})
    msgs.append({"role": "user", "content": f"মন্তব্য: {text}\nঘৃণাত্মক?"})
    return msgs


def parse_answer(generated: str) -> int:
    g = generated.strip()
    if re.search(r"হ্যাঁ|হ্যা|yes", g, re.IGNORECASE):
        return 1
    return 0


def main() -> None:
    setup_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--subset-size", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-quant", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = "smoke" if args.smoke else "full"
    subset_size = 40 if args.smoke else args.subset_size
    model_id = ("qwen2.5-7b-fewshot" if "7B" in args.model_name
                else args.model_name.split("/")[-1].lower() + "-fewshot")
    if args.smoke:
        model_id += "_smoke"

    print(f"loading {args.model_name} (quant={not args.no_quant}) "
          f"on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                              padding_side="left")
    kwargs = {}
    if not args.no_quant:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        kwargs["device_map"] = "auto"
    else:
        kwargs["dtype"] = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **kwargs)
    if args.no_quant:
        model = model.to(device)
    model.eval()

    ensure_dirs(PREDICTIONS_DIR)
    for test_name in ("clean", "bidwesh_heldout"):
        df = load_test_set(test_name)
        sub = (df.groupby("label", group_keys=False)
                 .apply(lambda g: g.sample(
                     n=min(len(g), subset_size // 2),
                     random_state=args.seed))).sort_index()
        print(f"[{test_name}] classifying {len(sub)} rows")

        texts = sub["text"].fillna("").str.slice(0, 1000).tolist()
        preds = []
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i:i + args.batch_size]
            prompts = [tokenizer.apply_chat_template(
                build_messages(t), tokenize=False,
                add_generation_prompt=True) for t in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True,
                            truncation=True, max_length=2048)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=8,
                                     do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id
                                     or tokenizer.eos_token_id)
            gen = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                         skip_special_tokens=True)
            preds.extend(parse_answer(g) for g in gen)
            if (i // args.batch_size) % 5 == 0:
                print(f"  {i + len(batch)}/{len(texts)}")

        preds = np.array(preds)
        record_results(model_id, "fewshot", f"{test_name}_subset",
                       sub["label"].values, preds, tag=tag, seed=args.seed,
                       make_figure=False)
        dump = pd.DataFrame({"row_id": sub.index.values,
                             "y_true": sub["label"].values,
                             "y_pred": preds})
        dump.to_csv(PREDICTIONS_DIR /
                    f"{model_id}_fewshot_s{args.seed}_{test_name}_subset.csv",
                    index=False)
    print("done")


if __name__ == "__main__":
    main()
