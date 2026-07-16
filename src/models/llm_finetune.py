"""QLoRA fine-tuned LLM baseline: Qwen2.5-7B-Instruct adapted to Bangla
hate speech classification with LoRA (peft). ``--unsloth`` switches to
Unsloth's fused kernels on CUDA (~2x faster + less VRAM on a T4); the
plain peft path keeps the script CPU-smokeable and portable.

Training: instruction SFT on a stratified subset of BD-SHS train
(prompt -> "হ্যাঁ"/"না", prompt tokens masked out of the loss).
Evaluation: same zero-shot prompt + subsets + prediction-dump format as
src.models.llm_baseline, so it lands in stats_summary.csv like any model
(model id qwen2.5-7b-qlora, variant "finetuned").

Usage:
    python -m src.models.llm_finetune --unsloth              # Colab T4
    python -m src.models.llm_finetune --smoke --no-quant \
        --model-name trl-internal-testing/tiny-Qwen2ForCausalLM-2.5
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from src.evaluation.evaluate import PREDICTIONS_DIR, load_test_set
from src.evaluation.metrics import record_results
from src.models.llm_baseline import SYSTEM, parse_answer
from src.utils.common import (
    DATA_PROCESSED,
    MODELS_DIR,
    ensure_dirs,
    save_json,
    set_seed,
    setup_utf8_stdout,
)

ANSWERS = {0: "না", 1: "হ্যাঁ"}


def build_prompt(tokenizer, text: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"মন্তব্য: {text}\nঘৃণাত্মক?"}]
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
    return f"{SYSTEM}\nমন্তব্য: {text}\nঘৃণাত্মক? উত্তর:"


class SFTDataset(torch.utils.data.Dataset):
    """Prompt->answer pairs; prompt tokens masked with -100."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.items = []
        for _, row in df.iterrows():
            prompt = build_prompt(tokenizer, str(row["text"])[:1000])
            answer = ANSWERS[int(row["label"])] + (tokenizer.eos_token or "")
            p_ids = tokenizer(prompt, add_special_tokens=False,
                              truncation=True, max_length=max_len - 8)
            a_ids = tokenizer(answer, add_special_tokens=False)
            input_ids = p_ids["input_ids"] + a_ids["input_ids"]
            labels = [-100] * len(p_ids["input_ids"]) + a_ids["input_ids"]
            self.items.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class SFTCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        ids = torch.full((len(feats), n), self.pad_id, dtype=torch.long)
        labels = torch.full((len(feats), n), -100, dtype=torch.long)
        mask = torch.zeros((len(feats), n), dtype=torch.long)
        for i, f in enumerate(feats):
            k = len(f["input_ids"])
            ids[i, :k] = torch.tensor(f["input_ids"])
            labels[i, :k] = torch.tensor(f["labels"])
            mask[i, :k] = 1
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def load_model(args):
    if args.unsloth:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            args.model_name, max_seq_length=args.max_len,
            load_in_4bit=not args.no_quant)
        model = FastLanguageModel.get_peft_model(
            model, r=16, lora_alpha=16, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            random_state=args.seed)
        return model, tokenizer
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                              padding_side="left")
    kwargs = {}
    if not args.no_quant:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        kwargs["device_map"] = "auto"
    else:
        kwargs["dtype"] = (torch.float32
                           if not torch.cuda.is_available()
                           else torch.float16)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **kwargs)
    if args.no_quant and torch.cuda.is_available():
        model = model.to("cuda")
    lora = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0,
                      task_type="CAUSAL_LM",
                      target_modules="all-linear")
    return get_peft_model(model, lora), tokenizer


def generate_labels(model, tokenizer, texts, batch_size, max_len):
    tokenizer.padding_side = "left"
    preds = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            prompts = [build_prompt(tokenizer, t)
                       for t in texts[i:i + batch_size]]
            enc = tokenizer(prompts, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model.generate(
                **enc, max_new_tokens=6, do_sample=False,
                pad_token_id=tokenizer.pad_token_id
                or tokenizer.eos_token_id)
            gen = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                         skip_special_tokens=True)
            preds.extend(parse_answer(g) for g in gen)
    return np.array(preds)


def main() -> None:
    setup_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--unsloth", action="store_true")
    parser.add_argument("--no-quant", action="store_true")
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--subset-size", type=int, default=1500)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)

    tag = "smoke" if args.smoke else "full"
    if args.smoke:
        args.train_size, args.subset_size, args.epochs = 64, 24, 1
        args.max_len = 256
    model_id = ("qwen2.5-7b-qlora" if "7B" in args.model_name
                else args.model_name.split("/")[-1].lower() + "-qlora")
    if args.smoke:
        model_id += "_smoke"

    print(f"loading {args.model_name} "
          f"(unsloth={args.unsloth}, quant={not args.no_quant})")
    model, tokenizer = load_model(args)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train = pd.read_csv(DATA_PROCESSED / "train.csv")
    sub = (train.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(n=min(len(g), args.train_size // 2),
                                          random_state=args.seed))
                .sample(frac=1, random_state=args.seed))
    train_ds = SFTDataset(sub, tokenizer, args.max_len)
    print(f"SFT on {len(train_ds)} examples, {args.epochs} epoch(s)")

    from transformers import Trainer, TrainingArguments

    out_dir = MODELS_DIR / f"{model_id}_finetuned_s{args.seed}"
    ensure_dirs(out_dir)
    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="no",
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      data_collator=SFTCollator(tokenizer),
                      processing_class=tokenizer)
    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0
    print(f"trained in {train_secs / 60:.1f} min")

    if args.unsloth:
        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(model)
    model.eval()

    ensure_dirs(PREDICTIONS_DIR)
    for test_name in ("clean", "bidwesh_heldout"):
        df = load_test_set(test_name)
        sub = (df.groupby("label", group_keys=False)
                 .apply(lambda g: g.sample(
                     n=min(len(g), args.subset_size // 2),
                     random_state=args.seed))).sort_index()
        print(f"[{test_name}] classifying {len(sub)} rows")
        preds = generate_labels(model, tokenizer,
                                sub["text"].fillna("").str.slice(0, 1000)
                                .tolist(), args.batch_size * 2, args.max_len)
        record_results(model_id, "finetuned", f"{test_name}_subset",
                       sub["label"].values, preds, tag=tag, seed=args.seed,
                       make_figure=False)
        pd.DataFrame({"row_id": sub.index.values,
                      "y_true": sub["label"].values,
                      "y_pred": preds}).to_csv(
            PREDICTIONS_DIR /
            f"{model_id}_finetuned_s{args.seed}_{test_name}_subset.csv",
            index=False)

    model.save_pretrained(str(out_dir))  # LoRA adapter only (small)
    tokenizer.save_pretrained(str(out_dir))
    save_json({"model_id": model_id, "model_type": "hf-lora-causal",
               "model_name": args.model_name, "train_variant": "finetuned",
               "input_column": "text", "tag": tag, "seed": args.seed,
               "train_rows": len(train_ds),
               "train_seconds": round(train_secs, 1),
               "unsloth": args.unsloth, "lr": args.lr,
               "epochs": args.epochs},
              out_dir / "meta.json")
    print(f"saved LoRA adapter to {out_dir}")


if __name__ == "__main__":
    main()
