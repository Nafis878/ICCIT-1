"""NCT — Noise-Consistency Training.

Each training example is paired with a synthetically-noised view
(``src.data.augment.augment_text``); the loss is

    L = 0.5*(CE(clean) + CE(noisy)) + lambda * symmetric-KL(p_clean, p_noisy)

which pushes the model to give consistent predictions under Bangla
social-media noise instead of merely seeing noisy copies as extra data.
Noisy views are sampled once per run (seeded), so different seeds see
different views. Evaluation batches (no paired keys) fall back to plain
cross-entropy, so the standard Trainer eval loop works unchanged.
"""
import random

import torch
import torch.nn.functional as F
from transformers import Trainer

from src.data.augment import augment_text


class PairedTextDataset(torch.utils.data.Dataset):
    """(clean, noisy) views of each row, pre-tokenized."""

    def __init__(self, df, tokenizer, max_len: int, seed: int,
                 text_column: str = "text"):
        rng = random.Random(seed)
        texts = df[text_column].fillna("").tolist()
        noisy = [augment_text(t, rng)[0] for t in texts]
        self.enc = tokenizer(texts, truncation=True, max_length=max_len)
        self.enc_aug = tokenizer(noisy, truncation=True, max_length=max_len)
        self.labels = df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": self.enc["input_ids"][i],
            "attention_mask": self.enc["attention_mask"][i],
            "aug_input_ids": self.enc_aug["input_ids"][i],
            "aug_attention_mask": self.enc_aug["attention_mask"][i],
            "labels": self.labels[i],
        }


class PairedCollator:
    """Pads the clean and noisy views independently."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        clean = [{"input_ids": f["input_ids"],
                  "attention_mask": f["attention_mask"]} for f in features]
        batch = self.tokenizer.pad(clean, return_tensors="pt")
        if "aug_input_ids" in features[0]:  # eval batches have no pair
            noisy = [{"input_ids": f["aug_input_ids"],
                      "attention_mask": f["aug_attention_mask"]}
                     for f in features]
            noisy_batch = self.tokenizer.pad(noisy, return_tensors="pt")
            batch["aug_input_ids"] = noisy_batch["input_ids"]
            batch["aug_attention_mask"] = noisy_batch["attention_mask"]
        batch["labels"] = torch.tensor([f["labels"] for f in features],
                                       dtype=torch.long)
        return batch


class ConsistencyTrainer(Trainer):
    def __init__(self, *args, consistency_lambda: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.consistency_lambda = consistency_lambda

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        aug_ids = inputs.pop("aug_input_ids", None)
        aug_mask = inputs.pop("aug_attention_mask", None)
        outputs = model(**inputs)
        if aug_ids is None:  # eval batches have no paired view
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        out_aug = model(input_ids=aug_ids, attention_mask=aug_mask,
                        labels=inputs["labels"])
        log_p = F.log_softmax(outputs.logits.float(), dim=-1)
        log_q = F.log_softmax(out_aug.logits.float(), dim=-1)
        kl = 0.5 * (
            F.kl_div(log_p, log_q.exp(), reduction="batchmean")
            + F.kl_div(log_q, log_p.exp(), reduction="batchmean"))
        loss = 0.5 * (outputs.loss + out_aug.loss) \
            + self.consistency_lambda * kl
        return (loss, outputs) if return_outputs else loss
