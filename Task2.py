from __future__ import annotations
import json, os, gc, psutil
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import nltk
from nltk.translate.meteor_score import single_meteor_score

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    Seq2SeqTrainingArguments,
)

PATHS = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test": "test.jsonl",
    "prediction_csv": "prediction_task2.csv",
    "submission_csv": "submission.csv",
    "output_dir": "runs/t5-base",
}

HPARAMS = dict(
    epochs=1,
    batch=2,
    lr=5e-5,
    seed=42,
    weight_decay=0.01,
)

LENGTHS = dict(
    max_input_len=512,
    max_target_len=32,
)

MODEL_NAME = "t5-base"


def read_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_pair(entry: Dict):
    post = entry.get("postText", [""])[0]
    ctx = " ".join(entry.get("targetParagraphs", []))
    spoiler = None
    if entry.get("humanSpoiler"):
        spoiler = entry["humanSpoiler"][0]
    elif entry.get("spoiler"):
        spoiler = entry["spoiler"][0]
    if spoiler is None:
        return None
    return {"input": f"post: {post} context: {ctx}", "spoiler": spoiler}


def load_test_spoiler_types(path: str) -> dict:
    if not os.path.exists(path):
        print(f"[load_test_spoiler_types] Not found: {path}.")
        return {}
    df = pd.read_csv(path)
    known = {"phrase": "phrase", "passage": "passage", "multi": "multi"}
    out = {}
    for _id, st in zip(df["id"], df["spoilerType"]):
        if isinstance(st, str):
            out[str(_id)] = known.get(st.strip().lower(), "unknown")
        else:
            out[str(_id)] = "unknown"
    return out


def make_df(path: str, *, test: bool = False, test_types: dict | None = None) -> pd.DataFrame:
    raw = read_jsonl(path)
    rows = []
    for entry in raw:
        uid = entry.get("uuid")
        post = entry.get("postText", "")
        title = entry.get("targetTitle", "")
        context = " ".join(entry.get("targetParagraphs", []))

        if test:
            stype = (test_types or {}).get(str(uid), "unknown")
            inp = f"type: {stype} | target: {post} title: {title} context: {context}"
            rows.append({"id": str(uid), "input": inp, "type": stype})
        else:
            tag_list = entry.get("tags") or ["unknown"]
            stype = tag_list[0] if isinstance(tag_list, list) and tag_list else "unknown"
            pair = build_pair(entry)
            if pair:
                inp = f"type: {stype} | target: {post} title: {title} context: {context}"
                rows.append({"input": inp, "spoiler": pair["spoiler"], "type": stype})
    return pd.DataFrame(rows)


def tokenize_data(batch, tokenizer, max_in: int, max_out: int, *, test: bool = False):
    out = tokenizer(
        batch["input"],
        max_length=max_in,
        truncation=True,
        padding="max_length",
    )
    if not test:
        labels = tokenizer(
            text_target=batch["spoiler"],
            max_length=max_out,
            truncation=True,
            padding="max_length",
        )
        out["labels"] = labels["input_ids"]
    return out


def compute_meteor(eval_pred, tokenizer):
    pred_ids, label_ids = eval_pred
    predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    pad_id = tokenizer.pad_token_id
    label_ids = np.where(label_ids != -100, label_ids, pad_id)
    references = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    scores = [single_meteor_score(reference.split(), prediction.split()) for reference, prediction in
              zip(references, predictions)]
    avg = float(np.mean(scores)) if scores else 0.0
    print(f"METEOR={avg}")
    return {"meteor": avg}


def max_tokens_for_type(stype: str, default_max: int) -> int:
    st = (stype or "").lower()
    if st == "phrase":
        return min(16, default_max)
    if st == "passage":
        return max(64, default_max)
    if st == "multi":
        return max(48, default_max)
    return default_max


class NoSaveTrainer(Trainer):
    # override save_model function
    def save_model(self, *args, **kwargs):
        print("save_model skipped")


def run_train():
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    train_df = make_df(PATHS["train"], test=False)
    val_df = make_df(PATHS["val"], test=False)
    print(f"Loaded data| train={len(train_df)} val={len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    print(f"Model + tokenizer ready")

    train_dataset = Dataset.from_pandas(train_df).map(
        lambda batch: tokenize_data(batch, tokenizer, LENGTHS["max_input_len"], LENGTHS["max_target_len"], test=False),
        batched=True,
        remove_columns=list(train_df.columns),
    )
    val_dataset = Dataset.from_pandas(val_df).map(
        lambda batch: tokenize_data(batch, tokenizer, LENGTHS["max_input_len"], LENGTHS["max_target_len"], test=False),
        batched=True,
        remove_columns=list(val_df.columns),
    )
    print(f"Tokenization complete")

    args = Seq2SeqTrainingArguments(
        output_dir=PATHS["output_dir"],
        save_strategy="no",
        learning_rate=HPARAMS["lr"],
        per_device_train_batch_size=HPARAMS["batch"],
        per_device_eval_batch_size=1,
        num_train_epochs=HPARAMS["epochs"],
        weight_decay=HPARAMS["weight_decay"],
        predict_with_generate=True,
        eval_accumulation_steps=4,
        generation_max_length=LENGTHS["max_target_len"],
        logging_steps=100,
        report_to="none",
        seed=HPARAMS["seed"],
        load_best_model_at_end=False,
    )

    ompute_metrics_fn = lambda eval_pred: compute_meteor(eval_pred, tokenizer)

    trainer = NoSaveTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=ompute_metrics_fn,
    )

    print(f"Starting training")
    trainer.train()
    print(f"Training complete")

    print("Evaluating on validation")
    val_metrics = trainer.evaluate(metric_key_prefix="val")
    print(f"Validation METEOR: {val_metrics.get('val_meteor', 0.0)}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model, tokenizer


def run_predict(model, tokenizer):
    test_types = load_test_spoiler_types(PATHS["submission_csv"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    test_df = make_df(PATHS["test"], test=True, test_types=test_types)
    print(f"Generating {len(test_df)} test predictions")

    preds = []
    for id, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test predict"):
        inputs = tokenizer(
            row["input"], return_tensors="pt",
            truncation=True, max_length=LENGTHS["max_input_len"]
        ).to(device)
        mtok = max_tokens_for_type(row.get("type"), LENGTHS["max_target_len"])
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                num_beams=1,
                do_sample=False,
                max_new_tokens=mtok
            )
        spoiler = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        preds.append({"id": row["id"], "spoiler": spoiler})

    out_df = pd.DataFrame(preds)
    out_df.to_csv(PATHS["prediction_csv"], index=False)
    print(f"Saved {PATHS['prediction_csv']}")


if __name__ == "__main__":
    model, tokenizer = run_train()
    run_predict(model, tokenizer)

