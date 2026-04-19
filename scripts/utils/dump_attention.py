"""Dump last-layer attention from a fine-tuned LitBertFinetune checkpoint.

Produces the two files that `flashchrombert.legacy.find_motifs` expects:
  - atten.npy        float32 [N, max_length]
  - pred_results.npy float32 [N, num_labels]

Per-sample score reduction mirrors ChromBERT/run_finetune.py:visualize()
lines 881-904 exactly:
  1) last-layer attention [H, L, L] → sum over heads → take [CLS] row
  2) drop CLS column, keep positions i in [1, L-kmer+1)
  3) tail-trim: first position where next score == 0 gets zeroed
  4) broadcast each k-mer score to its k base positions (average by coverage)
  5) L2-normalize per sample

Usage:
    python scripts/dump_attention.py \
        --ckpt checkpoints/ft_promoter_cls/epoch=3-val_auc=0.7603.ckpt \
        --dev-tsv data/ch_temp/promoter_finetune_data/classification/rpkm0_n_rpkm50/dev.tsv \
        --out-dir logs/motif/ft_cls_pretrained \
        --task classification
"""

from __future__ import annotations

import argparse
import os
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from flashchrombert.data.finetune import SeqLabelDataset
from flashchrombert.data.tokenizer import KmerCStateTokenizer
from flashchrombert.model import BertConfig, BertForSequenceClassification


@torch.no_grad()
def reduce_attention(attn_layer: torch.Tensor, kmer: int, max_len: int) -> np.ndarray:
    """Mirror ChromBERT visualize() per-sample reduction."""
    L = attn_layer.shape[-1]
    head_sum = attn_layer.float().sum(dim=0)
    cls_row = head_sum[0]
    upper = L - kmer + 1
    attn_score = cls_row[1:upper].cpu().numpy().astype(np.float64)

    for i in range(len(attn_score) - 1):
        if attn_score[i + 1] == 0:
            attn_score[i] = 0
            break

    n_base = len(attn_score) + kmer - 1
    counts = np.zeros(n_base, dtype=np.float64)
    real = np.zeros(n_base, dtype=np.float64)
    for i, s in enumerate(attn_score):
        for j in range(kmer):
            counts[i + j] += 1.0
            real[i + j] += s
    real = real / counts
    norm = np.linalg.norm(real)
    if norm > 0:
        real = real / norm

    out = np.zeros(max_len, dtype=np.float32)
    out[: min(len(real), max_len)] = real[: max_len]
    return out


def collate_fixed(batch, pad_id: int, target_len: int, is_regression: bool):
    input_ids = torch.full((len(batch), target_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), target_len), dtype=torch.long)
    dtype = torch.float if is_regression else torch.long
    labels = torch.tensor([y for _, y in batch], dtype=dtype)
    for i, (seq, _) in enumerate(batch):
        n = min(len(seq), target_len)
        input_ids[i, :n] = seq[:n]
        attn[i, :n] = 1
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dev-tsv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--task", default="classification", choices=["classification", "regression"])
    ap.add_argument("--kmer", type=int, default=4)
    ap.add_argument("--num-states", type=int, default=15)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--hidden-size", type=int, default=384)
    ap.add_argument("--num-hidden-layers", type=int, default=12)
    ap.add_argument("--num-attention-heads", type=int, default=12)
    ap.add_argument("--intermediate-size", type=int, default=1536)
    ap.add_argument("--classifier-dropout", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=-1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    is_regression = args.task == "regression"
    num_labels = 1 if is_regression else 2

    tokenizer = KmerCStateTokenizer(k=args.kmer, num_states=args.num_states)

    ds = SeqLabelDataset(
        args.dev_tsv,
        tokenizer,
        task=args.task,
        max_length=args.max_length,
        has_header=True,
        max_samples=args.max_samples if args.max_samples > 0 else None,
    )
    
    N = len(ds)
    print(f"[dump_attention] dev samples: {N}, task: {args.task}")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(
            collate_fixed,
            pad_id=tokenizer.pad_token_id,
            target_len=args.max_length,
            is_regression=is_regression,
        ),
    )

    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_length,
        hidden_dropout=0.1,
        attention_dropout=0.1,
    )
    model = BertForSequenceClassification(
        config, num_labels=num_labels, classifier_dropout=args.classifier_dropout
    )

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    raw = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    state = {k[len("model."):]: v for k, v in raw.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[dump_attention] ckpt load: missing={len(missing)} unexpected={len(unexpected)}")

    model.eval().to(args.device)

    atten_scores = np.zeros((N, args.max_length), dtype=np.float32)
    preds = np.zeros((N, num_labels), dtype=np.float32)

    idx = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(args.device)
            attn_mask = batch["attention_mask"].to(args.device)
            out = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                return_attn=True,
            )
            last = out.attentions[-1]
            B = last.shape[0]
            
            if is_regression:
                probs = out.logits.float().cpu().numpy()
            else:
                probs = torch.softmax(out.logits.float(), dim=-1).cpu().numpy()
                
            preds[idx:idx + B] = probs
            for b in range(B):
                atten_scores[idx + b] = reduce_attention(
                    last[b], args.kmer, args.max_length
                )
            idx += B
            if idx % (args.batch_size * 10) == 0 or idx == N:
                print(f"  processed {idx}/{N}")

    np.save(os.path.join(args.out_dir, "atten.npy"), atten_scores)
    np.save(os.path.join(args.out_dir, "pred_results.npy"), preds)
    print(
        f"[dump_attention] wrote {args.out_dir}/atten.npy {atten_scores.shape} "
        f"+ pred_results.npy {preds.shape}"
    )


if __name__ == "__main__":
    main()
