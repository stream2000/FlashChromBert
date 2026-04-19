"""Evaluate a FlashChromBert Lightning MLM checkpoint on promoter / crm / genome
held-out sets, plus three baselines.

Also writes a deterministic masked fixture per dataset so the legacy
ChromBERT runner can evaluate on identical inputs for direct comparison.

Usage (from repo root, after `./activate.sh`):
    python scripts/eval_mlm.py \
        --ckpt checkpoints/ch_promoter_tuned/epoch=0-val_loss=0.257.ckpt \
        --out docs/runs/2026-04-18_validation_fcb.json \
        --fixture-dir fixtures/mlm_eval/2026-04-18
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from flashchrombert.data import KmerCStateTokenizer
from flashchrombert.eval import (
    build_fixture,
    compute_baselines,
    evaluate_model,
    load_fixture,
    save_fixture,
)
from flashchrombert.model import BertConfig, BertForMaskedLM


DATASETS = [
    ("promoter_val", "data/ch_temp/promoter_pretrain_data/val_split.txt"),
    ("crm_ood", "data/ch_temp/crm_pretrain_data/crm_lim10_allcell_4merized.txt"),
    ("genome_ood", "data/ch_temp/genome_pretrain_data/pretrain_genome_all.txt"),
]


def load_fcb_model(ckpt_path: str, vocab_size: int, device: str) -> BertForMaskedLM:
    """Rehydrate BertForMaskedLM from a Lightning ckpt (strip 'model.' prefix)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = BertConfig(
        vocab_size=vocab_size,
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=1536,
        max_position_embeddings=512,
    )
    model = BertForMaskedLM(cfg)
    state = {
        k[len("model."):]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] missing={missing[:3]}... unexpected={unexpected[:3]}...")
    model.eval().to(device)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="JSON results path")
    ap.add_argument("--fixture-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260418)
    ap.add_argument("--mlm-prob", type=float, default=0.0375)
    ap.add_argument("--reuse-fixtures", action="store_true",
                    help="Load existing fixture .pt files instead of rebuilding.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}")

    tokenizer = KmerCStateTokenizer(k=4, num_states=15)
    model = load_fcb_model(args.ckpt, tokenizer.vocab_size, device)

    fixture_dir = Path(args.fixture_dir)
    fixture_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "ckpt": args.ckpt,
        "seed": args.seed,
        "mlm_probability": args.mlm_prob,
        "max_samples": args.max_samples,
        "datasets": {},
    }

    def fwd(input_ids, attention_mask):
        return model(input_ids, attention_mask=attention_mask).logits

    for name, path in DATASETS:
        fix_path = fixture_dir / f"{name}.pt"
        if args.reuse_fixtures and fix_path.exists():
            print(f"[info] loading cached fixture {fix_path}")
            fix = load_fixture(fix_path)
        else:
            t0 = time.time()
            fix = build_fixture(
                data_path=path,
                dataset_name=name,
                tokenizer=tokenizer,
                max_length=512,
                max_samples=args.max_samples,
                mlm_probability=args.mlm_prob,
                seed=args.seed,
            )
            save_fixture(fix, fix_path)
            print(f"[info] built {name}: {fix.input_ids.shape} "
                  f"n_masked≈{(fix.labels != -100).sum().item()} "
                  f"({time.time() - t0:.1f}s) → {fix_path}")

        t0 = time.time()
        model_metrics = evaluate_model(
            fix, fwd, batch_size=args.batch_size, device=device
        )
        print(f"[info] fcb on {name}: {model_metrics} ({time.time() - t0:.1f}s)")

        baselines = compute_baselines(fix)
        results["datasets"][name] = {
            "n_samples": int(fix.input_ids.size(0)),
            "seq_len": int(fix.input_ids.size(1)),
            "fcb_tuned": model_metrics,
            **baselines,
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[info] wrote {args.out}")


if __name__ == "__main__":
    main()
