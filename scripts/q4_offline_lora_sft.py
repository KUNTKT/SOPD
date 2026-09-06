#!/usr/bin/env python3
"""Q4 Step2: LoRA SFT on steered-teacher correct trajs + confirm400 holdout eval."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

SSOPD_ROOT = Path("/scratch/ktang115/SSOPD")
if str(SSOPD_ROOT) not in sys.path:
    sys.path.insert(0, str(SSOPD_ROOT))

from ssopd_math.models.math_model import HFMathModel
from ssopd_math.nxt.bootstrap import task_cluster_bootstrap
from ssopd_math.nxt.positions import build_prompt_text
from ssopd_math.verifier.reward import POSITIVE, verify_completion
import ssopd_math.models.math_model as math_model_mod
import ssopd_math.nxt.positions as positions_mod


def patch_disable_thinking() -> None:
    def build_prompt_text_no_think(tokenizer, prompt_user: str, system_prompt: str | None = None) -> str:
        messages = [
            {"role": "system", "content": system_prompt or positions_mod.DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_user},
        ]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)

    positions_mod.build_prompt_text = build_prompt_text_no_think
    math_model_mod.build_prompt_text = build_prompt_text_no_think
    # SFTDataset imports the name at module level; keep both bindings in sync.
    global build_prompt_text
    build_prompt_text = build_prompt_text_no_think


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def one_per_problem(records: list[dict], problem_ids: set[str]) -> list[dict]:
    out, seen = [], set()
    for r in records:
        pid = str(r["problem_id"])
        if pid not in problem_ids or pid in seen:
            continue
        seen.add(pid)
        out.append(r)
    return out


class SFTDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int = 3072):
        self.rows = rows
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        prompt = r.get("prompt_text") or build_prompt_text(
            self.tok, r.get("prompt_user", r["problem"])
        )
        completion = r["completion_text"]
        # Ensure completion ends cleanly for causal LM
        full = prompt + completion
        enc_full = self.tok(
            full,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc_prompt = self.tok(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc_full["input_ids"][0]
        labels = input_ids.clone()
        plen = int(enc_prompt["input_ids"].shape[1])
        labels[:plen] = -100
        attention_mask = enc_full["attention_mask"][0]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate(batch: list[dict], pad_id: int) -> dict:
    max_len = max(x["input_ids"].shape[0] for x in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    for i, x in enumerate(batch):
        n = x["input_ids"].shape[0]
        input_ids[i, :n] = x["input_ids"]
        attention_mask[i, :n] = x["attention_mask"]
        labels[i, :n] = x["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def evaluate_split(
    model: HFMathModel,
    records: list[dict],
    *,
    seed: int,
    batch_size: int,
    log,
) -> dict:
    correct = 0
    parse_ok = 0
    lengths: list[int] = []
    per: dict[str, dict] = {}
    n = len(records)
    t0 = time.time()
    for start in range(0, n, batch_size):
        chunk = records[start : start + batch_size]
        prompts = [r.get("prompt_user", r["problem"]) for r in chunk]
        golds = [r["gold_answer"] for r in chunk]
        gens = model.generate_batch(prompts, golds, seed=seed + start)
        for rec, gen, gold in zip(chunk, gens, golds):
            ver = verify_completion(gen.text, gold)
            c = int(ver["verification_status"] == POSITIVE)
            p = int(bool(ver.get("parse_ok", False)))
            correct += c
            parse_ok += p
            lengths.append(len(gen.completion_token_ids))
            per[str(rec["problem_id"])] = {
                "correct": c,
                "parse_ok": p,
                "length": len(gen.completion_token_ids),
            }
        done = min(start + batch_size, n)
        log(f"eval {done}/{n} acc={correct/max(done,1):.4f}")
    return {
        "n": n,
        "accuracy": correct / max(n, 1),
        "parse_rate": parse_ok / max(n, 1),
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "per_problem": per,
        "wall_s": time.time() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-jsonl", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-path", default="/scratch/ktang115/models/Qwen3-1.7B")
    ap.add_argument("--rollouts", default="/scratch/ktang115/SSOPD/ssopd_math/data/ssopd01_qwen3_1_7b_scale2k/rollouts.jsonl")
    ap.add_argument("--splits", default="/scratch/ktang115/SSOPD/ssopd_math/data/ssopd02_qwen3_1_7b_scale2k/splits.json")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=3072)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=48)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-base-eval", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--reuse-base-eval", default=None, help="JSON with base_eval + optional base_per_problem")
    ap.add_argument("--post-eval-seed", type=int, default=None, help="Default: same as --seed for paired compare")
    ap.add_argument("--no-thinking", action="store_true")
    args = ap.parse_args()
    if args.no_thinking:
        patch_disable_thinking()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_eval.log"
    ckpt_dir = out_dir / "lora_adapter"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    teachers = [r for r in load_jsonl(Path(args.teacher_jsonl)) if int(r.get("correct", 0)) == 1]
    log(f"loaded correct teachers n={len(teachers)}")
    if not teachers:
        raise SystemExit("no correct teacher trajectories")

    # hold out 10% of teacher problems for train val loss only (not confirm)
    pids = sorted({str(r["problem_id"]) for r in teachers})
    rng = random.Random(args.seed)
    rng.shuffle(pids)
    n_val = max(1, int(0.1 * len(pids)))
    val_pids = set(pids[:n_val])
    train_rows = [r for r in teachers if str(r["problem_id"]) not in val_pids]
    val_rows = [r for r in teachers if str(r["problem_id"]) in val_pids]
    log(f"train_rows={len(train_rows)} val_rows={len(val_rows)} val_problems={len(val_pids)}")

    splits = json.loads(Path(args.splits).read_text())
    confirm_ids = sorted(pid for pid, s in splits["problem_splits"].items() if s == "confirm")
    confirm_recs = one_per_problem(load_jsonl(Path(args.rollouts)), set(confirm_ids))
    confirm_recs = sorted(confirm_recs, key=lambda r: str(r["problem_id"]))
    log(f"confirm eval n={len(confirm_recs)}")

    results: dict = {
        "protocol": "q4_offline_lora_sft_teacher_correct",
        "n_teacher_correct": len(teachers),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "epochs": args.epochs,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "max_new_tokens": args.max_new_tokens,
        "no_thinking": bool(args.no_thinking),
    }

    # ---- base eval on confirm ----
    post_seed = int(args.post_eval_seed if args.post_eval_seed is not None else args.seed)
    if args.reuse_base_eval:
        reused = json.loads(Path(args.reuse_base_eval).read_text())
        if "base_eval" in reused:
            results["base_eval"] = reused["base_eval"]
            results["base_per_problem"] = reused.get("base_per_problem") or {}
            bpp = Path(args.reuse_base_eval).parent / "base_per_problem.json"
            if not results["base_per_problem"] and bpp.exists():
                results["base_per_problem"] = json.loads(bpp.read_text())
        else:
            results["base_eval"] = reused
            bpp = Path(args.reuse_base_eval).parent / "base_per_problem.json"
            results["base_per_problem"] = json.loads(bpp.read_text()) if bpp.exists() else {}
        (out_dir / "base_eval.json").write_text(json.dumps(results["base_eval"], indent=2))
        log(f"reused base confirm acc={results['base_eval']['accuracy']:.4f} from {args.reuse_base_eval}")
        base_eval = {"per_problem": results["base_per_problem"], "accuracy": results["base_eval"]["accuracy"]}
    elif not args.skip_base_eval:
        log("BASE eval confirm400")
        base_model = HFMathModel(
            args.model_path,
            device="cuda",
            torch_dtype="bfloat16",
            local_files_only=True,
            use_cache=True,
            max_new_tokens=args.max_new_tokens,
            temperature=1.0,
            top_p=0.95,
            do_sample=True,
        )
        base_eval = evaluate_split(
            base_model, confirm_recs, seed=args.seed, batch_size=args.eval_batch_size, log=log
        )
        results["base_eval"] = {k: base_eval[k] for k in ("n", "accuracy", "parse_rate", "mean_length", "wall_s")}
        results["base_per_problem"] = base_eval["per_problem"]
        (out_dir / "base_eval.json").write_text(json.dumps(results["base_eval"], indent=2))
        log(f"base confirm acc={base_eval['accuracy']:.4f}")
        base_model.close()
        del base_model
        torch.cuda.empty_cache()
    else:
        base_eval = {"per_problem": {}, "accuracy": 0.0}

    if not args.skip_train:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType

        log("loading model for LoRA SFT")
        tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=True,
            device_map=None,
        ).cuda()
        model.config.use_cache = False
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
        model.train()

        train_ds = SFTDataset(train_rows, tok, max_length=args.max_length)
        val_ds = SFTDataset(val_rows, tok, max_length=args.max_length)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate(b, tok.pad_token_id),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate(b, tok.pad_token_id),
        )
        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

        history = []
        best_val = float("inf")
        t_train = time.time()
        for epoch in range(args.epochs):
            model.train()
            opt.zero_grad(set_to_none=True)
            losses = []
            step = 0
            for batch in train_loader:
                batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
                out = model(**batch)
                loss = out.loss / args.grad_accum
                loss.backward()
                step += 1
                if step % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                losses.append(float(out.loss.detach().cpu()))
                if step % 50 == 0:
                    log(f"epoch={epoch} step={step} loss={np.mean(losses[-50:]):.4f}")
            if step % args.grad_accum != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            model.eval()
            vlosses = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
                    vlosses.append(float(model(**batch).loss.cpu()))
            train_loss = float(np.mean(losses)) if losses else 0.0
            val_loss = float(np.mean(vlosses)) if vlosses else 0.0
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            log(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                tok.save_pretrained(ckpt_dir)
                log(f"saved best adapter -> {ckpt_dir}")

        results["train"] = {
            "history": history,
            "best_val_loss": best_val,
            "wall_s": time.time() - t_train,
        }
        del model
        torch.cuda.empty_cache()
    else:
        log("skip train; expecting existing adapter")

    # ---- post eval: load base + adapter ----
    log("POST eval confirm400 with LoRA")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
    ).cuda()
    if not ckpt_dir.exists():
        raise SystemExit(f"missing adapter at {ckpt_dir}")
    peft_model = PeftModel.from_pretrained(base, str(ckpt_dir))
    peft_model.eval()

    # Wrap into HFMathModel-like generate using peft model
    post_model = HFMathModel(
        args.model_path,
        device="cuda",
        torch_dtype="bfloat16",
        local_files_only=True,
        use_cache=True,
        max_new_tokens=args.max_new_tokens,
        temperature=1.0,
        top_p=0.95,
        do_sample=True,
    )
    # replace underlying model with peft-merged-for-generate path
    post_model.model = peft_model
    post_model.tokenizer = tok

    log(f"POST eval seed={post_seed} (base seed={args.seed})")
    post_eval = evaluate_split(
        post_model, confirm_recs, seed=post_seed, batch_size=args.eval_batch_size, log=log
    )
    results["post_eval"] = {k: post_eval[k] for k in ("n", "accuracy", "parse_rate", "mean_length", "wall_s")}
    results["post_per_problem"] = post_eval["per_problem"]

    base_acc = float(results.get("base_eval", {}).get("accuracy", 0.0))
    post_acc = float(post_eval["accuracy"])
    gain = (post_acc - base_acc) * 100.0
    results["accuracy_gain_pp"] = gain

    if results.get("base_per_problem") and post_eval["per_problem"]:
        paired = {
            pid: float(post_eval["per_problem"][pid]["correct"] - results["base_per_problem"][pid]["correct"])
            for pid in results["base_per_problem"]
            if pid in post_eval["per_problem"]
        }
        boot = task_cluster_bootstrap(paired, seed=args.seed)
        results["bootstrap_gain"] = {
            "mean_pp": boot["mean"] * 100.0,
            "ci_low_pp": boot["ci_low"] * 100.0,
            "ci_high_pp": boot["ci_high"] * 100.0,
            "n_tasks": boot["n_tasks"],
        }

    status = "PASS" if gain >= 2.0 else ("ITERATE" if gain > -2.0 else "FAIL")
    results["status"] = status
    results["gate_pass_ge_2pp"] = bool(gain >= 2.0)

    out = out_dir / "results.json"
    # drop bulky per_problem from main results file into side files
    slim = {k: v for k, v in results.items() if k not in ("base_per_problem", "post_per_problem")}
    out.write_text(json.dumps(slim, indent=2))
    (out_dir / "base_per_problem.json").write_text(json.dumps(results.get("base_per_problem", {}), indent=2))
    (out_dir / "post_per_problem.json").write_text(json.dumps(results.get("post_per_problem", {}), indent=2))

    md = out_dir / "report.md"
    md.write_text(
        "\n".join(
            [
                "# Q4 offline LoRA SFT on steered-teacher (confirm400 holdout)",
                "",
                f"- teachers correct: **{len(teachers)}**; no_thinking={args.no_thinking}",
                f"- train/val rows: {len(train_rows)}/{len(val_rows)}; epochs={args.epochs}; lora_r={args.lora_r}",
                f"- base confirm acc: **{base_acc*100:.2f}%**",
                f"- post confirm acc: **{post_acc*100:.2f}%**",
                f"- gain: **{gain:.2f} pp**",
                f"- status: **{status}** (gate >=2pp: {results['gate_pass_ge_2pp']})",
                f"- bootstrap CI: {results.get('bootstrap_gain')}",
                "",
            ]
        )
    )
    log(f"DONE gain_pp={gain:.2f} status={status} wrote {out}")


if __name__ == "__main__":
    main()
