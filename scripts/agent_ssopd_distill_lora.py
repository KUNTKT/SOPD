#!/usr/bin/env python3
"""A2: Offline LoRA SFT on agent teacher trajectories + valid_unseen eval."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from alfworld_common import (  # noqa: E402
    dump_json,
    ensure_alfworld_env,
    load_eval_tasks,
    load_jsonl,
    load_yaml_cfg,
    make_env_factory,
    trajectory_metrics,
)
from rollout.multistep import collect_episodes  # noqa: E402
from steerable_alfworld_agent import SteerableAlfworldAgent, build_agent  # noqa: E402


class AgentSFTDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int = 3072):
        self.rows = rows
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        prompt = str(r["prompt_text"])
        completion = str(r["completion_text"])
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
        return {
            "input_ids": input_ids,
            "attention_mask": enc_full["attention_mask"][0],
            "labels": labels,
        }


def collate(batch: list[dict]) -> dict:
    max_len = max(int(x["input_ids"].shape[0]) for x in batch)
    pad_id = 0

    def pad(t: torch.Tensor, fill: int) -> torch.Tensor:
        if t.shape[0] == max_len:
            return t
        out = torch.full((max_len,), fill, dtype=t.dtype)
        out[: t.shape[0]] = t
        return out

    return {
        "input_ids": torch.stack([pad(b["input_ids"], pad_id) for b in batch]),
        "attention_mask": torch.stack([pad(b["attention_mask"], 0) for b in batch]),
        "labels": torch.stack([pad(b["labels"], -100) for b in batch]),
    }


def train_lora(cfg: dict, teacher_jsonl: Path, out_adapter: Path) -> dict:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dcfg = cfg.get("distill", {})
    model_path = cfg["model"]["path"]
    rows = load_jsonl(teacher_jsonl)
    if not rows:
        raise SystemExit(f"no rows in {teacher_jsonl}")

    tok = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if cfg["model"].get("torch_dtype") == "bfloat16" else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="cuda",
        trust_remote_code=True,
        local_files_only=True,
    )
    # Start from coldstart adapter if present.
    adapter_path = cfg["model"].get("adapter_path")
    if adapter_path:
        from peft import PeftModel

        base = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
        base = base.merge_and_unload()
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(dcfg.get("lora_r", 16)),
        lora_alpha=int(dcfg.get("lora_alpha", 32)),
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(base, lora)
    model.train()
    ds = AgentSFTDataset(rows, tok, max_length=int(dcfg.get("max_length", 3072)))
    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=float(dcfg.get("lr", 1e-4)))
    epochs = int(dcfg.get("epochs", 2))
    t0 = time.time()
    n_steps = 0
    losses = []
    for ep in range(epochs):
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            n_steps += 1
            if n_steps % 20 == 0:
                print(f"  ep{ep} step{n_steps} loss={losses[-1]:.4f}", flush=True)
    out_adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_adapter))
    tok.save_pretrained(str(out_adapter))
    return {
        "n_rows": len(rows),
        "n_steps": n_steps,
        "mean_loss": sum(losses) / max(len(losses), 1),
        "wall_s": time.time() - t0,
        "adapter": str(out_adapter),
    }


def eval_adapter(cfg: dict, adapter: Path, out_jsonl: Path) -> dict:
    cfg = json.loads(json.dumps(cfg))  # deep-ish copy via json
    cfg.setdefault("model", {})["adapter_path"] = str(adapter)
    agent = build_agent(cfg)
    tasks, _ = load_eval_tasks(cfg)
    recs = collect_episodes(
        tasks,
        agent,
        make_env_factory(cfg),
        n_rollouts=1,
        dataset_split=str(cfg["env"].get("eval_split", "valid_unseen")),
        base_seed=int(cfg["rollout"]["seed"]) + 7,
        out_path=out_jsonl,
        resume=True,
        max_steps=int(cfg["rollout"]["max_steps"]),
        batch_size=int(cfg["rollout"].get("batch_size", 8)),
        n_env_workers=int(cfg["rollout"].get("env_workers", 4)),
    )
    agent.close()
    return trajectory_metrics(recs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "configs/experiment_agent_ssopd_alfworld.yaml"),
    )
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml_cfg(Path(args.config))
    ensure_alfworld_env(cfg)
    a1 = json.loads(Path(cfg["paths"]["reports_dir"], "a1_results.json").read_text())
    if not a1.get("analysis", {}).get("gate_pass"):
        raise SystemExit("A1 gate failed; refuse distill")

    collect_dir = Path(cfg["paths"]["reports_dir"]) / "a2_collect"
    out_dir = Path(cfg["paths"]["reports_dir"]) / "a2_distill"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, jsonl in (
        ("steered", collect_dir / "steered_correct.jsonl"),
        ("vanilla", collect_dir / "vanilla_correct.jsonl"),
    ):
        adapter = out_dir / f"lora_{name}"
        if not args.skip_train or not adapter.exists():
            print(f"train LoRA from {jsonl}", flush=True)
            results[f"train_{name}"] = train_lora(cfg, jsonl, adapter)
        print(f"eval {name}", flush=True)
        results[f"eval_{name}"] = eval_adapter(cfg, adapter, out_dir / f"eval_{name}.jsonl")
        print(
            f"  success={results[f'eval_{name}']['success_rate']:.3f}",
            flush=True,
        )

    # Coldstart baseline reuse a1 base if present
    base_m = a1["conditions"]["base"]["metrics"]
    steered_m = results["eval_steered"]
    vanilla_m = results["eval_vanilla"]
    gate = {
        "delta_vs_coldstart_pp": (steered_m["success_rate"] - base_m["success_rate"]) * 100,
        "delta_vs_vanilla_sft_pp": (steered_m["success_rate"] - vanilla_m["success_rate"]) * 100,
        "gate_pass": (
            (steered_m["success_rate"] - base_m["success_rate"]) * 100 >= 3.0
            and steered_m["success_rate"] >= vanilla_m["success_rate"]
        ),
    }
    payload = {"results": results, "analysis": gate, "base_success": base_m["success_rate"]}
    dump_json(out_dir / "results.json", payload)
    dump_json(Path(cfg["paths"]["reports_dir"]) / "a2_results.json", payload)
    print("A2_DONE", json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
