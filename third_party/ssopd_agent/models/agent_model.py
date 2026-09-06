"""Thin HF generation wrapper. No activation hooks or steering."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

DEFAULT_SYSTEM_PROMPT = (
    "You are a tool-using assistant. Think briefly (under 40 words), then "
    "call exactly one listed tool using the <tool> and <args> format. "
    "Do not give the final answer yourself."
)


@dataclass
class GenerateResult:
    text: str
    prompt_token_count: int
    completion_token_count: int
    full_text: str


class AgentLike(Protocol):
    def generate(self, observation: str, seed: int | None = None) -> GenerateResult: ...

    def close(self) -> None: ...


class MockAgent:
    """Deterministic stand-in for CPU tests."""

    def __init__(self, mode: str = "correct", tool_names: list[str] | None = None, seed: int = 0):
        self.mode = mode
        self.tool_names = tool_names or []
        self.seed = seed
        self._n = 0
        self._env = None

    def bind_env(self, env: Any) -> None:
        self._env = env

    def generate(self, observation: str, seed: int | None = None) -> GenerateResult:
        self._n += 1
        if self.mode == "correct":
            gold = "calculator"
            gold_args: dict[str, Any] = {}
            if self._env is not None and getattr(self._env, "task", None):
                step = self._env.task["steps"][self._env.step_idx]
                gold = step["gold_tool"]
                gold_args = dict(step.get("gold_args") or {})
            else:
                for line in observation.splitlines():
                    if line.startswith("GOLD_TOOL:"):
                        gold = line.split(":", 1)[1].strip()
            args_txt = json.dumps(gold_args) if gold_args else "{}"
            text = f"<tool>{gold}</tool>\n<args>{args_txt}</args>"
        elif self.mode == "random":
            import random

            rng = random.Random(seed if seed is not None else self.seed + self._n)
            names = self.tool_names or [
                "calculator",
                "search",
                "weather",
                "translator",
                "calendar",
                "file_read",
                "email_send",
                "unit_convert",
            ]
            name = rng.choice(names)
            text = f"<tool>{name}</tool>\n<args>{{}}</args>"
        elif self.mode == "garbage":
            text = "I think the answer is 42."
        else:
            raise ValueError(self.mode)
        return GenerateResult(
            text=text,
            prompt_token_count=max(1, len(observation.split())),
            completion_token_count=max(1, len(text.split())),
            full_text=observation + "\n" + text,
        )

    def close(self) -> None:
        return None

    def generate_batch(self, observations: list[str], seed: int | None = None) -> list[GenerateResult]:
        return [
            self.generate(obs, seed=None if seed is None else int(seed) + i)
            for i, obs in enumerate(observations)
        ]


class HFAgent:
    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 384,
        temperature: float = 0.8,
        top_p: float = 0.95,
        do_sample: bool = True,
        system_prompt: str | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = model_path
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.do_sample = bool(do_sample)
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.device = device
        if self.model.generation_config is not None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self._steer_hook = None
        self._steer_spec = None

    def set_steering(
        self,
        layer: int,
        vector,
        alpha: float,
        *,
        mode: str = "unit_additive",
    ) -> None:
        """Attach or replace a single-layer residual steering hook."""
        import torch

        from ssopd_agent.representation.steering import SteeringSpec, attach_steering_hook, remove_hook

        remove_hook(self._steer_hook)
        self._steer_hook = None
        self._steer_spec = None
        if abs(float(alpha)) < 1e-12:
            return
        if not isinstance(vector, torch.Tensor):
            vector = torch.tensor(vector, dtype=torch.float32)
        spec = SteeringSpec(
            layer=int(layer),
            alpha=float(alpha),
            vector=vector,
            mode=mode,  # type: ignore[arg-type]
        )
        self._steer_hook = attach_steering_hook(self.model, spec)
        self._steer_spec = {
            "layer": int(layer),
            "alpha": float(alpha),
            "mode": mode,
            "vector_l2": float(vector.float().norm().item()),
        }

    def clear_steering(self) -> None:
        from ssopd_agent.representation.steering import remove_hook

        remove_hook(self._steer_hook)
        self._steer_hook = None
        self._steer_spec = None

    def _prompt_text(self, observation: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": observation},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, observation: str, seed: int | None = None) -> GenerateResult:
        prompt = self._prompt_text(observation)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        prompt_len = int(encoded["input_ids"].shape[1])
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p
        import torch

        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        with torch.inference_mode():
            out = self.model.generate(**encoded, **gen_kwargs)
        completion_ids = out[0, prompt_len:]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return GenerateResult(
            text=text,
            prompt_token_count=prompt_len,
            completion_token_count=int(completion_ids.shape[0]),
            full_text=prompt + text,
        )

    def generate_batch(self, observations: list[str], seed: int | None = None) -> list[GenerateResult]:
        if not observations:
            return []
        import torch

        prompts = [self._prompt_text(o) for o in observations]
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True)
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        padded_len = int(encoded["input_ids"].shape[1])
        prompt_lens = encoded["attention_mask"].sum(dim=1).tolist()
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        with torch.inference_mode():
            out = self.model.generate(**encoded, **gen_kwargs)
        results: list[GenerateResult] = []
        for i, seq in enumerate(out):
            completion_ids = seq[padded_len:]
            text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            results.append(
                GenerateResult(
                    text=text,
                    prompt_token_count=int(prompt_lens[i]),
                    completion_token_count=int(completion_ids.shape[0]),
                    full_text=prompts[i] + text,
                )
            )
        return results

    def close(self) -> None:
        import torch

        self.clear_steering()
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
